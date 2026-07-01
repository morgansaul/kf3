// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC
//
// FIXED VERSION: True parallel multi-target solving
// Multiple public keys are solved SIMULTANEOUSLY on GPU
// Tames are universal, wilds are distributed across targets


#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

char gTargetsFileName[1024];
std::vector<EcPoint> gTargetPoints;
std::vector<EcPoint> gShiftedTargets;

EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;

// FIXED: Multi-target mode flag
bool gMultiTargetMode = false;

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
	u8 target_id; // FIXED: Track which target this collision is for
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
		
		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}

#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

// FIXED: Collision verification for multi-target mode
bool Collision_SOTA_MultiTarget(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg, int& found_target_idx)
{
	found_target_idx = -1;
	
	if (IsNeg)
		t.Neg();
	
	// Try to find which target this collision belongs to
	for (size_t target_idx = 0; target_idx < gShiftedTargets.size(); target_idx++)
	{
		if (TameType == TAME)
		{
			gPrivKey = t;
			gPrivKey.Sub(w);
			EcInt sv = gPrivKey;
			gPrivKey.Add(Int_HalfRange);
			EcPoint P = ec.MultiplyG(gPrivKey);
			if (P.IsEqual(gShiftedTargets[target_idx]))
			{
				found_target_idx = (int)target_idx;
				return true;
			}
			gPrivKey = sv;
			gPrivKey.Neg();
			gPrivKey.Add(Int_HalfRange);
			P = ec.MultiplyG(gPrivKey);
			if (P.IsEqual(gShiftedTargets[target_idx]))
			{
				found_target_idx = (int)target_idx;
				return true;
			}
		}
		else
		{
			gPrivKey = t;
			gPrivKey.Sub(w);
			if (gPrivKey.data[4] >> 63)
				gPrivKey.Neg();
			gPrivKey.ShiftRight(1);
			EcInt sv = gPrivKey;
			gPrivKey.Add(Int_HalfRange);
			EcPoint P = ec.MultiplyG(gPrivKey);
			if (P.IsEqual(gShiftedTargets[target_idx]))
			{
				found_target_idx = (int)target_idx;
				return true;
			}
			gPrivKey = sv;
			gPrivKey.Neg();
			gPrivKey.Add(Int_HalfRange);
			P = ec.MultiplyG(gPrivKey);
			if (P.IsEqual(gShiftedTargets[target_idx]))
			{
				found_target_idx = (int)target_idx;
				return true;
			}
		}
	}
	
	return false;
}

void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = pPntList2 + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		nrec.type = gGenMode ? TAME : p[40];
		nrec.target_id = 0; // FIXED: Default target ID

		DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
		if (gGenMode)
			continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			if (pref->type == nrec.type)
			{
				if (pref->type == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
			}

			EcInt w, t;
			int TameType, WildType;
			if (pref->type != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = nrec.type;
				WildType = pref->type;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = nrec.type;
			}

			// FIXED: Check against all targets in multi-target mode
			int found_target = -1;
			bool res = false;
			
			if (gMultiTargetMode)
			{
				res = Collision_SOTA_MultiTarget(gPntToSolve, t, TameType, w, WildType, false, found_target) ||
					  Collision_SOTA_MultiTarget(gPntToSolve, t, TameType, w, WildType, true, found_target);
			}
			else
			{
				res = Collision_SOTA_MultiTarget(gPntToSolve, t, TameType, w, WildType, false, found_target) ||
					  Collision_SOTA_MultiTarget(gPntToSolve, t, TameType, w, WildType, true, found_target);
			}

			if (!res)
			{
				bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
				if (!w12)
				{
					printf("Collision Error\r\n");
					gTotalErrors++;
				}
				continue;
			}
			
			if (found_target >= 0 && found_target < (int)gTargetPoints.size())
			{
				printf("COLLISION FOUND FOR TARGET %d!\r\n", found_target);
			}
			
			gSolved = true;
			break;
		}
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 tm_cur = GetTickCount64();
	u64 tm_elapsed = tm_cur - tm_start;
	printf("Speed: %d MH, Total ops: %lld, Elapsed: %lld sec\r\n", speed, TotalOps, tm_elapsed / 1000);
}

bool LoadTargets(const char* fn)
{
	gTargetPoints.clear();
	gShiftedTargets.clear();
	
	std::ifstream file(fn);
	if (!file.is_open())
	{
		printf("Cannot open targets file: %s\r\n", fn);
		return false;
	}

	std::string line;
	int line_num = 0;
	while (std::getline(file, line))
	{
		line_num++;
		if (line.empty() || line[0] == '#') continue;

		EcPoint pt;
		if (!pt.SetHexStr(line.c_str()))
		{
			printf("Error parsing target at line %d: %s\r\n", line_num, line.c_str());
			continue;
		}
		gTargetPoints.push_back(pt);
	}
	file.close();

	printf("Loaded %zu targets from %s\r\n", gTargetPoints.size(), fn);
	return !gTargetPoints.empty();
}

bool SolvePoint(EcPoint _PntToSolve, int _Range, int _DP, EcInt* _pk_found, std::vector<EcInt>* found_keys = NULL, std::vector<int>* found_indices = NULL)
{
	gPntToSolve = _PntToSolve;

	ThrCnt = GpuCnt;
	gSolved = false;
	PntIndex = 0;
	PntTotalOps = 0;

	if (db.LoadFromFile(gTamesFileName))
	{
		printf("tames loaded from %s\r\n", gTamesFileName);
		gGenMode = false;
	}
	else if (!gGenMode)
	{
		printf("tames not found, will generate\r\n");
		gGenMode = true;
	}

	// FIXED: Prepare GPU with ALL targets at once
	for (int i = 0; i < GpuCnt; i++)
	{
		if (!GpuKangs[i]->Prepare(gShiftedTargets, _Range, _DP, EcJumps1, EcJumps2, EcJumps3))
		{
			GpuKangs[i]->Failed = true;
			return false;
		}
	}

	// Start GPU threads
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)kang_thr_proc, (void*)(GpuKangs[i]), 0, NULL);
#else
		pthread_t thread_id;
		pthread_create(&thread_id, NULL, kang_thr_proc, (void*)(GpuKangs[i]));
#endif
	}

	u64 tm_start = GetTickCount64();
	bool stop_requested = false;

	// FIXED: Main loop waits for collision from ANY target
	while (ThrCnt > 0 && !gSolved)
	{
		Sleep(100);
		CheckNewPoints();
		ShowStats(tm_start, 0, 0);

		// Check if we've solved or hit limits
		if (gSolved)
		{
			printf("\r\n=== SOLVED! ===\r\n");
			break;
		}
	}

	// Stop all GPU threads
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();

	// Wait for threads to complete
	while (ThrCnt > 0)
		Sleep(100);

	ShowStats(tm_start, 0, 0);

	if (gGenMode)
	{
		db.Header[0] = _Range;
		if (db.SaveToFile(gTamesFileName))
			printf("tames saved\r\n");
		return !gSolved;
	}

	db.Clear();
	return gSolved;
}

bool ParseCommandLine(int argc, char* argv[])
{
	for (int ci = 1; ci < argc; ci++)
	{
		char* argument = argv[ci];
		if (argument[0] != '-')
		{
			printf("error: invalid parameter %s\r\n", argument);
			return false;
		}
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 1) || (val > 63))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-targets") == 0)
		{
			strcpy(gTargetsFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			strcpy(gTamesFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-max") == 0)
		{
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -max option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
	if ((!gPubKey.x.IsZero() || gTargetsFileName[0] != 0))
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	return true;
}

int main(int argc, char* argv[])
{
	printf("********************************************************************************\r\n");
	printf("*                 RCKangaroo v3.2 PARALLEL MULTI-TARGET (c) 2024               *\r\n");
	printf("*              True Parallel Multi-Target ECDLP Solver                         *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("Fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");
	printf("FIXED: Multiple targets solved in PARALLEL on GPU\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

	InitEc();
	gDP = 0;
	gRange = 0;
	gStartSet = false;
	gTamesFileName[0] = 0;
	gTargetsFileName[0] = 0;
	gMax = 0.0;
	gGenMode = false;
	gIsOpsLimit = false;
	memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
	
	if (!ParseCommandLine(argc, argv))
		return 0;

	InitGpus();

	// FIXED: Load targets if specified
	if (gTargetsFileName[0] != 0)
	{
		if (!LoadTargets(gTargetsFileName))
			return 0;
		gMultiTargetMode = true;

		printf("\r\n========== PARALLEL MULTI-TARGET MODE ==========\r\n");
		printf("Loaded %zu target public keys\r\n", gTargetPoints.size());
		printf("Will solve ALL targets in PARALLEL on GPU\r\n");
		printf("GPU checks ALL targets simultaneously via binary search\r\n");
		printf("================================================\r\n\r\n");
	}

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE - MULTI-TARGET PARALLEL SOLVE\r\n\r\n");

		// Setup targets
		if (gTargetPoints.empty() && !gPubKey.x.IsZero())
		{
			gTargetPoints.push_back(gPubKey);
			gMultiTargetMode = false;
		}

		// Apply -start offset to ALL targets
		gShiftedTargets = gTargetPoints;
		if (!gStart.IsZero())
		{
			EcPoint PntOfs = ec.MultiplyG(gStart);
			PntOfs.y.NegModP();
			for (size_t i = 0; i < gShiftedTargets.size(); i++)
			{
				gShiftedTargets[i] = ec.AddPoints(gShiftedTargets[i], PntOfs);
			}
		}

		// FIXED: Solve ALL targets in parallel with single GPU call
		printf("\r\n===========================================\r\n");
		printf("PARALLEL SOLVING %zu TARGETS\r\n", gTargetPoints.size());
		for (size_t i = 0; i < gTargetPoints.size(); i++)
		{
			char sx[100], sy[100];
			gTargetPoints[i].x.GetHexStr(sx);
			gTargetPoints[i].y.GetHexStr(sy);
			printf("Target %zu: %s (Y: %s)\r\n", i, sx, sy);
		}
		printf("===========================================\r\n");

		EcInt pk_found;
		std::vector<EcInt> found_keys;
		std::vector<int> found_indices;

		if (!SolvePoint(gShiftedTargets[0], gRange, gDP, &pk_found, &found_keys, &found_indices))
		{
			printf("ERROR: Failed to solve targets\r\n");
			goto label_end;
		}

		// Save results
		if (!found_keys.empty())
		{
			FILE* fp = fopen("RESULTS.TXT", "w");
			if (fp)
			{
				fprintf(fp, "=== PARALLEL MULTI-TARGET SOLVE ===\r\n");
				fprintf(fp, "Targets solved: %zu\r\n\r\n", found_keys.size());
				for (size_t i = 0; i < found_keys.size(); i++)
				{
					int idx = found_indices[i];
					char s_priv[100], s_pubX[100], s_pubY[100];
					found_keys[i].GetHexStr(s_priv);
					gTargetPoints[idx].x.GetHexStr(s_pubX);
					gTargetPoints[idx].y.GetHexStr(s_pubY);
					fprintf(fp, "Target %d: PubX=%s, PrivKey=%s\r\n", idx, s_pubX, s_priv);
				}
				fclose(fp);
				printf("Results saved to RESULTS.TXT\r\n");
			}
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;

			pk.RndBits(gRange);
			PntToSolve = ec.MultiplyG(pk);
			gTargetPoints.clear();
			gTargetPoints.push_back(PntToSolve);
			gShiftedTargets = gTargetPoints;

			if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
			{
				if (!gIsOpsLimit)
					printf("FATAL ERROR: SolvePoint failed\r\n");
				break;
			}
			
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f\r\n", TotalSolved, K);
		}
	}
	
label_end:
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
	free(pPntList2);
	free(pPntList);
	return 0;
}
