#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include <cublas_v2.h>

static int currentDevice;

EXTERN_C
{
	EXPORT int _GetDevice(int& dev);

	EXPORT int _GetDeviceCount(int& count);

	EXPORT int _ThreadSynchronize();

	EXPORT int _SetDevice(const int dev);

	EXPORT int _GetDeviceStatus();

	EXPORT int _GetBestDevice(int& dev);

	EXPORT int _GetDeviceProperties(cudaDeviceProp& prop, const int dev);
}

const cublasHandle_t& CublasHandle();

void GetBestDimension(dim3& block, dim3& grid, const ptr_t nBlocks, const ptr_t problemDimension);