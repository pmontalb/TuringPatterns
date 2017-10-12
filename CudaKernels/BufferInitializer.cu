#include "BufferInitializer.cuh"
#include "DeviceManager.cuh"
#include <stdio.h>

extern "C"
{
	EXPORT int _Initialize(CMemoryBuffer buf, const double value)
	{
		switch (buf.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_SINGLE(__Initialize__<float>, (float*)buf.pointer, buf.size, (float)value);
			break;
		case EMathDomain::Double:
			CUDA_CALL_DOUBLE(__Initialize__<double>, (double*)buf.pointer, buf.size, value);
			break;
		default:
			break;
		}

		return cudaGetLastError();
	}

	EXPORT int _LinSpace(CMemoryBuffer buf, const double x0, const double x1)
	{
		const double dx = (x1 - x0) / (buf.size - 1);

		switch (buf.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_SINGLE(__LinSpace__<float>, (float*)buf.pointer, buf.size, (float)x0, (float)dx);
			break;
		case EMathDomain::Double:
			CUDA_CALL_DOUBLE(__LinSpace__<double>, (double*)buf.pointer, buf.size, (double)x0, (double)dx);
			break;
		default:
			break;
		}

		return cudaGetLastError();
	}

	EXPORT int _RandUniform(CMemoryBuffer buf, const unsigned seed)
	{
		if (buf.size & 1)
			return -1;

		dim3 block, grid;
		ptr_t halfSz = buf.size >> 1;
		GetBestDimension(block, grid, N_BLOCKS_SINGLE, halfSz);

		curandState *states = 0;
		if (cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState)))
			return -1;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, halfSz, seed);

		switch (buf.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_XYZ(__RandUniform__<float>, grid, block, block.x * sizeof(unsigned int), (float*)buf.pointer, states, halfSz);
			break;
		case EMathDomain::Double:
			CUDA_CALL_XYZ(__RandUniform__<double>, grid, block, block.x * sizeof(unsigned int), (double*)buf.pointer, states, halfSz);
			break;
		default:
			break;
		}

		cudaFree(states);

		return cudaGetLastError();
	}

	EXPORT int _RandNormal(CMemoryBuffer buf, const unsigned seed)
	{
		if (buf.size & 1)
			return -1;

		dim3 block, grid;
		ptr_t halfSz = buf.size >> 1;
		GetBestDimension(block, grid, N_BLOCKS_SINGLE, halfSz);

		curandState *states = 0;
		if (cudaMalloc((void **)&states, grid.x * block.x * sizeof(curandState)))
			return -1;
		CUDA_CALL_XY(__SetupCuRand__, grid, block, states, halfSz, seed);

		switch (buf.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_XYZ(__RandNormal__<float>, grid, block, block.x * sizeof(unsigned int), (float*)buf.pointer, states, halfSz);
			break;
		case EMathDomain::Double:
			CUDA_CALL_XYZ(__RandNormal__<double>, grid, block, block.x * sizeof(unsigned int), (double*)buf.pointer, states, halfSz);
			break;
		default:
			break;
		}

		cudaFree(states);

		return cudaGetLastError();
	}
}

template <typename T>
GLOBAL void __Initialize__(T* RESTRICT ptr, const ptr_t sz, const T value)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		ptr[i] = value;

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __LinSpace__(T* RESTRICT ptr, const ptr_t sz, const T x0, const T dx)
{
	CUDA_FUNCTION_PROLOGUE
	CUDA_FOR_LOOP_PROLOGUE

		ptr[i] = x0 + i * dx;

	CUDA_FOR_LOOP_EPILOGUE
}

GLOBAL void __SetupCuRand__(CURAND_STATE_PTR states, const ptr_t sz, const unsigned seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialise the RNG
	curand_init(seed, tid, 0, &states[tid]);
}

template <typename T>
GLOBAL void __RandUniform__(T* RESTRICT ptr, CURAND_STATE_PTR states, const ptr_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	curandState localState = states[tid];

	CUDA_FOR_LOOP_PROLOGUE

		ptr[2 * i] = curand_uniform(&localState);
		ptr[2 * i + 1] = 1.0f - ptr[2 * i];

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RandNormal__(T* RESTRICT ptr, CURAND_STATE_PTR states, const ptr_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	curandState localState = states[tid];

	CUDA_FOR_LOOP_PROLOGUE

		ptr[2 * i] = curand_normal(&localState);
		ptr[2 * i + 1] = -ptr[2 * i];

	CUDA_FOR_LOOP_EPILOGUE
}