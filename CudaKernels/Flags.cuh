#define EXPORT __declspec(dllexport)

#ifdef __CUDACC__
	#define HOST __host__  // call from host
	#define DEVICE __device__  // call from device
	#define GLOBAL __global__ // call from both, run on device
#else
	#define HOST
	#define DEVICE
	#define GLOBAL
#endif 

#define HOST_DEVICE HOST DEVICE
#define EXPORT_HOST EXPORT HOST
#define EXPORT_DEVICE EXPORT DEVICE
#define EXPORT_HOST_DEVICE EXPORT HOST DEVICE

#define RESTRICT __restrict__
#define FLOAT_PTR float* RESTRICT

#ifndef FORCE_32_BIT
	typedef size_t ptr_t;
#else
	typedef unsigned ptr_t;
#endif

#define DOUBLE_PTR FLOAT_PTR

#define N_BLOCKS_SINGLE 32
#define N_THREAD_PER_BLOCK_SINGLE 512
#define N_THREADS_SINGLE (N_BLOCKS_SINGLE * N_THREAD_PER_BLOCK_SINGLE)

#define N_BLOCKS_DOUBLE 16
#define N_THREAD_PER_BLOCK_DOUBLE 256
#define N_THREADS_DOUBLE (N_BLOCKS_DOUBLE * N_THREAD_PER_BLOCK_DOUBLE)

#define CUDA_CALL_DOUBLE(F, ...) F<<<N_BLOCKS_DOUBLE, N_THREAD_PER_BLOCK_DOUBLE>>>(__VA_ARGS__);

#define CUDA_CALL_SINGLE(F, ...) F<<<N_BLOCKS_SINGLE, N_THREAD_PER_BLOCK_SINGLE>>>(__VA_ARGS__);
#define CUDA_CALL_XY(F, GRID, BLOCK, ...) F<<<GRID, BLOCK>>>(__VA_ARGS__);
#define CUDA_CALL_XYZ(F, GRID, BLOCK, Z, ...) F<<<GRID, BLOCK, Z>>>(__VA_ARGS__);

#define CUDA_FUNCTION_PROLOGUE\
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;\
	unsigned int step = gridDim.x * blockDim.x;

#define CUDA_FOR_LOOP_PROLOGUE\
	for (size_t i = tid; i < sz; i += step)\
	{

#define CUDA_FOR_LOOP_EPILOGUE\
	}