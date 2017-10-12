#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"
#include <curand.h>
#include <curand_kernel.h>

EXTERN_C
{
	EXPORT int _Initialize(CMemoryBuffer buf, const double value);

	EXPORT int _LinSpace(CMemoryBuffer buf, const double x0, const double x1);

	EXPORT int _RandUniform(CMemoryBuffer buf, const unsigned seed);

	EXPORT int _RandNormal(CMemoryBuffer buf, const unsigned seed);
}

#define CURAND_STATE_PTR curandState* const RESTRICT

template <typename T>
GLOBAL void __Initialize__(T* RESTRICT ptr, const ptr_t sz, const T value);

template <typename T>
GLOBAL void __LinSpace__(T* RESTRICT ptr, const ptr_t sz, const T x0, const T dx);

GLOBAL void __SetupCuRand__(CURAND_STATE_PTR state, const ptr_t sz, const unsigned seed);

template <typename T>
GLOBAL void __RandUniform__(T* RESTRICT ptr, CURAND_STATE_PTR state, const ptr_t sz);

template <typename T>
GLOBAL void __RandNormal__(T* RESTRICT ptr, CURAND_STATE_PTR state, const ptr_t sz);
