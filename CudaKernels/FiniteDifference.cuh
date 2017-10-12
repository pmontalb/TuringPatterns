#pragma once

#include "Common.cuh"
#include "Flags.cuh"

#include "Types.h"

#pragma once

EXTERN_C
{
	EXPORT int _Iterate1D(CMemoryBuffer uNew, const CMemoryBuffer u, const CMemoryBuffer grid, const double dt, const double diffusionCoefficient, const EBoundaryCondition boundaryConditionType);
}

template <typename T>
GLOBAL void __Iterate1D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT grid, const ptr_t sz, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType);