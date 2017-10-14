#pragma once

#include "Common.cuh"
#include "Flags.cuh"

#include "Types.h"

#pragma once

EXTERN_C
{
	EXPORT int _Iterate1D(CMemoryBuffer uNew, const CMemoryBuffer u, const CMemoryBuffer grid, const double dt, const double diffusionCoefficient, const EBoundaryCondition boundaryConditionType);
	EXPORT int _Iterate2D(CMemoryTile uNew, const CMemoryTile u, const CMemoryBuffer xGrid, const CMemoryBuffer yGrid, const double dt, const double diffusionCoefficient, const EBoundaryCondition boundaryConditionType);
}

template <typename T>
GLOBAL void __Iterate1D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT grid, const size_t sz, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType);

template <typename T>
GLOBAL void __Iterate2D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT xGrid, const T* RESTRICT yGrid, const size_t nRows, const size_t nCols, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType);