#include "FiniteDifference.cuh"

#pragma once

EXTERN_C
{
	EXPORT int _Iterate1D(CMemoryBuffer uNew, const CMemoryBuffer u, const CMemoryBuffer grid, const double dt, const double diffusionCoefficient, const EBoundaryCondition boundaryConditionType)
	{
		switch (uNew.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_XY(__Iterate1D__<float>, 32, 8, (float*)uNew.pointer, (float*)u.pointer, (float*)grid.pointer, uNew.size, (float)dt, (float)diffusionCoefficient, boundaryConditionType);
			break;
		case EMathDomain::Double:
			CUDA_CALL_XY(__Iterate1D__<double>, 32, 8, (double*)uNew.pointer, (double*)u.pointer, (double*)grid.pointer, uNew.size, dt, diffusionCoefficient, boundaryConditionType);
			break;
		default:
			break;
		}

		return cudaGetLastError();
	}
}

template <typename T>
GLOBAL void __Iterate1D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT grid, const ptr_t sz, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int step = gridDim.x * blockDim.x;

	// first thread deals with boundary condition
	if (tid == 0)
	{
		switch (boundaryConditionType)
		{
		case EBoundaryCondition::ZeroFlux:
			uNew[0] = uNew[1];
			uNew[sz - 1] = uNew[sz - 2];
			break;
		case EBoundaryCondition::Periodic:
			uNew[0] = uNew[sz - 2];
			uNew[sz - 1] = uNew[1];
			break;
		default:
			break;
		}
	}

	T dx = 0.0, dxPlus = 0.0, dxMinus = 0.0;
	T uxMinus, uxPlus = 0.0, ux0 = 0.0, uxx = 0.0;
	for (size_t i = tid + 1; i < sz - 1; i += step)\
	{
		dxPlus = grid[i + 1] - grid[i];
		dxMinus = grid[i] - grid[i - 1];
		dx = dxPlus + dxMinus;

		// space discretization
		uxMinus = 2.0 * diffusionCoefficient / (dxMinus * dx);
		uxPlus = 2.0 * diffusionCoefficient / (dxPlus  * dx);
		ux0 = -uxMinus - uxPlus;

		uxx = diffusionCoefficient * (uxMinus * u[i - 1] + ux0 * u[i] + uxPlus * u[i + 1]);

		uNew[i] = u[i] + dt * uxx;
	}
}