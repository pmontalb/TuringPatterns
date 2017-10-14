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

	EXPORT int _Iterate2D(CMemoryTile uNew, const CMemoryTile u, const CMemoryBuffer xGrid, const CMemoryBuffer yGrid, const double dt, const double diffusionCoefficient, const EBoundaryCondition boundaryConditionType)
	{
		dim3 blocks(1, 1);
		dim3 threads(1, 1);
		switch (uNew.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_XY(__Iterate2D__<float>, blocks, threads, (float*)uNew.pointer, (float*)u.pointer, (float*)xGrid.pointer, (float*)yGrid.pointer, uNew.nRows, uNew.nCols, (float)dt, (float)diffusionCoefficient, boundaryConditionType);
			break;
		case EMathDomain::Double:
			CUDA_CALL_XY(__Iterate2D__<double>, blocks, threads, (double*)uNew.pointer, (double*)u.pointer, (double*)xGrid.pointer, (double*)yGrid.pointer, uNew.nRows, uNew.nCols, dt, diffusionCoefficient, boundaryConditionType);
			break;
		default:
			break;
		}

		return cudaGetLastError();
	}
}

#define COORD(X, Y) (X) + nRows * (Y)

template <typename T>
GLOBAL void __Iterate1D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT grid, const size_t sz, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType)
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
		uxMinus = 2.0 / (dxMinus * dx);
		uxPlus = 2.0  / (dxPlus  * dx);
		ux0 = -uxMinus - uxPlus;

		uxx = diffusionCoefficient * (uxMinus * u[i - 1] + ux0 * u[i] + uxPlus * u[i + 1]);

		uNew[i] = u[i] + dt * uxx;
	}
}

template <typename T>
GLOBAL void __Iterate2D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT xGrid, const T* RESTRICT yGrid, const size_t nRows, const size_t nCols, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType)
{
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stepX = gridDim.x * blockDim.x;

	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int stepY = gridDim.y * blockDim.y;

	// first thread x deals with boundary condition
	if (tidX == 0)
	{
		switch (boundaryConditionType)
		{
		case EBoundaryCondition::ZeroFlux:
			for (size_t j = 1; j < nCols - 1; j++)
			{
				uNew[COORD(0, j)] = uNew[COORD(1, j)];
				uNew[COORD(nRows - 1, j)] = uNew[COORD(nRows - 2, j)];
			}
			break;
		case EBoundaryCondition::Periodic:
			for (size_t j = 1; j < nCols - 1; j++)
			{
				uNew[COORD(0, j)] = uNew[COORD(nRows - 2, j)];
				uNew[COORD(nRows - 1, j)] = uNew[COORD(1, j)];
			}
			break;
		default:
			break;
		}
	}

	// first thread y deals with boundary condition
	if (tidY == 0)
	{
		switch (boundaryConditionType)
		{
		case EBoundaryCondition::ZeroFlux:
			for (size_t i = 1; i < nRows - 1; i++)
			{
				uNew[COORD(i, 0)] = uNew[COORD(i, 1)];
				uNew[COORD(i, nCols - 1)] = uNew[COORD(i, nCols - 2)];
			}
			break;
		case EBoundaryCondition::Periodic:
			for (size_t i = 1; i < nRows - 1; i++)
			{
				uNew[COORD(i, 0)] = uNew[COORD(i, nCols - 2)];
				uNew[COORD(i, nCols - 1)] = uNew[COORD(i, 1)];
			}
			break;
		default:
			break;
		}
	}

	// first thread x and y deals with corners
	if (tidX * tidY == 0)
	{
		switch (boundaryConditionType)
		{
		case EBoundaryCondition::ZeroFlux:
				uNew[COORD(0, 0)] = .5 * (uNew[COORD(0, 1)] + uNew[COORD(1, 0)]);
				uNew[COORD(nRows - 1, 0)] = .5 * (uNew[COORD(nRows - 2, 0)] + uNew[COORD(nRows - 1, 1)]);
				uNew[COORD(0, nCols - 1)] = .5 * (uNew[COORD(1, nCols - 1)] + uNew[COORD(0, nCols - 2)]);
				uNew[COORD(nRows - 1, nCols - 1)] = .5 * (uNew[COORD(nRows - 2, nCols - 1)] + uNew[COORD(nRows - 1, nCols - 2)]);
			break;
		case EBoundaryCondition::Periodic:
				uNew[COORD(0, 0)] = .5 * (uNew[COORD(0, nCols - 2)] + uNew[COORD(nRows - 2, 0)]);
				uNew[COORD(nRows - 1, 0)] = .5 * (uNew[COORD(1, 0)] + uNew[COORD(nRows - 1, nCols - 2)]);
				uNew[COORD(0, nCols - 1)] = .5 * (uNew[COORD(0, 1)] + uNew[COORD(nRows - 2, nCols - 1)]);
				uNew[COORD(nRows - 1, nCols - 1)] = .5 * (uNew[COORD(1, nCols - 1)] + uNew[COORD(nRows - 1, 1)]);
			break;
		default:
			break;
		}
	}

	T dx = 0.0, dxPlus = 0.0, dxMinus = 0.0;
	T uxMinus, uxPlus = 0.0, ux0 = 0.0, uxx = 0.0;

	T dy = 0.0, dyPlus = 0.0, dyMinus = 0.0;
	T uyMinus, uyPlus = 0.0, uy0 = 0.0, uyy = 0.0;
	T laplacian = 0.0;
	for (size_t j = tidY + 1; j < nCols - 1; j += stepY)
	{
		// Y discretization
		dyPlus = yGrid[j + 1] - yGrid[j];
		dyMinus = yGrid[j] - yGrid[j - 1];
		dy = dyPlus + dyMinus;

		uyMinus = 2.0 / (dyMinus * dy);
		uyPlus = 2.0 / (dyPlus  * dy);
		uy0 = -uyMinus - uyPlus;

		for (size_t i = tidX + 1; i < nRows - 1; i += stepX)
		{
			// X discretization
			dxPlus = xGrid[i + 1] - xGrid[i];
			dxMinus = xGrid[i] - xGrid[i - 1];
			dx = dxPlus + dxMinus;

			uxMinus = 2.0 / (dxMinus * dx);
			uxPlus = 2.0 / (dxPlus  * dx);
			ux0 = -uxMinus - uxPlus;

			uxx = (uxMinus * u[COORD(i - 1, j)] + ux0 * u[COORD(i, j)] + uxPlus * u[COORD(i + 1, j)]);
			uyy = (uyMinus * u[COORD(i, j - 1)] + uy0 * u[COORD(i, j)] + uyPlus * u[COORD(i, j + 1)]);

			laplacian = diffusionCoefficient * (uxx + uyy);

			uNew[COORD(i, j)] = u[COORD(i, j)] + dt * laplacian;
		}
	}
}