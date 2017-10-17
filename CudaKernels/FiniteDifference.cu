#include "FiniteDifference.cuh"
#include "DeviceManager.cuh"

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
		dim3 grid(32, 32);
		dim3 blocks(8, 8);
		switch (uNew.mathDomain)
		{
		case EMathDomain::Float:
			CUDA_CALL_XY(__Iterate2D__<float>, grid, blocks, (float*)uNew.pointer, (float*)u.pointer, (float*)xGrid.pointer, (float*)yGrid.pointer, uNew.nRows, uNew.nCols, (float)dt, (float)diffusionCoefficient, boundaryConditionType);
			break;
		case EMathDomain::Double:
			CUDA_CALL_XY(__Iterate2D__<double>, grid, blocks, (double*)uNew.pointer, (double*)u.pointer, (double*)xGrid.pointer, (double*)yGrid.pointer, uNew.nRows, uNew.nCols, dt, diffusionCoefficient, boundaryConditionType);
			break;
		default:
			break;
		}

		return cudaGetLastError();
	}

	EXPORT int _Iterate2DPattern(CPatternInput2D input)
	{
		dim3 grid(32, 32);
		dim3 blocks(8, 8);

#define CALL_PATTERN_FUNCTION(PATTERN_TYPE)\
			CUDA_CALL_XY((__Iterate2DPattern__< float, PATTERN_TYPE >),\
			grid, blocks,\
			(float*)input.uNew.pointer, (float*)input.u.pointer,\
			(float*)input.vNew.pointer, (float*)input.v.pointer,\
			(float*)input.xGrid.pointer, (float*)input.yGrid.pointer,\
			input.uNew.nRows, input.uNew.nCols, (float)input.dt,\
			(float)input.uDiffusionCoefficient,\
			(float)input.vDiffusionCoefficient,\
			input.boundaryConditionType,\
			(float)input.patternParam1,\
			(float)input.patternParam2);


		switch (input.u.mathDomain)
		{
		case EMathDomain::Float:
			switch (input.patternType)
			{
			case EPatternType::FitzHughNagumo:
				CALL_PATTERN_FUNCTION(EPatternType::FitzHughNagumo);
				break;
			case EPatternType::Thomas:
				CALL_PATTERN_FUNCTION(EPatternType::Thomas);
				break;
			case EPatternType::Schnakernberg:
				CALL_PATTERN_FUNCTION(EPatternType::Schnakernberg);
				break;
			case EPatternType::Brussellator:
				CALL_PATTERN_FUNCTION(EPatternType::Brussellator);
				break;
			case EPatternType::GrayScott:
				CALL_PATTERN_FUNCTION(EPatternType::GrayScott);
				break;
			default:
				return -1;
			}
			break;
		case EMathDomain::Double:
			return -1;
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
DEVICE void SetBoundaryCondition(unsigned int tidX, unsigned int tidY, T* RESTRICT uNew, size_t nRows, size_t nCols, const EBoundaryCondition boundaryConditionType)
{
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

}

template <typename T>
GLOBAL void __Iterate2D__(T* RESTRICT uNew, const T* RESTRICT u, const T* RESTRICT xGrid, const T* RESTRICT yGrid, const size_t nRows, const size_t nCols, const T dt, const T diffusionCoefficient, const EBoundaryCondition boundaryConditionType)
{
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stepX = gridDim.x * blockDim.x;

	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int stepY = gridDim.y * blockDim.y;

	SetBoundaryCondition<T>(tidX, tidY, uNew, nRows, nCols, boundaryConditionType);

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

template <typename T>
__device__ void FitzHughNagumoDynamic(T* RESTRICT uNew, T* RESTRICT vNew, const size_t coord, const T u, const T v, const T dt, const T param1, const T param2)
{
	uNew[coord] += dt * (u * (1.0 - u * u) - v + param1);
	vNew[coord] += dt * param2 * (u - v);
}

template <typename T>
__device__ void ThomasDynamic(T* RESTRICT uNew, T* RESTRICT vNew, const size_t coord, const T u, const T v, const T dt, const T param1, const T param2)
{
	constexpr T rho = { 13.0 };
	constexpr T K = { 0.05 };
	constexpr T alpha = { 1.5 };
	constexpr T gamma = { 200.0 };

	const T h = rho * u * v / (1.0 + u + K * u * u);

	uNew[coord] += dt * gamma * (param1 - u - h);
	vNew[coord] += dt * gamma * (alpha * (param2 - v) - h);
}

template <typename T>
__device__ void SchnakenbergDynamic(T* RESTRICT uNew, T* RESTRICT vNew, const size_t coord, const T u, const T v, const T dt, const T param1, const T param2)
{
	const T u2v = u * u * v;

	uNew[coord] += dt * (param1 - u + u2v);
	vNew[coord] += dt * (param2 - u2v);
}

template <typename T>
__device__ void BrussellatorDynamic(T* RESTRICT uNew, T* RESTRICT vNew, const size_t coord, const T u, const T v, const T dt, const T param1, const T param2)
{
	const T u2v = u * u * v;

	uNew[coord] += dt * (param1 - (param2 + 1.0) * u + u2v);
	vNew[coord] += dt * (param2 * u - u2v);
}

template <typename T>
__device__ void GrayScottDynamic(T* RESTRICT uNew, T* RESTRICT vNew, const size_t coord, const T u, const T v, const T dt, const T param1, const T param2)
{
	const T uv2 = u * v * v;

	uNew[coord] += dt * (-uv2 + param1 * (1.0 - u));
	vNew[coord] += dt * (uv2 - (param1 + param2) * v);
}

template <typename T, EPatternType patternType>
GLOBAL void __Iterate2DPattern__(T* RESTRICT uNew,
	const T* RESTRICT u,
	T* RESTRICT vNew,
	const T* RESTRICT v,
	const T* RESTRICT xGrid,
	const T* RESTRICT yGrid,
	const size_t nRows,
	const size_t nCols,
	const T dt,
	const T uDiffusionCoefficient,
	const T vDiffusionCoefficient,
	const EBoundaryCondition boundaryConditionType,
	const T sourceParam1,
	const T sourceParam2)
{
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stepX = gridDim.x * blockDim.x;

	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int stepY = gridDim.y * blockDim.y;

	SetBoundaryCondition<T>(tidX, tidY, uNew, nRows, nCols, boundaryConditionType);
	SetBoundaryCondition<T>(tidX, tidY, vNew, nRows, nCols, boundaryConditionType);

	T dx = 0.0, dxPlus = 0.0, dxMinus = 0.0;
	T Aminus = 0.0, Aplus = 0.0, A0 = 0.0;
	T uxx = 0.0, vxx = 0.0;

	T dy = 0.0, dyPlus = 0.0, dyMinus = 0.0;
	T Bminus = 0.0, Bplus = 0.0, B0 = 0.0;
	T uyy = 0.0, vyy = 0.0;

	T uLaplacian = 0.0, vLaplacian = 0.0;
	for (size_t j = tidY + 1; j < nCols - 1; j += stepY)
	{
		// Y discretization
		dyPlus = yGrid[j + 1] - yGrid[j];
		dyMinus = yGrid[j] - yGrid[j - 1];
		dy = dyPlus + dyMinus;

		Aminus = 2.0 / (dyMinus * dy);
		Aplus = 2.0 / (dyPlus  * dy);
		A0 = -Aminus - Aplus;

		for (size_t i = tidX + 1; i < nRows - 1; i += stepX)
		{
			// X discretization
			dxPlus = xGrid[i + 1] - xGrid[i];
			dxMinus = xGrid[i] - xGrid[i - 1];
			dx = dxPlus + dxMinus;

			Bminus = 2.0 / (dxMinus * dx);
			Bplus = 2.0 / (dxPlus  * dx);
			B0 = -Bminus - Bplus;

			uxx = (Aminus * u[COORD(i - 1, j)] + A0 * u[COORD(i, j)] + Aplus * u[COORD(i + 1, j)]);
			uyy = (Bminus * u[COORD(i, j - 1)] + B0 * u[COORD(i, j)] + Bplus * u[COORD(i, j + 1)]);

			vxx = (Aminus * v[COORD(i - 1, j)] + A0 * v[COORD(i, j)] + Aplus * v[COORD(i + 1, j)]);
			vyy = (Bminus * v[COORD(i, j - 1)] + B0 * v[COORD(i, j)] + Bplus * v[COORD(i, j + 1)]);

			uLaplacian = uDiffusionCoefficient * (uxx + uyy);
			vLaplacian = vDiffusionCoefficient * (vxx + vyy);

			uNew[COORD(i, j)] = u[COORD(i, j)] + dt * uLaplacian;
			vNew[COORD(i, j)] = v[COORD(i, j)] + dt * vLaplacian;

			switch (patternType)
			{
			case EPatternType::Null:
				break;
			case EPatternType::FitzHughNagumo:
				FitzHughNagumoDynamic<T>(uNew, vNew, COORD(i, j), u[COORD(i, j)], v[COORD(i, j)], dt, sourceParam1, sourceParam2);
				break;
			case EPatternType::Thomas:
				ThomasDynamic<T>(uNew, vNew, COORD(i, j), u[COORD(i, j)], v[COORD(i, j)], dt, sourceParam1, sourceParam2);
				break;
			case EPatternType::Schnakernberg:
				SchnakenbergDynamic<T>(uNew, vNew, COORD(i, j), u[COORD(i, j)], v[COORD(i, j)], dt, sourceParam1, sourceParam2);
				break;
			case EPatternType::Brussellator:
				BrussellatorDynamic<T>(uNew, vNew, COORD(i, j), u[COORD(i, j)], v[COORD(i, j)], dt, sourceParam1, sourceParam2);
				break;
			case EPatternType::GrayScott:
				GrayScottDynamic<T>(uNew, vNew, COORD(i, j), u[COORD(i, j)], v[COORD(i, j)], dt, sourceParam1, sourceParam2);
				break;
			default:
				break;
			}
		}
	}
}