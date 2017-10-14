#pragma once

#include "Common.h"
#include "../CudaKernels/Types.h"
#include "../CudaManager/Vector.h"
#include "../CudaManager/Matrix.h"

namespace fd
{
	class FINITEDIFFERENCE_API CParabolicData1D 
	{
	public:
		CParabolicData1D(const la::CVector& grid, const la::CVector& initialCondition, const double diffusionCoefficient, const EBoundaryCondition boundaryCondition) noexcept;

		const la::CVector& grid;
		const la::CVector& initialCondition;
		const double diffusionCoefficient;
		const EBoundaryCondition boundaryCondition;
	};

	class FINITEDIFFERENCE_API CParabolicData2D
	{
	public:
		CParabolicData2D(const la::CVector& xGrid, const la::CVector& yGrid, const la::CMatrix& initialCondition, const double diffusionCoefficient, const EBoundaryCondition boundaryCondition) noexcept;

		const la::CVector& xGrid;
		const la::CVector& yGrid;
		const la::CMatrix& initialCondition;
		const double diffusionCoefficient;
		const EBoundaryCondition boundaryCondition;
	};

	class FINITEDIFFERENCE_API CParabolicSolver
	{
	public:
		CParabolicSolver(const double dt);

		const double dt;
	};

	class FINITEDIFFERENCE_API CParabolicSolver1D : public CParabolicSolver
	{
	public:
		CParabolicSolver1D(const CParabolicData1D& input, const double dt);

		void Initialize(la::CVector& solution) const;
		void Iterate(la::CVector& solution, const size_t nIterations) const;


	protected:
		const CParabolicData1D& input;
	};

	class FINITEDIFFERENCE_API CParabolicSolver2D : public CParabolicSolver
	{
	public:
		CParabolicSolver2D(const CParabolicData2D& input, const double dt);

		void Initialize(la::CMatrix& solution) const;
		void Iterate(la::CMatrix& solution, const size_t nIterations) const;


	protected:
		const CParabolicData2D& input;
	};
}
