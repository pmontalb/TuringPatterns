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
}
