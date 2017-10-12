#include "stdafx.h"
#include "Parabolic.h"

namespace fd
{
	CParabolicData1D::CParabolicData1D(const la::CVector& grid,
										const la::CVector& initialCondition, 
										const  double diffusionCoefficient,
										const EBoundaryCondition boundaryCondition) noexcept
		: grid(grid), initialCondition(initialCondition), diffusionCoefficient(diffusionCoefficient), boundaryCondition(boundaryCondition)
	{
	}

	CParabolicSolver::CParabolicSolver(const double dt)
		: dt(dt)
	{

	}

	CParabolicSolver1D::CParabolicSolver1D(const CParabolicData1D& input, const double dt)
		: CParabolicSolver(dt), input(input)
	{
	}

	void CParabolicSolver1D::Initialize(la::CVector& solution) const
	{
		solution.ReadFrom(input.initialCondition);
	}

	void CParabolicSolver1D::Iterate(la::CVector& solution, const size_t nIterations) const
	{
		la::CVector buffer(solution);

		la::CVector* runningBuffer[2] = { &buffer, &solution };
		for (size_t i = 0; i < nIterations; i++)
		{
			dev::detail::Iterate1D(runningBuffer[i & 1]->GetBuffer(),
				runningBuffer[(i + 1) & 1]->GetBuffer(),
				input.grid.GetBuffer(),
				dt,
				input.diffusionCoefficient,
				input.boundaryCondition);
		}

		if ((nIterations & 1))
			solution.ReadFrom(buffer);
	}
}
