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

	CParabolicData2D::CParabolicData2D(const la::CVector& xGrid,
		const la::CVector& yGrid,
		const la::CMatrix& initialCondition,
		const  double diffusionCoefficient,
		const EBoundaryCondition boundaryCondition) noexcept
		: xGrid(xGrid), yGrid(yGrid), initialCondition(initialCondition), diffusionCoefficient(diffusionCoefficient), boundaryCondition(boundaryCondition)
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

	CParabolicSolver2D::CParabolicSolver2D(const CParabolicData2D& input, const double dt)
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

	void CParabolicSolver2D::Initialize(la::CMatrix& solution) const
	{
		solution.ReadFrom(input.initialCondition);
	}

	void CParabolicSolver2D::Iterate(la::CMatrix& solution, const size_t nIterations) const
	{
		la::CMatrix buffer(solution);

		la::CMatrix* runningBuffer[2] = { &buffer, &solution };
		for (size_t i = 0; i < nIterations; i++)
		{
			dev::detail::Iterate2D(runningBuffer[i & 1]->GetBuffer(),
				runningBuffer[(i + 1) & 1]->GetBuffer(),
				input.xGrid.GetBuffer(),
				input.yGrid.GetBuffer(),
				dt,
				input.diffusionCoefficient,
				input.boundaryCondition);
		}

		if ((nIterations & 1))
			solution.ReadFrom(buffer);
	}
}
