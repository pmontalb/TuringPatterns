// TuringPatterns.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CudaManager\DeviceManager.h"
#include "CudaManager\Vector.h"
#include "CudaManager\Matrix.h"
#include "FiniteDifference\Parabolic.h"


int main()
{
	using namespace la;

	CVector grid = la::LinSpace(0.0, 1.0, 16);
	auto x = grid.Get();
	std::vector<float> initialCondition(grid.size());
	for (size_t i = 0; i < initialCondition.size(); i++)
	{
		initialCondition[i] = exp(-5 * (x[i] - .5) * (x[i] - .5));
	}

	CVector ic(grid);
	ic.ReadFrom(initialCondition);
	fd::CParabolicData1D input(grid, ic, 1.0, EBoundaryCondition::ZeroFlux);

	double dt = 1e-4;
	fd::CParabolicSolver1D solver(input, dt);

	CVector solution(grid.size());
	solver.Initialize(solution);

	CMatrix toPlot(solution.size(), 100);
	
	toPlot.columns[0]->ReadFrom(solution);
	size_t nIterPerRound = 10;
	for (size_t n = 1; n < toPlot.nCols(); n++)
	{
		solver.Iterate(solution, nIterPerRound);
		toPlot.columns[n]->ReadFrom(solution);
	}

	toPlot.ToCsv("results.csv");

    return 0;
}

