// TuringPatterns.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CudaManager\DeviceManager.h"
#include "CudaManager\Vector.h"
#include "CudaManager\Matrix.h"
#include "CudaManager\Tensor.h"
#include "FiniteDifference\Parabolic.h"

void Example1D()
{
	using namespace la;

	CVector grid = la::LinSpace(0.0, 1.0, 16);
	auto x = grid.Get();
	std::vector<float> initialCondition(grid.size());
	for (size_t i = 0; i < initialCondition.size(); i++)
		initialCondition[i] = exp(-5 * (x[i] - .5) * (x[i] - .5));

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

	toPlot.ToCsv("results1D.csv");
}

void Example2D()
{
	using namespace la;

	CVector xGrid = la::LinSpace(0.0, 1.0, 16);
	CVector yGrid = la::LinSpace(0.0, 1.0, 16);
	auto x = xGrid.Get();
	auto y = yGrid.Get();
	std::vector<float> initialCondition(xGrid.size() * yGrid.size());
	for (size_t j = 0; j < yGrid.size(); j++)
		for (size_t i = 0; i < xGrid.size(); i++)
			initialCondition[i + x.size() * j] = exp(-5 * ((x[i] - .5) * (x[i] - .5) + (y[j] - .5) * (y[j] - .5)));

	CMatrix ic(xGrid.size(), yGrid.size());
	ic.ReadFrom(initialCondition);
	fd::CParabolicData2D input(xGrid, yGrid, ic, 1.0, EBoundaryCondition::ZeroFlux);

	double dt = 1e-4;
	fd::CParabolicSolver2D solver(input, dt);

	CMatrix solution(xGrid.size(), yGrid.size());
	solver.Initialize(solution);

	CTensor toPlot(solution.nRows(), 100);

	toPlot.matrices[0]->ReadFrom(solution);
	size_t nIterPerRound = 10;
	for (size_t n = 1; n < toPlot.nMatrices(); n++)
	{
		solver.Iterate(solution, nIterPerRound);
		toPlot.matrices[n]->ReadFrom(solution);
	}

	toPlot.ToCsv("results2D.csv");
}


int main()
{
	Example2D();

    return 0;
}

