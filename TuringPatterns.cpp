// TuringPatterns.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CudaManager\DeviceManager.h"
#include "CudaManager\Vector.h"
#include "CudaManager\Matrix.h"
#include "FiniteDifference\Parabolic.h"


int main()
{
	using namespace std;
	using namespace dev;
	using namespace la;

	CDeviceManager& devManager = CDeviceManager::Get();
	auto nGpu = devManager.GetDeviceCount();

	cout << "There are " << nGpu << " GPUs" << endl;

	//CVector v(10, -123);
	//v.Print();

	//auto w = la::LinSpace(2, 5, 25);
	//w.Print();

	//auto z = la::RandomUniform(16);
	//z.Print();

	//auto t = la::RandomGaussian(10);
	//t.Print();

	//CMatrix A = la::RandomGaussian(4, 4);
	//A.Print();

	//CMatrix v(10, 10, -123, EMemorySpace::Device, EMathDomain::Double);
	//v.Print("v");

	//CMatrix u(10, 10, 123, EMemorySpace::Device, EMathDomain::Double);
	//u.Print("u");

	//v.AddEqual(u, 2.0);
	//u.Print("u after first add");

	//v.AddEqual(u, 2.0);
	//v.Print("u after second add");

	CVector grid = la::LinSpace(0.0, 1.0, 16);
	auto x = grid.Get();
	std::vector<float> initialCondition(grid.size());
	for (size_t i = 0; i < initialCondition.size(); i++)
	{
		initialCondition[i] = exp(-5 * (x[i] - .5) * (x[i] - .5));
	}

	CVector ic(grid);
	ic.ReadFrom(initialCondition);
	ic.Print();
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

    return 0;
}

