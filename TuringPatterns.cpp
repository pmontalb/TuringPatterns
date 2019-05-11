// TuringPatterns.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <CudaManager\Vector.h>
#include <CudaManager\ColumnWiseMatrix.h>
#include <CudaManager\Tensor.h>
#include "FiniteDifference\Parabolic.h"
#include "FiniteDifference\Pattern.h"
#include "cnpy.h"

#include <chrono>
#include <functional>

#pragma region Example parabolic PDE solvers

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

	cnpy::npy_save("results1D.npy", toPlot.Get(), "w");
}

void Example2D()
{
	using namespace la;
	using namespace std::chrono;

	high_resolution_clock::time_point start = high_resolution_clock::now();
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	CVector xGrid = la::LinSpace(0.0, 1.0, 128);
	CVector yGrid = la::LinSpace(0.0, 1.0, 128);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	std::cout << "Created grid in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;

	t1 = high_resolution_clock::now();

	auto x = xGrid.Get();
	auto y = yGrid.Get();
	std::vector<float> initialCondition(xGrid.size() * yGrid.size());
	for (size_t j = 0; j < yGrid.size(); j++)
		for (size_t i = 0; i < xGrid.size(); i++)
			initialCondition[i + x.size() * j] = exp(-5 * ((x[i] - .5) * (x[i] - .5) + (y[j] - .5) * (y[j] - .5)));

	CMatrix ic(xGrid.size(), yGrid.size());
	ic.ReadFrom(initialCondition);

	t2 = high_resolution_clock::now();
	std::cout << "Created initial condition in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;

	fd::CParabolicData2D input(xGrid, yGrid, ic, 1.0, EBoundaryCondition::ZeroFlux);

	double dt = 1e-6;
	fd::CParabolicSolver2D solver(input, dt);

	CMatrix solution(xGrid.size(), yGrid.size());
	solver.Initialize(solution);

	CTensor toPlot(solution.nRows(), 100);

	toPlot.matrices[0]->ReadFrom(solution);
	size_t nIterPerRound = 1000;
	for (size_t n = 1; n < toPlot.nMatrices(); n++)
	{
		t1 = high_resolution_clock::now();

		solver.Iterate(solution, nIterPerRound);

		t2 = high_resolution_clock::now();
		std::cout << "Done " << nIterPerRound << " iterations in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;

		toPlot.matrices[n]->ReadFrom(solution);
	}


	t1 = high_resolution_clock::now();
	cnpy::npy_save("results2D.npy", &toPlot.Get()[0], { toPlot.nMatrices(), toPlot.nCols(), toPlot.nRows() }, "w");
	t2 = high_resolution_clock::now();
	std::cout << "Saved NPY in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;

	high_resolution_clock::time_point end = high_resolution_clock::now();

	std::cout << "Finished in " << duration_cast<duration<double>>(end - start).count() << " seconds." << std::endl;
}

#pragma endregion

struct RunParameters
{
	EPatternType patternType = EPatternType::GrayScott;
	EBoundaryCondition boundaryCondition = EBoundaryCondition::Periodic;

	size_t xDimension = 256;
	double xMin = 0.0;
	double xMax = 0.0;

	size_t yDimension = 256;
	double yMin = 0.0;
	double yMax = 0.0;

	/// Number of plots
	size_t nIter = 200;

	/// Number of iterations between each plot
	size_t nIterPerRound = 1000;

	double dt = 1.0;

	double whiteNoiseScale = .05;

	double uDiffusion = 0.0;
	double vDiffusion = 0.0;

	double patternParameter1 = 0.0;
	double patternParameter2 = 0.0;

	char* solutionFile = "";
};

#pragma region Implementation

typedef void (*MakeInitialConditionDelegate)(la::CMatrix& uInitialCondition, la::CMatrix& vInitialCondition, const RunParameters& params);

double BinarySearch(const std::function<double(double)>& f, double a, double b, const double tolerance=1e-8)
{
	double fa = f(a);
	double fb = f(b);

	if (fa * fb > 0)
		throw std::exception("f doesn't change sign");

	double c = 0;
	for (size_t i = 0; i < 1000; i++)
	{
		c = .5 * (a + b);
		const double fc = f(c);
		fa = f(a);
		fb = f(b);

		if (fabs(fc) < tolerance)
			return c;

		if (fa * fc > 0)
			a = c;
		else
			b = c;
	}

	std::cout << "CONVERGENCE ERROR" << std::endl;
	throw;
}

template<EPatternType patternType>
void MakeInitialCondition(la::CMatrix& uInitialCondition, la::CMatrix& vInitialCondition, const RunParameters& params)
{
	switch (patternType)
	{
	case EPatternType::FitzHughNagumo:
		uInitialCondition.Set(0.0);
		vInitialCondition.Set(0.0);
		break;
	case EPatternType::Thomas:
	{
		auto f = [&](const double v)
		{
			const double u = -1.5 * (params.patternParameter2 - v) + params.patternParameter1;
			const double h = 13.0 * u * v / (1.0 + u + 0.05 * u * u);
			return h - (params.patternParameter1 - u);
		};
		const double v0 = BinarySearch(f, 0.0, 100.0);
		const double u0 = -1.5 * (params.patternParameter2 - v0) + params.patternParameter1;

		uInitialCondition.Set(u0);
		vInitialCondition.Set(v0);
	}
		break;
	case EPatternType::Schnakernberg:
		uInitialCondition.Set(params.patternParameter1 + params.patternParameter2);
		vInitialCondition.Set(params.patternParameter2 / ((params.patternParameter1 + params.patternParameter2) * (params.patternParameter1 + params.patternParameter2)));
		break;
	case EPatternType::Brussellator:
		uInitialCondition.Set(params.patternParameter1);
		vInitialCondition.Set(params.patternParameter2 / params.patternParameter1);
		break;
	case EPatternType::GrayScott:
	{
		uInitialCondition.Set(1.0);
		vInitialCondition.Set(0.0);

		std::vector<float> uCenteredSquare(uInitialCondition.size());
		std::vector<float> vCenteredSquare(uInitialCondition.size());
		size_t squareStartX = uInitialCondition.nRows() * 2 / 5;
		size_t squareEndX = uInitialCondition.nRows() * 3 / 5;
		size_t squareStartY = uInitialCondition.nCols() * 2 / 5;
		size_t squareEndY = uInitialCondition.nCols() * 3 / 5;
		for (size_t j = squareStartY; j < squareEndY; j++)
		{
			for (size_t i = squareStartX; i < squareEndX; i++)
			{
				uCenteredSquare[i + uInitialCondition.nRows() * j] = -.5;
				vCenteredSquare[i + uInitialCondition.nRows() * j] = .25;
			}
		}
		la::CMatrix uAddition(uInitialCondition);
		uAddition.ReadFrom(uCenteredSquare);
		la::CMatrix vAddition(vInitialCondition);
		vAddition.ReadFrom(vCenteredSquare);

		uInitialCondition.AddEqual(uAddition);
		vInitialCondition.AddEqual(vAddition);
	}
		break;
	default:
		break;
	}
}

void RunDelegate(MakeInitialConditionDelegate makeInitialCondition, const RunParameters& params)
{
	using namespace la;
	using namespace std::chrono;

	high_resolution_clock::time_point start = high_resolution_clock::now();
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// ************ Make Grid ************
	CVector xGrid = la::LinSpace(params.xMin, params.xMax, params.xDimension);
	CVector yGrid = la::LinSpace(params.yMin, params.yMax, params.yDimension);

	high_resolution_clock::time_point t2 = high_resolution_clock::now(); \

		std::cout << "Created grid in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl; \
		// ***********************************

		// ***************** Make Initial Condition ******************
		t1 = high_resolution_clock::now();

	CMatrix whiteNoise = la::RandomGaussian(xGrid.size(), yGrid.size());
	whiteNoise.Scale(params.whiteNoiseScale);

	CMatrix uInitialCondition(xGrid.size(), yGrid.size());
	CMatrix vInitialCondition(xGrid.size(), yGrid.size());
	makeInitialCondition(uInitialCondition, vInitialCondition, params);

	uInitialCondition.AddEqual(whiteNoise);
	vInitialCondition.AddEqual(whiteNoise);

	t2 = high_resolution_clock::now();
	std::cout << "Created initial condition in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;
	// ***************************************************************

	// **************** Initialize Solver ***************
	t1 = high_resolution_clock::now();

	fd::CPatternData2D input(xGrid, yGrid, uInitialCondition, vInitialCondition,
		params.uDiffusion, params.vDiffusion, params.patternType, params.boundaryCondition);
	fd::CPatternSolver2D solver(input, params.dt);

	CMatrix uSolution(xGrid.size(), yGrid.size());
	CMatrix vSolution(xGrid.size(), yGrid.size());
	solver.Initialize(uSolution, vSolution);

	CTensor toPlot(uSolution.nRows(), uSolution.nCols(), params.nIter);

	toPlot.matrices[0]->ReadFrom(uSolution);
	std::cout << "Solver setup in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;
	// ****************************************************


	// ************** Iterate solver ***************
	for (size_t n = 1; n < toPlot.nMatrices(); n++)
	{
		t1 = high_resolution_clock::now();

		solver.Iterate(uSolution, vSolution, params.nIterPerRound, params.patternParameter1, params.patternParameter2);

		t2 = high_resolution_clock::now();
		std::cout << "Done " << params.nIterPerRound << " iterations in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;

		toPlot.matrices[n]->ReadFrom(uSolution);
	}
	// *********************************************

	// ************** Save solution to CSV ************
	t1 = high_resolution_clock::now();
	cnpy::npy_save(params.solutionFile, &toPlot.Get()[0], { toPlot.nMatrices(), toPlot.nCols(), toPlot.nRows() }, "w");
	t2 = high_resolution_clock::now();
	std::cout << "Saved csv in " << duration_cast<duration<double>>(t2 - t1).count() << " seconds." << std::endl;

	high_resolution_clock::time_point end = high_resolution_clock::now();

	std::cout << "Finished in " << duration_cast<duration<double>>(end - start).count() << " seconds." << std::endl;
	// *************************************************
}

#pragma endregion

void Run(const RunParameters& params)
{
	MakeInitialConditionDelegate makeInitialCondition = nullptr;

	switch (params.patternType)
	{
	case EPatternType::FitzHughNagumo:
		makeInitialCondition = MakeInitialCondition<EPatternType::FitzHughNagumo>;
		break;
	case EPatternType::Thomas:
		makeInitialCondition = MakeInitialCondition<EPatternType::Thomas>;
		break;
	case EPatternType::Schnakernberg:
		makeInitialCondition = MakeInitialCondition<EPatternType::Schnakernberg>;
		break;
	case EPatternType::Brussellator:
		makeInitialCondition = MakeInitialCondition<EPatternType::Brussellator>;
		break;
	case EPatternType::GrayScott:
		makeInitialCondition = MakeInitialCondition<EPatternType::GrayScott>;
		break;
	default:
		break;
	}

	RunDelegate(makeInitialCondition, params);
}

int main(int argc, char** argv)
{
	RunParameters params;

#define PARSE(PARAM, TYPE) if (!strcmp(argv[c], "-" #PARAM)) params.##PARAM = std::ato##TYPE(argv[++c]);

	for (size_t c = 1; c < argc; c++)
	{
		if (!strcmp(argv[c], "-pattern"))
		{
			++c;
			if (!strcmp(argv[c], "GrayScott"))
				params.patternType = EPatternType::GrayScott;
			else if (!strcmp(argv[c], "Brussellator"))
				params.patternType = EPatternType::Brussellator;
			else if (!strcmp(argv[c], "Schnakenberg"))
				params.patternType = EPatternType::Schnakernberg;
			else if (!strcmp(argv[c], "Thomas"))
				params.patternType = EPatternType::Thomas;
			else if (!strcmp(argv[c], "FitzHughNagumo"))
				params.patternType = EPatternType::FitzHughNagumo;
		}
		if (!strcmp(argv[c], "-boundaryCondition"))
		{
			++c;
			if (!strcmp(argv[c], "Periodic"))
				params.boundaryCondition = EBoundaryCondition::Periodic;
			else if (!strcmp(argv[c], "ZeroFlux"))
				params.boundaryCondition = EBoundaryCondition::ZeroFlux;
		}
		if (!strcmp(argv[c], "-solutionFile"))
		{
			params.solutionFile = argv[++c];
		}
		PARSE(xDimension, i);
		PARSE(xMin, f);
		PARSE(xMax, f);
		PARSE(yDimension, i);
		PARSE(yMin, f);
		PARSE(yMax, f);
		PARSE(nIter, i);
		PARSE(nIterPerRound, i);
		PARSE(dt, f);
		PARSE(whiteNoiseScale, f);
		PARSE(uDiffusion, f);
		PARSE(vDiffusion, f);
		PARSE(patternParameter1, f);
		PARSE(patternParameter2, f);
	}

	if (params.xMin == params.xMax)
	{
		params.xMin = 0;
		params.xMax = params.xDimension - 1;
	}
	if (params.yMin == params.yMax)
	{
		params.yMin = 0;
		params.yMax = params.yDimension - 1;
	}

	Run(params);

    return 0;
}

