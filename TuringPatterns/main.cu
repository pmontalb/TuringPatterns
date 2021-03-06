#include <fstream>
#include <algorithm>
#include <chrono>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <AdvectionDiffusionSolver1D.h>
#include <AdvectionDiffusionSolver2D.h>
#include <WaveEquationSolver1D.h>
#include <WaveEquationSolver2D.h>
#include <IterableEnum.h>
#include <Utils/CommandLineParser.h>
#include <Utils/EnumParser.h>

#include <PatternType.h>
#include <PatternDynamic.cuh>
#include <ForgeHelpers.cuh>

#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <fg/compute_copy.h>

//#define PLOT_3D
#define SAVE_TO_FILE

namespace ep
{
#define PARSE(E, X)\
    if (!strcmp(text.c_str(), #X))\
        return E::X;

	PatternType ParsePatternType(const std::string &text)
	{
#define PARSE_WORKER(X) PARSE(PatternType, X);

		PARSE_WORKER(FitzHughNagumo);
		PARSE_WORKER(Thomas);
		PARSE_WORKER(Schnakenberg);
		PARSE_WORKER(Brussellator);
		PARSE_WORKER(GrayScott);

#undef PARSE_WORKER

		return PatternType::Null;
	}

#undef PARSE
}

template<MathDomain md>
using vector = cl::Vector<MemorySpace::Device, md>;

template<MathDomain md>
using matrix = cl::ColumnWiseMatrix<MemorySpace::Device, md>;

template<MathDomain md>
using sType = typename vector<md>::stdType;

struct RunParameters
{
	PatternType patternType = PatternType::GrayScott;
	BoundaryConditionType boundaryCondition = BoundaryConditionType::Periodic;

	size_t xDimension = 64;
	double xMin = 0.0;
	double xMax = 1.0;

	size_t yDimension = 64;
	double yMin = 0.0;
	double yMax = 1.0;

	/// Number of plots
	size_t nIter = 2000;
	size_t nIterPerRound = 10;

	double dt = 1.0;

	double whiteNoiseScale = .05;

	double uDiffusion = 0.16;
	double vDiffusion = 0.08;

	double patternParameter1 = 0.035;
	double patternParameter2 = 0.065;

	double zMin = 0.1;
    double zMax = 1.5;

    std::string outputFile = "";
};

template<PatternType patternType, MathDomain md>
void MakeInitialCondition(matrix<md>& uInitialCondition, matrix<md>& vInitialCondition, const RunParameters& params)
{
	switch (patternType)
	{
		case PatternType::FitzHughNagumo:
			uInitialCondition.Set(0.0);
			vInitialCondition.Set(0.0);
			break;
		case PatternType::Thomas:
		{
			auto f = [&](const double v)
			{
				const double u = -1.5 * (params.patternParameter2 - v) + params.patternParameter1;
				const double h = 13.0 * u * v / (1.0 + u + 0.05 * u * u);
				return h - (params.patternParameter1 - u);
			};
			auto binarySearch = [&](double a, double b, const double tolerance = 1e-8)
			{
				double fa = f(a);
				double fb = f(b);

				if (fa * fb > 0)
					throw std::logic_error("f doesn't change sign");

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
			};

			const double v0 = binarySearch(0.0, 100.0);
			const double u0 = -1.5 * (params.patternParameter2 - v0) + params.patternParameter1;

			uInitialCondition.Set(u0);
			vInitialCondition.Set(v0);
		}
		break;
		case PatternType::Schnakenberg:
			uInitialCondition.Set(params.patternParameter1 + params.patternParameter2);
			vInitialCondition.Set(params.patternParameter2 / ((params.patternParameter1 + params.patternParameter2) * (params.patternParameter1 + params.patternParameter2)));
			break;
		case PatternType::Brussellator:
			uInitialCondition.Set(params.patternParameter1);
			vInitialCondition.Set(params.patternParameter2 / params.patternParameter1);
			break;
		case PatternType::GrayScott:
		{
			uInitialCondition.Set(1.0);
			vInitialCondition.Set(0.0);

			std::vector<sType<md>> uCenteredSquare(uInitialCondition.size());
			std::vector<sType<md>> vCenteredSquare(vInitialCondition.size());
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
			matrix<md> uAddition(uCenteredSquare, uInitialCondition.nRows(), uInitialCondition.nCols());
			matrix<md> vAddition(vCenteredSquare, vInitialCondition.nRows(), vInitialCondition.nCols());

			uInitialCondition += uAddition;
			vInitialCondition += vAddition;
		}
		break;
		default:
			break;
	}
}

template<MathDomain md, PatternType type>
void runner(const RunParameters& params)
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	// ************ Make Grid ************
	vector<md> xGrid = vector<md>::LinSpace(params.xMin, params.xMax, params.xDimension);
	vector<md> yGrid = vector<md>::LinSpace(params.yMin, params.yMax, params.yDimension);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); \

	std::cout << "Created grid in " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << " seconds." << std::endl; \
	// ***********************************

	// ***************** Make Initial Condition ******************
	t1 = std::chrono::high_resolution_clock::now();

	matrix<md> whiteNoise(xGrid.size(), yGrid.size()); 
	whiteNoise.RandomGaussian(1234);
	whiteNoise.Scale(params.whiteNoiseScale);

	matrix<md> uInitialCondition(xGrid.size(), yGrid.size());
	matrix<md> vInitialCondition(xGrid.size(), yGrid.size());
	MakeInitialCondition<type>(uInitialCondition, vInitialCondition, params);

	uInitialCondition += whiteNoise;
	vInitialCondition += whiteNoise;

	t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Created initial condition in " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << " seconds." << std::endl;
	// ***************************************************************

	// **************** Initialize Solver ***************
	t1 = std::chrono::high_resolution_clock::now();

	BoundaryCondition leftBc(params.boundaryCondition, 0.0);
	BoundaryCondition rightBc(params.boundaryCondition, 0.0);
	BoundaryCondition downBc(params.boundaryCondition, 0.0);
	BoundaryCondition upBc(params.boundaryCondition, 0.0);
	BoundaryCondition2D bc(leftBc, rightBc, downBc, upBc);

	pde::PdeInputData2D<MemorySpace::Device, md> uData(uInitialCondition, xGrid, yGrid, 0.0, 0.0, params.uDiffusion, params.dt, SolverType::ImplicitEuler, SpaceDiscretizerType::Centered, bc);
	pde::PdeInputData2D<MemorySpace::Device, md> vData(vInitialCondition, xGrid, yGrid, 0.0, 0.0, params.vDiffusion, params.dt, SolverType::ImplicitEuler, SpaceDiscretizerType::Centered, bc);

	pde::AdvectionDiffusionSolver2D<MemorySpace::Device, md> uSolver(std::move(uData));
	pde::AdvectionDiffusionSolver2D<MemorySpace::Device, md> vSolver(std::move(vData));

	std::cout << "Solver setup in " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << " seconds." << std::endl;
	// ****************************************************

	auto _xGrid = xGrid.Get();
	auto _yGrid = yGrid.Get();

#ifndef SAVE_TO_FILE
	// solution matrix is a collection of flattened solutions over time
	forge::Window wnd(1024, 768, "Pattern");
	wnd.makeCurrent();

	#ifdef PLOT_3D
		forge::Chart chart(FG_CHART_3D);

		auto _ic = uInitialCondition.Get();
		chart.setAxesLimits(_xGrid.front(), _xGrid.back(), _yGrid.front(), _yGrid.back(), params.zMin, params.zMax);
		chart.setAxesTitles("x-axis", "y-axis", "z-axis");

		forge::Surface surf = chart.surface(_xGrid.size(), _yGrid.size(), forge::f32);
		surf.setColor(FG_BLUE);

		GfxHandle* handle;
		createGLBuffer(&handle, surf.vertices(), FORGE_VERTEX_BUFFER);
	#else
		forge::Image img(_xGrid.size(), _yGrid.size(), FG_RGBA, forge::f32);

		GfxHandle* handle = 0;
		createGLBuffer(&handle, img.pixels(), FORGE_IMAGE_BUFFER);
	#endif

	bool toDo = true;

#ifdef PLOT_3D
	cl::Vector<MemorySpace::Device, MathDomain::Float> xyzTriple(3 * uInitialCondition.size());
#else
	MemoryBuffer colorMap(0, 4 * uInitialCondition.size(), MemorySpace::Device, MathDomain::Float);
	dm::detail::Alloc(colorMap);

	vector<md> uNormalised(uInitialCondition.size());
	vector<md> minDummy(uNormalised.size(), 1.0);
#endif

	do
	{
		if (toDo)
		{
			for (unsigned m = 0; m < params.nIter; ++m)
			{
				for (unsigned n = 0; n < params.nIterPerRound; ++n)
				{
					uSolver.Advance(1);
					vSolver.Advance(1);
					_ApplyPatternDynamic(uSolver.solution->columns[0]->GetBuffer(), vSolver.solution->columns[0]->GetBuffer(), type, params.dt, params.patternParameter1, params.patternParameter2);
				}

#ifdef PLOT_3D
				matrix<md>::MakeTriple(xyzTriple, xGrid, yGrid, *uSolver.solution->columns[0]);
				copyToGLBuffer(handle, reinterpret_cast<ComputeResourceHandle>(xyzTriple.GetBuffer().pointer), surf.verticesSize());
				wnd.draw(chart);
#else
				double min = uSolver.solution->columns[0]->Minimum();
				minDummy.Set(min);
				uNormalised.AddEqual(minDummy, -1.0);

				double max = uSolver.solution->columns[0]->AbsoluteMaximum();
				uNormalised.ReadFrom(*uSolver.solution->columns[0]);
				uNormalised.Scale(1.0 / max);
				_MakeRgbaJetColorMap(colorMap, uNormalised.GetBuffer());
				copyToGLBuffer(handle, (ComputeResourceHandle)colorMap.pointer, img.size());
				wnd.draw(img);
#endif
			}
		}

#ifdef PLOT_3D
		wnd.draw(chart);
#else
		wnd.draw(img);
#endif
		toDo = false;
	}
	while (!wnd.close());

	releaseGLBuffer(handle);
#else
	std::vector<sType<md>> solutionMatrix;

	for (unsigned m = 0; m < params.nIter; ++m)
	{
		t1 = std::chrono::high_resolution_clock::now();

		for (unsigned n = 0; n < params.nIterPerRound; ++n)
		{
			uSolver.Advance(1);
			vSolver.Advance(1);
			_ApplyPatternDynamic(uSolver.solution->columns[0]->GetBuffer(), vSolver.solution->columns[0]->GetBuffer(), type, params.dt, params.patternParameter1, params.patternParameter2);
		}

		t2 = std::chrono::high_resolution_clock::now();
		std::cout << "Step " << m << " done in " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << " seconds." << std::endl;

		const auto solution = uSolver.solution->columns[0]->Get();
		solutionMatrix.insert(solutionMatrix.end(), solution.begin(), solution.end());
	}

	cl::MatrixToBinaryFile<sType<md>>(solutionMatrix, params.nIter, uSolver.solution->columns[0]->size(), params.outputFile, false);
#endif
}

int main(int argc, char** argv)
{
	clp::CommandLineArgumentParser ap(argc, argv);

	auto mathDomain = ep::ParseMathDomain(ap.GetArgumentValue<std::string>("-md", "Float"));
	auto patternType = ep::ParsePatternType(ap.GetArgumentValue<std::string>("-pattern", "GrayScott"));

	RunParameters rp;
	std::string bc = ap.GetArgumentValue<std::string>("-bc", "Periodic");
	if (bc == "Periodic")
		rp.boundaryCondition = BoundaryConditionType::Periodic;
	if (bc == "ZeroFlux")
		rp.boundaryCondition = BoundaryConditionType::Neumann;
	rp.dt = ap.GetArgumentValue<double>("-dt", rp.dt);
	rp.nIter = ap.GetArgumentValue<double>("-nIter", rp.nIter);
	rp.nIterPerRound = ap.GetArgumentValue<double>("-nIterPerRound", rp.nIterPerRound);
	rp.xDimension = ap.GetArgumentValue<double>("-xd", rp.xDimension);
	rp.yDimension = ap.GetArgumentValue<double>("-yd", rp.yDimension);
	rp.xMin = ap.GetArgumentValue<double>("-xm", rp.xMin);
	rp.xMax = ap.GetArgumentValue<double>("-xM", 0.0);
	if (rp.xMax == 0)
		rp.xMax = rp.xDimension - 1.0;
	rp.yDimension = ap.GetArgumentValue<double>("-yd", rp.yDimension);
	rp.yMin = ap.GetArgumentValue<double>("-ym", rp.yMin);
	rp.yMax = ap.GetArgumentValue<double>("-yM", 0.0);
	if (rp.yMax == 0)
		rp.yMax = rp.yDimension - 1.0;
    rp.zMin = ap.GetArgumentValue<double>("-zm", rp.zMin);
    rp.zMax = ap.GetArgumentValue<double>("-zM", rp.zMax);

	rp.whiteNoiseScale = ap.GetArgumentValue<double>("-wns", rp.whiteNoiseScale);
	rp.uDiffusion = ap.GetArgumentValue<double>("-ud", rp.uDiffusion);
	rp.vDiffusion = ap.GetArgumentValue<double>("-vd", rp.vDiffusion);
	rp.patternParameter1 = ap.GetArgumentValue<double>("-p1", rp.patternParameter1);
	rp.patternParameter2 = ap.GetArgumentValue<double>("-p2", rp.patternParameter2);
	rp.outputFile = ap.GetArgumentValue<std::string>("-of", "sol.cl");

	switch (mathDomain)
	{
		case MathDomain::Float:
			switch (patternType)
			{
				case PatternType::FitzHughNagumo:
					runner<MathDomain::Float, PatternType::FitzHughNagumo>(rp);
					break;
				case PatternType::Thomas:
					runner<MathDomain::Float, PatternType::Thomas>(rp);
					break;
				case PatternType::Schnakenberg:
					runner<MathDomain::Float, PatternType::Schnakenberg>(rp);
					break;
				case PatternType::Brussellator:
					runner<MathDomain::Float, PatternType::Brussellator>(rp);
					break;
				case PatternType::GrayScott:
					runner<MathDomain::Float, PatternType::GrayScott>(rp);
					break;
				default:
					break;
			}
			break;
		case MathDomain::Double:
			switch (patternType)
			{
				case PatternType::FitzHughNagumo:
					runner<MathDomain::Double, PatternType::FitzHughNagumo>(rp);
					break;
				case PatternType::Thomas:
					runner<MathDomain::Double, PatternType::Thomas>(rp);
					break;
				case PatternType::Schnakenberg:
					runner<MathDomain::Double, PatternType::Schnakenberg>(rp);
					break;
				case PatternType::Brussellator:
					runner<MathDomain::Double, PatternType::Brussellator>(rp);
					break;
				case PatternType::GrayScott:
					runner<MathDomain::Double, PatternType::GrayScott>(rp);
					break;
				default:
					break;
			}
		default:
			throw NotImplementedException();
	}

	return 0;
}