#include <fstream>
#include <algorithm>
#include <chrono>

#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <ComputeCopy.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <AdvectionDiffusionSolver1D.h>
#include <AdvectionDiffusionSolver2D.h>
#include <WaveEquationSolver1D.h>
#include <WaveEquationSolver2D.h>
#include <IterableEnum.h>

#include <PatternType.h>
#include <PatternDynamic.cuh>

#pragma region Command Line Parser

class CommandLineArgumentParser
{
public:
	CommandLineArgumentParser(int argc, char **argv)
		: args(argv, argv + argc)
	{
	}

	template<typename T>
	T GetArgumentValue(const std::string& option) const;

	template<typename T>
	T GetArgumentValue(const std::string& option, const T& default) const noexcept
	{
		T ret;
		try
		{
			ret = GetArgumentValue<T>(option);
		}
		catch (int)
		{
			ret = default;
		}

		return ret;
	}

	bool GetFlag(const std::string& option) const
	{
		return std::find(args.begin(), args.end(), option) != args.end();
	}

private:
	std::vector<std::string> args;
};

template<>
std::string CommandLineArgumentParser::GetArgumentValue<std::string>(const std::string& option) const
{
	auto itr = std::find(args.begin(), args.end(), option);
	if (itr != args.end())
	{
		if (++itr == args.end())
			std::abort();
		return *itr;
	}

	throw 42;
}

template<>
int CommandLineArgumentParser::GetArgumentValue<int>(const std::string& option) const
{
	return std::atoi(GetArgumentValue<std::string>(option).c_str());
}

template<>
double CommandLineArgumentParser::GetArgumentValue<double>(const std::string& option) const
{
	return std::atof(GetArgumentValue<std::string>(option).c_str());
}

#pragma endregion

#pragma region Enum Mapping

#define PARSE(E, X)\
	if (!strcmp(text.c_str(), #X))\
		return E::X;

PatternType parsePatternType(const std::string& text)
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

MathDomain parseMathDomain(const std::string& text)
{
#define PARSE_WORKER(X) PARSE(MathDomain, X);

	PARSE_WORKER(Double);
	PARSE_WORKER(Float);

#undef PARSE_WORKER
	return MathDomain::Null;
}

#undef PARSE

#pragma endregion

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

	size_t xDimension = 128;
	double xMin = 0.0;
	double xMax = 1.0;

	size_t yDimension = 128;
	double yMin = 0.0;
	double yMax = 1.0;

	/// Number of plots
	size_t nIter = 200;
	size_t nIterPerRound = 10;

	double dt = 1.0;

	double whiteNoiseScale = .05;

	double uDiffusion = 0.16;
	double vDiffusion = 0.08;

	double patternParameter1 = 0.035;
	double patternParameter2 = 0.065;
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
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	// ************ Make Grid ************
	vector<md> xGrid = cl::LinSpace<MemorySpace::Device, md>(params.xMin, params.xMax, params.xDimension);
	vector<md> yGrid = cl::LinSpace<MemorySpace::Device, md>(params.yMin, params.yMax, params.yDimension);

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

	pde::AdvectionDiffusionSolver2D<MemorySpace::Device, md> uSolver(uData);
	pde::AdvectionDiffusionSolver2D<MemorySpace::Device, md> vSolver(vData);

	std::cout << "Solver setup in " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << " seconds." << std::endl;
	// ****************************************************

	// solution matrix is a collection of flattened solutions over time
	forge::Window wnd(1024, 768, "3d Surface Demo");
	wnd.makeCurrent();

	forge::Chart chart(FG_CHART_3D);

	auto _xGrid = xGrid.Get();
	auto _yGrid = yGrid.Get();
	auto _ic = uInitialCondition.Get();
	chart.setAxesLimits(_xGrid.front(), _xGrid.back(), _yGrid.front(), _yGrid.back(), .95 * *std::min_element(_ic.begin(), _ic.end()), 1.05 * *std::max_element(_ic.begin(), _ic.end()));
	chart.setAxesTitles("x-axis", "y-axis", "z-axis");

	forge::Surface surf = chart.surface(_xGrid.size(), _yGrid.size(), forge::f32);
	surf.setColor(FG_BLUE);

	GfxHandle* handle;
	createGLBuffer(&handle, surf.vertices(), FORGE_VERTEX_BUFFER);

	bool toDo = true;
	cl::Vector<MemorySpace::Device, MathDomain::Float>* xyzTriple = nullptr;

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

				if (!xyzTriple)
					xyzTriple = new cl::Vector<MemorySpace::Device, MathDomain::Float>(3 * xGrid.size() * yGrid.size());

				cl::MakeTriple(*xyzTriple, xGrid, yGrid, *uSolver.solution->columns[0]);
				copyToGLBuffer(handle, (ComputeResourceHandle)xyzTriple->GetBuffer().pointer, surf.verticesSize());
				wnd.draw(chart);
			}
		}

		wnd.draw(chart);
		toDo = false;
	}
	while (!wnd.close());

	releaseGLBuffer(handle);
	delete xyzTriple;
}

int main(int argc, char** argv)
{
	CommandLineArgumentParser ap(argc, argv);

	auto mathDomain = parseMathDomain(ap.GetArgumentValue<std::string>("-md", "Float"));
	auto patternType = parsePatternType(ap.GetArgumentValue<std::string>("-pattern", "GrayScott"));

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
	rp.whiteNoiseScale = ap.GetArgumentValue<double>("-wns", rp.whiteNoiseScale);
	rp.uDiffusion = ap.GetArgumentValue<double>("-ud", rp.uDiffusion);
	rp.vDiffusion = ap.GetArgumentValue<double>("-vd", rp.vDiffusion);
	rp.patternParameter1 = ap.GetArgumentValue<double>("-p1", rp.patternParameter1);
	rp.patternParameter2 = ap.GetArgumentValue<double>("-p2", rp.patternParameter2);


	switch (mathDomain)
	{
		case MathDomain::Float:
			switch (patternType)
			{
				case PatternType::FitzHughNagumo:
					runner<MathDomain::Float, PatternType::FitzHughNagumo>(rp);
				case PatternType::Thomas:
					runner<MathDomain::Float, PatternType::Thomas>(rp);
				case PatternType::Schnakenberg:
					runner<MathDomain::Float, PatternType::Schnakenberg>(rp);
				case PatternType::Brussellator:
					runner<MathDomain::Float, PatternType::Brussellator>(rp);
				case PatternType::GrayScott:
					runner<MathDomain::Float, PatternType::GrayScott>(rp);
				default:
					break;
			}
			break;
		case MathDomain::Double:
			switch (patternType)
			{
				case PatternType::FitzHughNagumo:
					runner<MathDomain::Double, PatternType::FitzHughNagumo>(rp);
				case PatternType::Thomas:
					runner<MathDomain::Double, PatternType::Thomas>(rp);
				case PatternType::Schnakenberg:
					runner<MathDomain::Double, PatternType::Schnakenberg>(rp);
				case PatternType::Brussellator:
					runner<MathDomain::Double, PatternType::Brussellator>(rp);
				case PatternType::GrayScott:
					runner<MathDomain::Double, PatternType::GrayScott>(rp);
				default:
					break;
			}
		default:
			throw NotImplementedException();
	}

	return 0;
}