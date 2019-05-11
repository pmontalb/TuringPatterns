#include "PatternDynamic.cuh"

EXTERN_C
{
	/**
	*   [u, v]' += dt * [f(u,v); g(u,v)]'
	*/
	EXPORT int _ApplyPatternDynamic(MemoryBuffer u, MemoryBuffer v, const PatternType type, const double dt, const double param1, const double param2)
    {
#define APPLY_PATTERN_DYNAMIC(TYPE)\
		switch (u.mathDomain)\
		{\
			case MathDomain::Float:\
				CUDA_CALL_SINGLE(__##TYPE##__<float>, (float*)u.pointer, (float*)v.pointer, (float)dt, (float)param1, (float)param2, u.size);\
				break;\
			case MathDomain::Double:\
				CUDA_CALL_SINGLE(__##TYPE##__<double>, (double*)u.pointer, (double*)v.pointer, (double)dt, (double)param1, (double)param2, u.size);\
				break;\
			default:\
				return CudaKernelException::_NotImplementedException;\
		}

	switch (type)
	{
		case PatternType::FitzHughNagumo:
			APPLY_PATTERN_DYNAMIC(FitzHughNagumo);
			break;
		case PatternType::Thomas:
			APPLY_PATTERN_DYNAMIC(Thomas);
			break;
		case PatternType::Schnakenberg:
			APPLY_PATTERN_DYNAMIC(Schnakenberg);
			break;
		case PatternType::Brussellator:
			APPLY_PATTERN_DYNAMIC(Brussellator);
			break;
		case PatternType::GrayScott:
			APPLY_PATTERN_DYNAMIC(GrayScott);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
	}

#undef APPLY_PATTERN_DYNAMIC
		return cudaGetLastError();
    }
}

template <typename T>
GLOBAL void __FitzHughNagumo__(T* RESTRICT uNew, T* RESTRICT vNew, const T dt, const T param1, const T param2, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE

		const T u = uNew[i];
	    const T v = vNew[i];
		uNew[i] += dt * (u * (1.0 - u * u) - v + param1);
		vNew[i] += dt * param2 * (u - v);

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Thomas__(T* RESTRICT uNew, T* RESTRICT vNew, const T dt, const T param1, const T param2, const size_t sz)
{
	static constexpr T rho = { 13.0 };
	static constexpr T K = { 0.05 };
	static constexpr T alpha = { 1.5 };
	static constexpr T gamma = { 200.0 };

	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE
		const T u = uNew[i];
	    const T v = vNew[i];

		const T h = rho * u * v / (1.0 + u + K * u * u);

	    uNew[i] += dt * gamma * (param1 - u - h);
		vNew[i] += dt * gamma * (alpha * (param2 - v) - h);

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Schnakenberg__(T* RESTRICT uNew, T* RESTRICT vNew, const T dt, const T param1, const T param2, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE
		const T u = uNew[i];
	    const T v = vNew[i];
		const T u2v = u * u * v;

	    uNew[i] += dt * (param1 - u + u2v);
	    vNew[i] += dt * (param2 - u2v);

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __Brussellator__(T* RESTRICT uNew, T* RESTRICT vNew, const T dt, const T param1, const T param2, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE
		const T u = uNew[i];
	    const T v = vNew[i];
		const T u2v = u * u * v;

	    uNew[i] += dt * (param1 - (param2 + 1.0) * u + u2v);
		vNew[i] += dt * (param2 * u - u2v);

	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __GrayScott__(T* RESTRICT uNew, T* RESTRICT vNew, const T dt, const T param1, const T param2, const size_t sz)
{
	CUDA_FUNCTION_PROLOGUE

	CUDA_FOR_LOOP_PROLOGUE
		const T u = uNew[i];
	    const T v = vNew[i];
		const T uv2 = u * v * v;

	    uNew[i] += dt * (-uv2 + param1 * (1.0 - u));
		vNew[i] += dt * (uv2 - (param1 + param2) * v);

	CUDA_FOR_LOOP_EPILOGUE
}