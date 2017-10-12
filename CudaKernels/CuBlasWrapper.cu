#include "CuBlasWrapper.cuh"
#include "DeviceManager.cuh"
#include <cublas.h>

EXTERN_C
{
	/**
	* z = x + y
	*/
	EXPORT int _Add(CMemoryBuffer z, const CMemoryBuffer x, const CMemoryBuffer y, const double alpha)
	{
		const cublasHandle_t& handle = CublasHandle();

		switch (z.mathDomain)
		{
		case EMathDomain::Float:
			{
				int err = cublasScopy(handle, z.size, (float*)y.pointer, 1, (float*)z.pointer, 1);

				if (err)
					return err;

				const float _alpha = (float)alpha;
				return cublasSaxpy(handle, z.size, &_alpha, (float*)x.pointer, 1, (float*)z.pointer, 1);
			}
		case EMathDomain::Double:
			{
				int err = cublasDcopy(handle, z.size, (double*)y.pointer, 1, (double*)z.pointer, 1);

				if (err)
					return err;

				return cublasDaxpy(handle, z.size, &alpha, (double*)x.pointer, 1, (double*)z.pointer, 1);
			}
		default:
			return -1;
		}
	}

	/**
	* z += x
	*/
	EXPORT int _AddEqual(CMemoryBuffer z, CMemoryBuffer x, double alpha)
	{
		const cublasHandle_t& handle = CublasHandle();
		cublasHandle_t h;
		cublasCreate(&h);

		switch (z.mathDomain)
		{
		case EMathDomain::Float:
		{
			const float _alpha = (float)alpha;
			return cublasSaxpy(handle, z.size, &_alpha, (float*)x.pointer, 1, (float*)z.pointer, 1);
		}
		case EMathDomain::Double:
			return cublasDaxpy(h, z.size, &alpha, (double*)x.pointer, 1, (double*)z.pointer, 1);
		default:
			return -1;
		}
	}


	/**
	* z *= alpha
	*/
	EXPORT int _Scale(CMemoryBuffer z, const double alpha)
	{
		const cublasHandle_t& handle = CublasHandle();
		switch (z.mathDomain)
		{
		case EMathDomain::Float:
		{
			const float _alpha = (float)alpha;
			return cublasSscal(handle, z.size, &_alpha, (float*)z.pointer, 1);
		}
		case EMathDomain::Double:
			return cublasDscal(handle, z.size, &alpha, (double*)z.pointer, 1);
		default:
			return -1;
		}
	}

}