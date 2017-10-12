#pragma once

#include "Common.cuh"
#include "Flags.cuh"
#include "Types.h"


EXTERN_C
{
	/**
	* z = x + y
	*/
	EXPORT int _Add(CMemoryBuffer z, const CMemoryBuffer x, const CMemoryBuffer y, const double alpha = 1.0);


	/**
	* z += x
	*/
	EXPORT int _AddEqual(CMemoryBuffer z, CMemoryBuffer x, double alpha = 1.0);


	/**
	* z *= alpha
	*/
	EXPORT int _Scale(CMemoryBuffer z, const double alpha);
}