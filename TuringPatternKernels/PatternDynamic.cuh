#pragma once

//#include <Common.cuh>
#include <Flags.cuh>
#include <Types.h>

#include "PatternType.h"

EXTERN_C
{
	/**
	*   [u, v]' += dt * [f(u,v); g(u,v)]'
	*/
	EXPORT int _ApplyPatternDynamic(MemoryBuffer u, MemoryBuffer v, const PatternType type, const double dt, const double param1, const double param2);
}

template <typename T>
GLOBAL void __FitzHughNagumo__(T* RESTRICT u, T* RESTRICT v, const T dt, const T param1, const T param2, const size_t sz);

template <typename T>
GLOBAL void __Thomas__(T* RESTRICT u, T* RESTRICT v, const T dt, const T param1, const T param2, const size_t sz);

template <typename T>
GLOBAL void __Schnakenberg__(T* RESTRICT u, T* RESTRICT v, const T dt, const T param1, const T param2, const size_t sz);

template <typename T>
GLOBAL void __Brussellator__(T* RESTRICT u, T* RESTRICT v, const T dt, const T param1, const T param2, const size_t sz);

template <typename T>
GLOBAL void __GrayScott__(T* RESTRICT u, T* RESTRICT v, const T dt, const T param1, const T param2, const size_t sz);
