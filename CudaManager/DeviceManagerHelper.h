#pragma once

#include "../CudaKernels/Types.h"
#include <exception>

#pragma region Macro Utilities

#define __CREATE_FUNCTION_0_ARG(NAME)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME();\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, TYPE0, ARG0)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0);\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1);\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
		}\
	}

#pragma endregion

// Device
__CREATE_FUNCTION_1_ARG(GetDevice, int&, dev);
__CREATE_FUNCTION_0_ARG(ThreadSynchronize);
__CREATE_FUNCTION_1_ARG(SetDevice, const int, dev);
__CREATE_FUNCTION_0_ARG(GetDeviceStatus);
__CREATE_FUNCTION_1_ARG(GetBestDevice, int&, dev);

__CREATE_FUNCTION_1_ARG(GetDeviceCount, int&, count);
__CREATE_FUNCTION_2_ARG(HostToHostCopy, CMemoryBuffer, dest, const CMemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(HostToDeviceCopy, CMemoryBuffer, dest, const CMemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(DeviceToDeviceCopy, CMemoryBuffer, dest, const CMemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(AutoCopy, CMemoryBuffer, dest, const CMemoryBuffer, source);
__CREATE_FUNCTION_1_ARG(Alloc, CMemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(AllocHost, CMemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(Free, const CMemoryBuffer, buf);
__CREATE_FUNCTION_1_ARG(FreeHost, const CMemoryBuffer, buf);

// Initializer
__CREATE_FUNCTION_2_ARG(Initialize, CMemoryBuffer, buf, const double, value);
__CREATE_FUNCTION_3_ARG(LinSpace, CMemoryBuffer, buf, const double, x0, const double, x1);
__CREATE_FUNCTION_2_ARG(RandUniform, CMemoryBuffer, buf, const unsigned, seed);
__CREATE_FUNCTION_2_ARG(RandNormal, CMemoryBuffer, buf, const unsigned, seed);

// CuBlasWrappers
__CREATE_FUNCTION_4_ARG(Add, CMemoryBuffer, z, const CMemoryBuffer, x, const CMemoryBuffer, y, const double, value);
__CREATE_FUNCTION_3_ARG(AddEqual, CMemoryBuffer, z, const CMemoryBuffer, x, const double, value);
__CREATE_FUNCTION_2_ARG(Scale, CMemoryBuffer, z, const double, value);

// FiniteDifference
__CREATE_FUNCTION_6_ARG(Iterate1D, CMemoryBuffer, uNew, const CMemoryBuffer, u, const CMemoryBuffer, grid, const double, dt, const double, diffusionCoefficient, const EBoundaryCondition, boundaryConditionType);

///**
//* z = x + y
//*/
//EXPORT int _Add(CMemoryBuffer z, const CMemoryBuffer x, const CMemoryBuffer y, const double alpha = 1.0);
//
//
///**
//* z += x
//*/
//EXPORT int _AddEqual(CMemoryBuffer z, const CMemoryBuffer x, const double alpha = 1.0);
//
//
///**
//* z *= alpha
//*/
//EXPORT int _Scale(CMemoryBuffer z, const double alpha);

#pragma region Undef macros

#undef __CREATE_FUNCTION_0_ARG
#undef __CREATE_FUNCTION_1_ARG
#undef __CREATE_FUNCTION_2_ARG
#undef __CREATE_FUNCTION_3_ARG
#undef __CREATE_FUNCTION_4_ARG

#pragma endregion