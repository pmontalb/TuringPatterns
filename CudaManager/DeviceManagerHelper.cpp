#include "stdafx.h"
#include "DeviceManagerHelper.h"

#pragma region

#define IMPORT __declspec(dllimport)

#define __CREATE_FUNCTION_0_ARG(NAME)\
	EXTERN_C IMPORT int _##NAME();\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME()\
			{\
				int err = _##NAME();\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, TYPE0, ARG0)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0)\
			{\
				int err = _##NAME(ARG0);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0, TYPE1 ARG1);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1)\
			{\
				int err = _##NAME(ARG0, ARG1);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
		}\
	}

#define __CREATE_FUNCTION_7_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6)\
	EXTERN_C IMPORT int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6);\
	namespace dev\
	{\
		namespace detail\
		{\
			void CUDAMANAGER_API NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);\
				if (err != 0)\
					throw std::exception("Failed to call " #NAME);\
			}\
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
__CREATE_FUNCTION_2_ARG(RandNormal, CMemoryBuffer, buf, const  unsigned, seed);

// CuBlasWrapper
__CREATE_FUNCTION_4_ARG(Add, CMemoryBuffer, z, const CMemoryBuffer, x, const CMemoryBuffer, y, const double, value);
__CREATE_FUNCTION_3_ARG(AddEqual, CMemoryBuffer, z, const CMemoryBuffer, x, const double, value);
__CREATE_FUNCTION_2_ARG(Scale, CMemoryBuffer, z, const double, value);

// Finite Difference
__CREATE_FUNCTION_6_ARG(Iterate1D, CMemoryBuffer, uNew, const CMemoryBuffer, u, const CMemoryBuffer, grid, const double, dt, const double, diffusionCoefficient, const EBoundaryCondition, boundaryConditionType);
__CREATE_FUNCTION_7_ARG(Iterate2D, CMemoryTile, uNew, const CMemoryTile, u, const CMemoryBuffer, xGrid, const CMemoryBuffer, yGrid, const double, dt, const double, diffusionCoefficient, const EBoundaryCondition, boundaryConditionType);
__CREATE_FUNCTION_1_ARG(Iterate2DPattern, CPatternInput2D, input);
