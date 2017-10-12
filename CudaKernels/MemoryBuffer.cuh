#pragma once

#include "Common.cuh"
#include "Flags.cuh"

#include "Types.h"

EXTERN_C
{
	EXPORT int _HostToHostCopy(CMemoryBuffer dest, const CMemoryBuffer source);

	EXPORT int _HostToDeviceCopy(CMemoryBuffer dest, const CMemoryBuffer source);

	EXPORT int _DeviceToHostCopy(CMemoryBuffer dest, const CMemoryBuffer source);

	EXPORT int _DeviceToDeviceCopy(CMemoryBuffer dest, const CMemoryBuffer source);

	EXPORT int _AutoCopy(CMemoryBuffer dest, const CMemoryBuffer source);

	EXPORT int _Alloc(CMemoryBuffer& ptr);

	EXPORT int _AllocHost(CMemoryBuffer& ptr);

	EXPORT int _Free(const CMemoryBuffer ptr);

	EXPORT int _FreeHost(const CMemoryBuffer ptr);
}