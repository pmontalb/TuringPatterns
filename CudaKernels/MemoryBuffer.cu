#include "MemoryBuffer.cuh"
#include <stdio.h>

EXTERN_C
{
	EXPORT int _HostToHostCopy(CMemoryBuffer dest, const CMemoryBuffer source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyHostToHost);
	}

	EXPORT int _HostToDeviceCopy(CMemoryBuffer dest, const CMemoryBuffer source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyHostToDevice);
	}

	EXPORT int _DeviceToHostCopy(CMemoryBuffer dest, const CMemoryBuffer source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDeviceToHost);
	}

	EXPORT int _DeviceToDeviceCopy(CMemoryBuffer dest, const CMemoryBuffer source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDeviceToDevice);
	}

	EXPORT int _AutoCopy(CMemoryBuffer dest, const CMemoryBuffer source)
	{
		return cudaMemcpy((void *)dest.pointer, (void *)source.pointer, dest.TotalSize(), cudaMemcpyDefault);
	}

	EXPORT int _Alloc(CMemoryBuffer& buf)
	{
		int ret = cudaMalloc((void **)&buf.pointer, buf.TotalSize());
		return ret;
	}

	EXPORT int _AllocHost(CMemoryBuffer& buf)
	{
		int ret = cudaMallocHost((void **)&buf.pointer, buf.TotalSize());
		return ret;
	}

	EXPORT int _Free(const CMemoryBuffer buf)
	{
		cudaThreadSynchronize();
		return cudaFree((void *)buf.pointer);
	}

	EXPORT int _FreeHost(const CMemoryBuffer buf)
	{
		return cudaFreeHost((void *)buf.pointer);
	}
}