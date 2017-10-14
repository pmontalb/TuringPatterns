#include "Flags.cuh"

#pragma once

extern "C"
{
	enum class EMemorySpace
	{
		Null,
		Host,
		Device
	};

	enum class EMathDomain
	{
		Null,
		Float,
		Double
	};

	class CMemoryBuffer
	{
	public:
		ptr_t pointer;
		size_t size;
		EMemorySpace memorySpace;
		EMathDomain mathDomain;

		inline size_t GetElementarySize() const noexcept
		{
			switch (mathDomain)
			{
			case EMathDomain::Double:
				return sizeof(double);
			case EMathDomain::Float:
				return sizeof(float);
			default:
				return 0;
			}
		}

		inline size_t TotalSize() const noexcept
		{
			return size * GetElementarySize();
		}

		explicit CMemoryBuffer(const ptr_t pointer = 0, const size_t size = 0, EMemorySpace memorySpace = EMemorySpace::Null, EMathDomain mathDomain = EMathDomain::Null)
			: pointer(pointer), size(size), memorySpace(memorySpace), mathDomain(mathDomain)
		{

		}
	};

	class CMemoryTile : public CMemoryBuffer
	{
	public:
		size_t nRows;
		size_t nCols;

		explicit CMemoryTile(const ptr_t pointer = 0, const size_t nRows = 0, const size_t nCols = 0, EMemorySpace memorySpace = EMemorySpace::Null, EMathDomain mathDomain = EMathDomain::Null)
			: CMemoryBuffer(pointer, nRows * nCols, memorySpace, mathDomain), nRows(nRows), nCols(nCols)
		{

		}

	protected:
		explicit CMemoryTile(const ptr_t pointer = 0, const size_t nRows = 0, const size_t nCols = 0, const size_t size = 0, EMemorySpace memorySpace = EMemorySpace::Null, EMathDomain mathDomain = EMathDomain::Null)
			: CMemoryBuffer(pointer, size, memorySpace, mathDomain), nRows(nRows), nCols(nCols)
		{

		}
	};

	class CMemoryCube : public CMemoryTile
	{
	public:
		size_t nCubes;

		explicit CMemoryCube(const ptr_t pointer = 0, const size_t nRows = 0, const size_t nCols = 0, size_t nCubes = 0, EMemorySpace memorySpace = EMemorySpace::Null, EMathDomain mathDomain = EMathDomain::Null)
			: CMemoryTile(pointer, nRows, nCols, nRows * nCols * nCubes, memorySpace, mathDomain), nCubes(nCubes)
		{

		}
	};

	enum class EBoundaryCondition
	{
		Null,
		ZeroFlux,
		Periodic
	};
}