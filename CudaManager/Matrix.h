#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "DeviceManager.h"
#include "../CudaKernels/Types.h"

#include "Vector.h"

namespace la
{
	class CUDAMANAGER_API CMatrix
	{
	public:
		friend class CTensor;

		CMatrix(size_t nRows, size_t nCols, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CMatrix(size_t nRows, size_t nCols, double value, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CMatrix(size_t nRows, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CMatrix(const CMatrix& rhs);

		CMatrix(const CVector& rhs);		

		void ReadFrom(const CMatrix& rhs);

		inline void ReadFrom(const std::vector<float>& rhs)
		{
			if (!buffer.pointer)
				throw std::exception("Buffer needs to be initialised first!");

			CMemoryTile rhsBuf((size_t)(rhs.data()), nRows(), nCols(), EMemorySpace::Host, EMathDomain::Double);
			dev::detail::AutoCopy(buffer, rhsBuf);
		}

		inline void ReadFrom(const std::vector<double>& rhs)
		{
			if (!buffer.pointer)
				throw std::exception("Buffer needs to be initialised first!");

			CMemoryTile rhsBuf((size_t)(rhs.data()), nRows(), nCols(), EMemorySpace::Host, EMathDomain::Double);
			dev::detail::AutoCopy(buffer, rhsBuf);
		}

		void ReadFrom(const CVector& rhs);

		inline void Set(double value)
		{
			dev::detail::Initialize(buffer, value);
		}

		void RandomUniform(unsigned seed = 1234);

		void RandomGaussian(unsigned seed = 1234);

		std::vector<double> Get() const;

		std::vector<double> Get(size_t column) const;

		void Print(const std::string& label = "") const;

		virtual ~CMatrix();

		inline size_t size() const noexcept { return buffer.size; }
		inline size_t nRows() const noexcept { return buffer.nRows; }
		inline size_t nCols() const noexcept { return buffer.nCols; }
		inline EMemorySpace memorySpace() const noexcept { return buffer.memorySpace; }
		inline EMathDomain mathDomain() const noexcept { return buffer.mathDomain; }
		inline CMemoryTile GetBuffer() const noexcept { return buffer; }

		std::vector<std::shared_ptr<CVector>> columns;

		#pragma region Linear Algebra

		CMatrix operator +(const CMatrix& rhs) const;
		CMatrix& operator +=(const CMatrix& rhs);
		CMatrix operator -(const CMatrix& rhs) const;
		CMatrix& operator -=(const CMatrix& rhs);

		CMatrix Add(const CMatrix& rhs, double alpha = 1.0) const;
		void AddEqual(const CMatrix& rhs, double alpha = 1.0);

		void Scale(double alpha);

		#pragma endregion

	protected:
		CMatrix(CMemoryTile buffer, bool isOwner = true);

		CMemoryTile buffer;

	private:
		bool isOwner = true;
	};

	CUDAMANAGER_API CMatrix MakeCopy(const CMatrix& source);
	CUDAMANAGER_API CMatrix RandomUniform(size_t nRows = 100, size_t nCols = 100, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float, unsigned seed = 1234);
	CUDAMANAGER_API CMatrix RandomGaussian(size_t nRows = 100, size_t nCols = 100, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float, unsigned seed = 1234);
	CUDAMANAGER_API void Print(const CMatrix& vec, const std::string& label = "");

	CUDAMANAGER_API CMatrix Add(const CMatrix& lhs, const CMatrix& rhs, double alpha = 1.0);
	CUDAMANAGER_API void Scale(CMatrix& lhs, double alpha);
}


