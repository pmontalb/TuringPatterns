#pragma once

#include <vector>
#include <string>

#include "DeviceManager.h"
#include "../CudaKernels/Types.h"

namespace la
{
	class CUDAMANAGER_API CVector
	{
	public:
		friend class CMatrix;

		CVector(size_t size, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CVector(size_t size, double value, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CVector(const CVector& rhs);

		void ReadFrom(const CVector& rhs);

		inline void ReadFrom(const std::vector<float>& rhs)
		{
			if (!buffer.pointer)
				throw std::exception("Buffer needs to be initialised first!");

			CMemoryBuffer rhsBuf((size_t)(rhs.data()), rhs.size(), EMemorySpace::Host, EMathDomain::Float);
			dev::detail::AutoCopy(buffer, rhsBuf);
		}

		inline void ReadFrom(const std::vector<double>& rhs)
		{
			if (!buffer.pointer)
				throw std::exception("Buffer needs to be initialised first!");

			CMemoryBuffer rhsBuf((size_t)(rhs.data()), rhs.size(), EMemorySpace::Host, EMathDomain::Double);
			dev::detail::AutoCopy(buffer, rhsBuf);
		}

		void LinSpace(double x0, double x1);

		void RandomUniform(unsigned seed = 1234);

		void RandomGaussian(unsigned seed = 1234);

		std::vector<double> Get() const;

		void Print(const std::string& label = "") const;

		virtual ~CVector();

		inline size_t size() const noexcept { return buffer.size; }
		inline EMemorySpace memorySpace() const noexcept { return buffer.memorySpace; }
		inline EMathDomain mathDomain() const noexcept { return buffer.mathDomain; }
		inline CMemoryBuffer GetBuffer() const noexcept { return buffer; }

		#pragma region Linear Algebra

		CVector operator +(const CVector& rhs) const;
		CVector& operator +=(const CVector& rhs);
		CVector operator -(const CVector& rhs) const;
		CVector& operator -=(const CVector& rhs);

		CVector Add(const CVector& rhs, double alpha = 1.0) const;
		CVector& AddEqual(const CVector& rhs, double alpha = 1.0);

		CVector& Scale(double alpha);

		#pragma endregion

	protected:
		CVector(CMemoryBuffer buffer, bool isOwner = true);

		CMemoryBuffer buffer;

	private:
		bool isOwner = true;
	};

	CUDAMANAGER_API CVector MakeCopy(const CVector& source);
	CUDAMANAGER_API CVector LinSpace(double x0, double x1, size_t size = 100, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);
	CUDAMANAGER_API CVector RandomUniform(size_t size = 100, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float, unsigned seed = 1234);
	CUDAMANAGER_API CVector RandomGaussian(size_t size = 100, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float, unsigned seed = 1234);
	CUDAMANAGER_API void Print(const CVector& vec, const std::string& label = "");

	CUDAMANAGER_API CVector Add(const CVector& lhs, const CVector& rhs, double alpha = 1.0);
	CUDAMANAGER_API void Scale(CVector& lhs, double alpha);
}

