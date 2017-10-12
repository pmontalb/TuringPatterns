#include "stdafx.h"
#include "Vector.h"

#include <iostream>

namespace la
{
	CVector::CVector(size_t size, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CVector(CMemoryBuffer(0, size, memorySpace, mathDomain))
	{

	}

	CVector::CVector(size_t size, double value, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CVector(CMemoryBuffer(0, size, memorySpace, mathDomain))
	{
		dev::detail::Initialize(buffer, value);
	}

	CVector::CVector(CMemoryBuffer buffer, bool isOwner)
		: buffer(buffer), isOwner(isOwner)
	{
		// if is not the owner it has already been allocated!
		if (!isOwner)
		{
			if (!buffer.pointer)
				throw std::exception("Pointer must be allocated first!");
			return;
		}

		switch (buffer.memorySpace)
		{
		case EMemorySpace::Device:
			dev::detail::Alloc(this->buffer);
			break;
		case EMemorySpace::Host:
			dev::detail::AllocHost(this->buffer);
			break;
		default:
			break;
		}
	}

	CVector::CVector(const CVector& rhs)
		: CVector(rhs.buffer) // allocations are carried out inside here!
	{
		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	void CVector::ReadFrom(const CVector& rhs)
	{
		if (!buffer.pointer)
			throw std::exception("Buffer needs to be initialised first!");

		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	std::vector<double> CVector::Get() const
	{
		dev::detail::ThreadSynchronize();

		CMemoryBuffer newBuf(buffer);
		newBuf.memorySpace = EMemorySpace::Host;

		dev::detail::AllocHost(newBuf);
		dev::detail::AutoCopy(newBuf, buffer);

		std::vector<double> ret(buffer.size, -123);
		switch (buffer.mathDomain)
		{
		case EMathDomain::Double:
		{
			double* ptr = (double*)newBuf.pointer;
			for (size_t i = 0; i < buffer.size; i++)
				ret[i] = ptr[i];
		}
		case EMathDomain::Float:
		{
			float* ptr = (float*)newBuf.pointer;
			for (size_t i = 0; i < buffer.size; i++)
				ret[i] = ptr[i];
		}
		default:
			break;
		}

		dev::detail::FreeHost(newBuf);

		return ret;
	}

	void CVector::Print(const std::string& label) const
	{
		auto vec = Get();

		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t i = 0; i < vec.size(); i++)
			std::cout << "\tv[" << i << "] \t=\t " << vec[i] << std::endl;
		std::cout << "**********************" << std::endl;
	}

	void CVector::LinSpace(double x0, double x1)
	{
		dev::detail::LinSpace(buffer, x0, x1);
	}

	void CVector::RandomUniform(unsigned seed)
	{
		dev::detail::RandUniform(buffer, seed);
	}

	void CVector::RandomGaussian(unsigned seed)
	{
		dev::detail::RandNormal(buffer, seed);
	}

	CVector::~CVector()
	{
		// if this is not the owner of the buffer, it must not free it
		if (!isOwner)
			return;

		switch (buffer.memorySpace)
		{
		case EMemorySpace::Device:
			dev::detail::Free(this->buffer);
			break;
		case EMemorySpace::Host:
			dev::detail::FreeHost(this->buffer);
			break;
		default:
			break;
		}
	}

#pragma region Linear Algebra

	CVector CVector::operator +(const CVector& rhs) const
	{
		CVector ret(*this);
		dev::detail::AddEqual(ret.buffer, rhs.buffer, 1.0);

		return ret;
	}

	CVector& CVector::operator +=(const CVector& rhs) 
	{
		dev::detail::AddEqual(buffer, rhs.buffer, 1.0);
		return *this;
	}

	CVector CVector::operator -(const CVector& rhs) const
	{
		CVector ret(*this);
		dev::detail::AddEqual(ret.buffer, rhs.buffer, -1.0);

		return ret;
	}
	
	CVector& CVector::operator -=(const CVector& rhs)
	{
		dev::detail::AddEqual(buffer, rhs.buffer, -1.0);
		return *this;
	}

	CVector CVector::Add(const CVector& rhs, double alpha) const
	{
		CVector ret(*this);
		dev::detail::AddEqual(ret.buffer, rhs.buffer, alpha);

		return ret;
	}

	CVector& CVector::AddEqual(const CVector& rhs, double alpha)
	{
		dev::detail::AddEqual(buffer, rhs.buffer, alpha);
		return *this;
	}

	CVector& CVector::Scale(double alpha)
	{
		dev::detail::Scale(buffer, alpha);
		return *this;
	}

#pragma endregion

	CUDAMANAGER_API CVector MakeCopy(const CVector& source)
	{
		CVector ret(source);
		return ret;
	}

	CUDAMANAGER_API CVector LinSpace(double x0, double x1, size_t size, EMemorySpace memorySpace, EMathDomain mathDomain)
	{
		CVector ret(size, memorySpace, mathDomain);
		ret.LinSpace(x0, x1);

		return ret;
	}

	CUDAMANAGER_API CVector RandomUniform(size_t size, EMemorySpace memorySpace, EMathDomain mathDomain, unsigned seed)
	{
		CVector ret(size, memorySpace, mathDomain);
		ret.RandomUniform(seed);

		return ret;
	}

	CUDAMANAGER_API CVector RandomGaussian(size_t size, EMemorySpace memorySpace, EMathDomain mathDomain, unsigned seed)
	{
		CVector ret(size, memorySpace, mathDomain);
		ret.RandomGaussian(seed);

		return ret;
	}

	CUDAMANAGER_API void Print(const CVector& vec, const std::string& label)
	{
		vec.Print(label);
	}

	CUDAMANAGER_API CVector Add(const CVector& lhs, const CVector& rhs, double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	CUDAMANAGER_API void Scale(CVector& lhs, double alpha)
	{
		lhs.Scale(alpha);
	}
}
