#include "stdafx.h"
#include "Matrix.h"

#include <iostream>

namespace la
{
	CMatrix::CMatrix(size_t nRows, size_t nCols, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CMatrix(CMemoryTile(0, nRows, nCols, memorySpace, mathDomain))
	{

	}

	CMatrix::CMatrix(size_t nRows, size_t nCols, double value, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CMatrix(nRows, nCols, memorySpace, mathDomain)
	{
		dev::detail::Initialize(buffer, value);
	}

	CMatrix::CMatrix(size_t nRows, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CMatrix(nRows, nRows, memorySpace, mathDomain)
	{

	}

	CMatrix::CMatrix(CMemoryTile buffer, bool isOwner)
		: buffer(buffer), isOwner(isOwner)
	{
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

		columns.resize(nCols());
		for (size_t i = 0; i < nCols(); i++)
		{
			const size_t colShift = i * nRows() * this->buffer.GetElementarySize();
			CMemoryBuffer colBuffer(this->buffer.pointer + colShift, buffer.nRows, buffer.memorySpace, buffer.mathDomain);
			columns[i] = std::shared_ptr<CVector>(new CVector(colBuffer, false));
		}
	}

	CMatrix::CMatrix(const CMatrix& rhs)
		: CMatrix(rhs.buffer)  // allocations are carried out inside here!
	{
		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	CMatrix::CMatrix(const CVector& rhs)
		: CMatrix(CMemoryTile(0, 1, rhs.size(), rhs.buffer.memorySpace, rhs.buffer.mathDomain))
	{
		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	void CMatrix::ReadFrom(const CMatrix& rhs)
	{
		if (!buffer.pointer)
			throw std::exception("Buffer needs to be initialised first!");

		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	void CMatrix::ReadFrom(const CVector& rhs)
	{
		if (!buffer.pointer)
			throw std::exception("Buffer needs to be initialised first!");

		dev::detail::AutoCopy(columns[0]->buffer, rhs.buffer);
	}

	std::vector<std::vector<double>> CMatrix::Get() const
	{
		dev::detail::ThreadSynchronize();

		CMemoryBuffer newBuf(buffer);
		newBuf.memorySpace = EMemorySpace::Host;

		dev::detail::AllocHost(newBuf);
		dev::detail::AutoCopy(newBuf, buffer);

		std::vector<std::vector<double>> ret(nCols());
		for (size_t i = 0; i < nCols(); i++)
			ret[i].resize(nRows());
		
		switch (buffer.mathDomain)
		{
		case EMathDomain::Double:
		{
			double* ptr = (double*)newBuf.pointer;
			for (size_t j = 0; j < nCols(); j++)
				for (size_t i = 0; i < nRows(); i++)
					ret[j][i] = ptr[i + nRows() * j];
		}
		break;
		case EMathDomain::Float:
		{
			float* ptr = (float*)newBuf.pointer;
			for (size_t j = 0; j < nCols(); j++)
				for (size_t i = 0; i < nRows(); i++)
					ret[j][i] = ptr[i + nRows() * j];
		}
		break;
		default:
			break;
		}

		dev::detail::FreeHost(newBuf);

		return ret;
	}

	std::vector<double> CMatrix::Get(size_t column) const
	{
		dev::detail::ThreadSynchronize();

		CMemoryBuffer newBuf(buffer);
		newBuf.memorySpace = EMemorySpace::Host;

		dev::detail::AllocHost(newBuf);
		dev::detail::AutoCopy(newBuf, buffer);

		std::vector<double> ret(nRows());

		switch (buffer.mathDomain)
		{
		case EMathDomain::Double:
		{
			double* ptr = (double*)newBuf.pointer;
			for (size_t i = 0; i < nRows(); i++)
				ret[i] = ptr[i + nRows() * column];
		}
		case EMathDomain::Float:
		{
			float* ptr = (float*)newBuf.pointer;
			for (size_t i = 0; i < nRows(); i++)
				ret[i] = ptr[i + nRows() * column];
		}
		default:
			break;
		}

		dev::detail::FreeHost(newBuf);

		return ret;
	}

	void CMatrix::Print(const std::string& label) const
	{
		auto mat = Get();

		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t j = 0; j < mat.size(); j++)
		{
			std::cout << "\t";
			for (size_t i = 0; i < mat[j].size(); i++)
				std::cout << " v[" << i << "][" << j << "] = " << mat[j][i];
			std::cout << std::endl;
		}
		std::cout << "**********************" << std::endl;
	}

	void CMatrix::RandomUniform(unsigned seed)
	{
		dev::detail::RandUniform(buffer, seed);
	}

	void CMatrix::RandomGaussian(unsigned seed)
	{
		dev::detail::RandNormal(buffer, seed);
	}

	CMatrix::~CMatrix()
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

	CMatrix CMatrix::operator +(const CMatrix& rhs) const
	{
		CMatrix ret(*this);
		dev::detail::AddEqual(ret.buffer, rhs.buffer, 1.0);

		return ret;
	}

	CMatrix& CMatrix::operator +=(const CMatrix& rhs)
	{
		dev::detail::AddEqual(buffer, rhs.buffer, 1.0);
		return *this;
	}

	CMatrix CMatrix::operator -(const CMatrix& rhs) const
	{
		CMatrix ret(*this);
		dev::detail::AddEqual(ret.buffer, rhs.buffer, -1.0);

		return ret;
	}

	CMatrix& CMatrix::operator -=(const CMatrix& rhs)
	{
		dev::detail::AddEqual(buffer, rhs.buffer, -1.0);
		return *this;
	}

	CMatrix CMatrix::Add(const CMatrix& rhs, double alpha) const
	{
		CMatrix ret(*this);
		dev::detail::AddEqual(ret.buffer, rhs.buffer, alpha);

		return ret;
	}

	void CMatrix::AddEqual(const CMatrix& rhs, double alpha)
	{
		dev::detail::AddEqual(buffer, rhs.buffer, alpha);
	}

	void CMatrix::Scale(double alpha)
	{
		dev::detail::Scale(buffer, alpha);
	}

#pragma endregion

	CUDAMANAGER_API CMatrix MakeCopy(const CMatrix& source)
	{
		CMatrix ret(source);
		return ret;
	}

	CUDAMANAGER_API CMatrix RandomUniform(size_t nRows, size_t nCols, EMemorySpace memorySpace, EMathDomain mathDomain, unsigned seed)
	{
		CMatrix ret(nRows, nCols, memorySpace, mathDomain);
		ret.RandomUniform(seed);

		return ret;
	}

	CUDAMANAGER_API CMatrix RandomGaussian(size_t nRows, size_t nCols, EMemorySpace memorySpace, EMathDomain mathDomain, unsigned seed)
	{
		CMatrix ret(nRows, nCols, memorySpace, mathDomain);
		ret.RandomGaussian(seed);

		return ret;
	}

	CUDAMANAGER_API void Print(const CMatrix& vec, const std::string& label)
	{
		vec.Print(label);
	}

	CUDAMANAGER_API CMatrix Add(const CMatrix& lhs, const CMatrix& rhs, double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	CUDAMANAGER_API void Scale(CMatrix& lhs, double alpha)
	{
		lhs.Scale(alpha);
	}
}
