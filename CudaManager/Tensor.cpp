#include "stdafx.h"
#include "Tensor.h"

#include <iostream>

namespace la
{
	CTensor::CTensor(size_t nRows, size_t nCols, size_t nMatrices, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CTensor(CMemoryCube(0, nRows, nCols, nMatrices, memorySpace, mathDomain))
	{

	}

	CTensor::CTensor(size_t nRows, size_t nCols, size_t nMatrices, double value, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CTensor(nRows, nCols, nMatrices, memorySpace, mathDomain)
	{
		dev::detail::Initialize(buffer, value);
	}

	CTensor::CTensor(size_t nRows, size_t nMatrices, EMemorySpace memorySpace, EMathDomain mathDomain)
		: CTensor(nRows, nRows, nMatrices, memorySpace, mathDomain)
	{

	}

	CTensor::CTensor(CMemoryCube buffer, bool isOwner)
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

		matrices.resize(nMatrices());
		for (size_t i = 0; i < nMatrices(); i++)
		{
			const size_t matrixShift = i * nRows() * nCols() * this->buffer.GetElementarySize();
			CMemoryTile matrixBuffer(this->buffer.pointer + matrixShift, buffer.nRows, buffer.nCols, buffer.memorySpace, buffer.mathDomain);
			matrices[i] = std::shared_ptr<CMatrix>(new CMatrix(matrixBuffer, false));
		}
	}

	CTensor::CTensor(const CTensor& rhs)
		: CTensor(rhs.buffer)  // allocations are carried out inside here!
	{
		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	CTensor::CTensor(const CMatrix& rhs)
		: CTensor(CMemoryCube(0, rhs.nRows(), rhs.nCols(), 1, rhs.buffer.memorySpace, rhs.buffer.mathDomain))
	{
		dev::detail::AutoCopy(buffer, rhs.buffer);
	}

	CTensor::CTensor(const CVector& rhs)
		: CTensor(CMemoryCube(0, 1, rhs.size(), 1, rhs.GetBuffer().memorySpace, rhs.GetBuffer().mathDomain))
	{
		dev::detail::AutoCopy(buffer, rhs.GetBuffer());
	}

	std::vector<std::vector<std::vector<double>>> CTensor::Get() const
	{
		dev::detail::ThreadSynchronize();

		CMemoryBuffer newBuf(buffer);
		newBuf.memorySpace = EMemorySpace::Host;

		dev::detail::AllocHost(newBuf);
		dev::detail::AutoCopy(newBuf, buffer);

		std::vector<std::vector<std::vector<double>>> ret(nMatrices());
		for (size_t i = 0; i < nMatrices(); i++)
		{
			ret[i].resize(nCols());
			for (size_t j = 0; j < nCols(); j++)
				ret[i][j].resize(nRows());
		}

		switch (buffer.mathDomain)
		{
		case EMathDomain::Double:
		{
			double* ptr = (double*)newBuf.pointer;
			for (size_t k = 0; k < nMatrices(); k++)
				for (size_t j = 0; j < nCols(); j++)
					for (size_t i = 0; i < nRows(); i++)
						ret[k][j][i] = ptr[i + nRows() * (j + nCols()* k)];
		}
		break;
		case EMathDomain::Float:
		{
			float* ptr = (float*)newBuf.pointer;
			for (size_t k = 0; k < nMatrices(); k++)
				for (size_t j = 0; j < nCols(); j++)
					for (size_t i = 0; i < nRows(); i++)
						ret[k][j][i] = ptr[i + nRows() * (j + nCols()* k)];
		}
		break;
		default:
			break;
		}

		dev::detail::FreeHost(newBuf);

		return ret;
	}

	std::vector<std::vector<double>> CTensor::Get(size_t matrix) const
	{
		return matrices[matrix]->Get();
	}

	void CTensor::Print(const std::string& label) const
	{
		auto mat = Get();

		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t k = 0; k < mat.size(); k++)
		{
			for (size_t j = 0; j < mat[k].size(); j++)
			{
				std::cout << "\t";
				for (size_t i = 0; i < mat[k][j].size(); i++)
					std::cout << " v[" << i << "][" << j << "][" << k << "] = " << mat[k][j][i];
				std::cout << std::endl;
			}
		}
		std::cout << "**********************" << std::endl;
	}

	CTensor::~CTensor()
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

	CUDAMANAGER_API void Print(const CTensor& ten, const std::string& label)
	{
		ten.Print(label);
	}
}
