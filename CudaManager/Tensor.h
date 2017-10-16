#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "DeviceManager.h"
#include "../CudaKernels/Types.h"

#include "Matrix.h"

namespace la
{
	class CUDAMANAGER_API CTensor
	{
	public:
		CTensor(size_t nRows, size_t nCols, size_t nMatrices, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CTensor(size_t nRows, size_t nCols, size_t nMatrices, double value, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CTensor(size_t nRows, size_t nMatrices, EMemorySpace memorySpace = EMemorySpace::Device, EMathDomain mathDomain = EMathDomain::Float);

		CTensor(const CTensor& rhs);

		CTensor(const CMatrix& rhs);

		CTensor(const CVector& rhs);

		std::vector<std::vector<std::vector<double>>> Get() const;

		std::vector<std::vector<double>> Get(size_t matrix) const;

		void Print(const std::string& label = "") const;

		inline void ToCsv(const std::string& fileName) const
		{
			auto mat = Get();

			std::ofstream csvFile;
			csvFile.open(fileName);

			for (size_t m = 0; m < mat.size(); m++)
			{
				for (size_t col = 0; col < mat[m].size(); col++)
				{
					for (size_t row = 0; row < mat[m][col].size(); row++)
					{
						csvFile << mat[m][col][row] << ",";
					}

					csvFile << std::endl;
				}

				csvFile << std::endl;
			}

			csvFile.close();
		}

		virtual ~CTensor();

		inline size_t size() const noexcept { return buffer.size; }
		inline size_t nRows() const noexcept { return buffer.nRows; }
		inline size_t nCols() const noexcept { return buffer.nCols; }
		inline size_t nMatrices() const noexcept { return buffer.nCubes; }
		inline EMemorySpace memorySpace() const noexcept { return buffer.memorySpace; }
		inline EMathDomain mathDomain() const noexcept { return buffer.mathDomain; }
		inline CMemoryTile GetBuffer() const noexcept { return buffer; }

		std::vector<std::shared_ptr<CMatrix>> matrices;

	protected:
		CTensor(CMemoryCube buffer, bool isOwner = true);

		CMemoryCube buffer;

	private:
		bool isOwner = true;
	};

	CUDAMANAGER_API void Print(const CTensor& vec, const std::string& label = "");
}
