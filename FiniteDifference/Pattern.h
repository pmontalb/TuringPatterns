#pragma once

#include "Common.h"
#include "../CudaKernels/Types.h"
#include "../CudaManager/Vector.h"
#include "../CudaManager/Matrix.h"

namespace fd
{
	class FINITEDIFFERENCE_API CPatternData2D
	{
	public:
		CPatternData2D(const la::CVector& xGrid,
			const la::CVector& yGrid,
			const la::CMatrix& uInitialCondition,
			const la::CMatrix& vInitialCondition,
			const double uDiffusionCoefficient,
			const double vDiffusionCoefficient,
			const EPatternType patternType,
			const EBoundaryCondition boundaryCondition) noexcept;

		const la::CVector& xGrid;
		const la::CVector& yGrid;
		const la::CMatrix& uInitialCondition;
		const la::CMatrix& vInitialCondition;
		const double uDiffusionCoefficient;
		const double vDiffusionCoefficient;
		const EPatternType patternType;
		const EBoundaryCondition boundaryCondition;
	};

	class FINITEDIFFERENCE_API CPatternSolver2D
	{
	public:
		CPatternSolver2D(const CPatternData2D& input, const double dt);

		void Initialize(la::CMatrix& uSolution, la::CMatrix& vSolution) const;
		void Iterate(la::CMatrix& uSolution, la::CMatrix& vSolution, const size_t nIterations, const double patternParam1 = 0.0, const double patternParam2 = 0.0) const;


	protected:
		const CPatternData2D& input;	
		const double dt;
	};
}
