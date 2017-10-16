#include "stdafx.h"
#include "Pattern.h"

namespace fd
{
	CPatternData2D::CPatternData2D(const la::CVector& xGrid,
		const la::CVector& yGrid,
		const la::CMatrix& uInitialCondition,
		const la::CMatrix& vInitialCondition,
		const double uDiffusionCoefficient,
		const double vDiffusionCoefficient,
		const EPatternType patternType,
		const EBoundaryCondition boundaryCondition) noexcept
			: xGrid(xGrid), 
			yGrid(yGrid), 
			uInitialCondition(uInitialCondition), 
			vInitialCondition(vInitialCondition),
			uDiffusionCoefficient(uDiffusionCoefficient), 
			vDiffusionCoefficient(vDiffusionCoefficient),
			patternType(patternType),
			boundaryCondition(boundaryCondition)
	{
	}

	CPatternSolver2D::CPatternSolver2D(const CPatternData2D& input, const double dt)
		: input(input), dt(dt)
	{
	}

	void CPatternSolver2D::Initialize(la::CMatrix& uSolution, la::CMatrix& vSolution) const
	{
		uSolution.ReadFrom(input.uInitialCondition);
		vSolution.ReadFrom(input.vInitialCondition);
	}

	void CPatternSolver2D::Iterate(la::CMatrix& uSolution, la::CMatrix& vSolution, const size_t nIterations, const double patternParam1, const double patternParam2) const
	{
		la::CMatrix uBuffer(uSolution);
		la::CMatrix vBuffer(vSolution);

		la::CMatrix* uRunningBuffer[2] = { &uBuffer, &uSolution };
		la::CMatrix* vRunningBuffer[2] = { &vBuffer, &vSolution };

		CPatternInput2D solverInput(uSolution.GetBuffer(), vSolution.GetBuffer(),
			uBuffer.GetBuffer(), vBuffer.GetBuffer(),
			input.xGrid.GetBuffer(), input.yGrid.GetBuffer(),
			input.patternType, input.boundaryCondition,
			input.uDiffusionCoefficient, input.vDiffusionCoefficient, dt,
			patternParam1, patternParam2);

		for (size_t i = 0; i < nIterations; i++)
		{
			solverInput.uNew = uRunningBuffer[i & 1]->GetBuffer();
			solverInput.vNew = vRunningBuffer[i & 1]->GetBuffer();

			solverInput.u = uRunningBuffer[(i + 1) & 1]->GetBuffer();
			solverInput.v = vRunningBuffer[(i + 1) & 1]->GetBuffer();

			dev::detail::Iterate2DPattern(solverInput);
		}

		if ((nIterations & 1))
		{
			uSolution.ReadFrom(uBuffer);
			vSolution.ReadFrom(vBuffer);
		}
	}
}