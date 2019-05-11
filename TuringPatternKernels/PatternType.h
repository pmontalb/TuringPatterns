#pragma once

#include <Types.h>

EXTERN_C
{
	/**
	* Defines the time discretizer type
	*/
	enum class PatternType
	{
		Null = 0,

		FitzHughNagumo = 1,
		Thomas = 2,
		Schnakenberg = 3,
		Brussellator = 4,
		GrayScott = 5,
	};
}