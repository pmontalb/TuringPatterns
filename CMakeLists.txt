cmake_minimum_required(VERSION 3.14)
project(TuringPatterns)

include(cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

# PdeFiniteDifference
add_subdirectory(PdeFiniteDifferenceSolver ${CMAKE_BINARY_DIR}/PdeFiniteDifferenceSolver EXCLUDE_FROM_ALL)

create_cuda_library(
	NAME
        TuringPatternKernels
	SOURCES
        TuringPatternKernels/PatternDynamic.cu
	DEPENDENCIES
		CudaLightKernels
	PUBLIC_INCLUDE_DIRECTORIES
		TuringPatternKernels
)

create_cuda_executable(
	NAME
        TuringPatterns
	SOURCES
        TuringPatterns/main.cu
	DEPENDENCIES
		TuringPatternKernels PdeFiniteDifferenceSolverManager
	PRIVATE_INCLUDE_DIRECTORIES
		../PdeFiniteDifferenceSolver
	SYSTEM_DEPENDENCIES
		forge -lGL
)

