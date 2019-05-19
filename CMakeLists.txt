cmake_minimum_required(VERSION 3.14)
project(TuringPatterns)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc CACHE PATH "" FORCE)
enable_language(CUDA)

# PdeFiniteDifference
include_directories(../PdeFiniteDifferenceSolver)
add_subdirectory(../PdeFiniteDifferenceSolver ${CMAKE_BINARY_DIR}/PdeFiniteDifferenceSolver EXCLUDE_FROM_ALL)

include_directories(../CudaLight/CudaLight)
include_directories(../CudaLightKernels)
include_directories(../NpyCpp/Npy++)
include_directories(../NpyCpp/MemoryMapping)
include_directories(../PdeFiniteDifferenceKernels)
include_directories(../PdeFiniteDifferenceSolver/PdeFiniteDifferenceSolverManager)
include_directories(../forge)
include_directories(../forge/include)
include_directories(.)
include_directories(TuringPatternKernels)

add_library(
        TuringPatternKernels
        TuringPatternKernels/PatternDynamic.cu
)
target_compile_options(TuringPatternKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_61,code=sm_61 -m64 -restrict --compiler-options -Wall>)
target_compile_options(TuringPatternKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -Wno-unknown-pragmas>)
target_link_libraries(TuringPatternKernels -lcudart -lcublas -lcusolver -lcusparse)

add_executable(
        TuringPatterns
        TuringPatterns/main.cu
)
target_compile_options(TuringPatterns PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_61,code=sm_61 -m64 -restrict --compiler-options -Wall>)
target_compile_options(TuringPatterns PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -Wno-unknown-pragmas>)
target_link_libraries(TuringPatterns forge TuringPatternKernels CudaLight PdeFiniteDifferenceSolverManager -lcudart -lcublas -lcusolver -lcusparse)

