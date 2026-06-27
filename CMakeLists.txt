cmake_minimum_required(VERSION 3.24)

# The system default `nvcc` (/usr/bin -> /usr/lib/nvidia-cuda-toolkit, CUDA 12.0)
# is too old for the RTX 5090 (sm_120). Prefer the local 13.x toolkit unless the
# user pinned one. Must run before project(... CUDA).
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  foreach(_cuda_root /usr/local/cuda-13.3 /usr/local/cuda-13.1 /usr/local/cuda)
    if(EXISTS ${_cuda_root}/bin/nvcc)
      set(CMAKE_CUDA_COMPILER ${_cuda_root}/bin/nvcc)
      break()
    endif()
  endforeach()
endif()

# RTX 5090 = compute capability 12.0 (sm_120). "native" lets CMake/nvcc detect
# the installed GPU; override with -DCMAKE_CUDA_ARCHITECTURES=120.
#
# This MUST be set before project(... CUDA): enabling the CUDA language inside
# project() initializes CMAKE_CUDA_ARCHITECTURES to a compiler default (was 75
# here), after which a `NOT DEFINED` guard would never fire. Setting it first
# lets CMake resolve the `native` keyword during compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(ArcWarden LANGUAGES CXX CUDA)

# ---- standards ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# cudart / cufft via modern imported targets (CUDA::cudart, CUDA::cufft)
find_package(CUDAToolkit REQUIRED)

# ---- main executable (Step 1: just prints the GPU) ----
add_executable(arcwarden src/main.cpp)
target_include_directories(arcwarden PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arcwarden PRIVATE CUDA::cudart CUDA::cufft)

# ---- tests (placeholder smoke test; real test exes added in later steps) ----
enable_testing()
add_test(NAME smoke_device COMMAND arcwarden)
