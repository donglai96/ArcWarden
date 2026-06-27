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

# ---- tests ----
enable_testing()
add_test(NAME smoke_device COMMAND arcwarden)

# Step 6 [S1]: FFT round trip + Parseval (links cuFFT).
add_executable(test_fft_roundtrip tests/test_fft_roundtrip.cu)
target_include_directories(test_fft_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_fft_roundtrip PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME fft_roundtrip COMMAND test_fft_roundtrip)

# Step 8 [S5]: spectral Poisson on a single sine mode.
add_executable(test_poisson_sine tests/test_poisson_sine.cu)
target_include_directories(test_poisson_sine PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_poisson_sine PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME poisson_sine COMMAND test_poisson_sine)

# Step 9: particle loading (quiet Maxwellian) + migrate.
add_executable(test_particle_load tests/test_particle_load.cu)
target_include_directories(test_particle_load PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_particle_load PRIVATE CUDA::cudart)
add_test(NAME particle_load COMMAND test_particle_load)

# Step 10 [S4]: charge conservation (CIC deposit).
add_executable(test_charge_conservation tests/test_charge_conservation.cu)
target_include_directories(test_charge_conservation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_charge_conservation PRIVATE CUDA::cudart)
add_test(NAME charge_conservation COMMAND test_charge_conservation)

# Step 11 [S6]: single-particle Boris trajectories (uniform E; gyration).
add_executable(test_single_particle tests/test_single_particle.cu)
target_include_directories(test_single_particle PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_single_particle PRIVATE CUDA::cudart)
add_test(NAME single_particle COMMAND test_single_particle)
