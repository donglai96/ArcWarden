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

# ---- visualization tools (not tests) ----
add_executable(two_stream_movie tools/two_stream_movie.cu)
target_include_directories(two_stream_movie PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(two_stream_movie PRIVATE CUDA::cudart CUDA::cufft CUDA::nvml)

add_executable(bump_on_tail_movie tools/bump_on_tail_movie.cu)
target_include_directories(bump_on_tail_movie PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bump_on_tail_movie PRIVATE CUDA::cudart CUDA::cufft CUDA::nvml)

# ---- unified deck-driven runner (physics + modular diagnostics from one .ini) ----
add_executable(arcsim tools/arcsim.cu)
target_include_directories(arcsim PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arcsim PRIVATE CUDA::cudart CUDA::cufft)

# ---- deck-driven runner (physics from a text input deck, no recompile) ----
add_executable(run_deck tools/run_deck.cu)
target_include_directories(run_deck PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(run_deck PRIVATE CUDA::cudart CUDA::cufft)

# ---- deposit microbenchmark (atomic-global vs shared-tile) ----
add_executable(whistler_pump tools/whistler_pump.cu)
target_include_directories(whistler_pump PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whistler_pump PRIVATE CUDA::cudart CUDA::cufft)

# ---- 1D field-aligned electron-hybrid chorus runner (L-shell plan, M4 gate) ----
add_executable(chirp1d tools/chirp1d.cu)
target_include_directories(chirp1d PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(chirp1d PRIVATE CUDA::cudart CUDA::cufft)

add_executable(deposit_bench tools/deposit_bench.cu)
target_include_directories(deposit_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(deposit_bench PRIVATE CUDA::cudart CUDA::cufft)

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

# Step 12: diagnostic reductions (energy/charge/max|E|).
add_executable(test_diagnostics tests/test_diagnostics.cu)
target_include_directories(test_diagnostics PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_diagnostics PRIVATE CUDA::cudart)
add_test(NAME diagnostics COMMAND test_diagnostics)

# Step 13 [S7]: end-to-end cold Langmuir oscillation (links cuFFT).
add_executable(test_langmuir tests/test_langmuir.cu)
target_include_directories(test_langmuir PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_langmuir PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME langmuir COMMAND test_langmuir)

# Step 14 [S9]: two-stream instability growth rate + trapping.
add_executable(test_two_stream tests/test_two_stream.cu)
target_include_directories(test_two_stream PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_two_stream PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME two_stream COMMAND test_two_stream)

# Step 14 [S8]: long-run energy/charge conservation.
add_executable(test_conservation_longrun tests/test_conservation_longrun.cu)
target_include_directories(test_conservation_longrun PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_conservation_longrun PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME conservation_longrun COMMAND test_conservation_longrun)

# Darwin GATE 1: magnetostatic B-solve (J -> B) + current deposit.
add_executable(test_darwin_magnetostatic tests/test_darwin_magnetostatic.cu)
target_include_directories(test_darwin_magnetostatic PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_darwin_magnetostatic PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME darwin_magnetostatic COMMAND test_darwin_magnetostatic)

# chirp1d gate A: single-particle mirror bounce + mu conservation in B0(h).
add_executable(test_chirp1d_mirror tests/test_chirp1d_mirror.cu)
target_include_directories(test_chirp1d_mirror PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_chirp1d_mirror PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME chirp1d_mirror COMMAND test_chirp1d_mirror)

# chirp1d gate B: cold whistler eigenmode frequency vs cold dispersion.
add_executable(test_chirp1d_dispersion tests/test_chirp1d_dispersion.cu)
target_include_directories(test_chirp1d_dispersion PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_chirp1d_dispersion PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME chirp1d_dispersion COMMAND test_chirp1d_dispersion)

# chirp1d gate C: whistler anisotropy-instability growth rate vs kinetic theory.
add_executable(test_chirp1d_growth tests/test_chirp1d_growth.cu)
target_include_directories(test_chirp1d_growth PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_chirp1d_growth PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME chirp1d_growth COMMAND test_chirp1d_growth)

# Darwin GATE 2: Weibel/filamentation instability (full Darwin loop incl. E_T).
add_executable(test_weibel tests/test_weibel.cu)
target_include_directories(test_weibel PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_weibel PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME weibel COMMAND test_weibel)
