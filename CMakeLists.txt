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

# provenance: configure-time git hash embedded in checkpoints (pic2d)
execute_process(COMMAND git rev-parse HEAD
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE AW_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT AW_GIT_HASH)
  set(AW_GIT_HASH "unknown")
endif()
add_compile_definitions(AW_GIT_HASH="${AW_GIT_HASH}")

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

# An et al. 2019 on the full-Maxwell branch (M1 cross-validation vs Darwin)
add_executable(an2019_yee tools/an2019_yee.cu)
target_include_directories(an2019_yee PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(an2019_yee PRIVATE CUDA::cudart CUDA::cufft)

# Ma et al. PoP 2024 case 7 — 2D oblique whistler from anisotropy (M1, true 2D)
add_executable(eaw2d_yee tools/eaw2d_yee.cu)
target_include_directories(eaw2d_yee PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(eaw2d_yee PRIVATE CUDA::cudart)

# M2: whistler reflection coefficient of the x-end damping layers
add_executable(boundary_reflection tools/boundary_reflection.cu)
target_include_directories(boundary_reflection PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(boundary_reflection PRIVATE CUDA::cudart)

add_executable(chirp2d tools/chirp2d.cu)
target_include_directories(chirp2d PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(liband_yee tools/liband_yee.cu)
target_include_directories(liband_yee PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(mirror2d tools/mirror2d.cu)
target_include_directories(mirror2d PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(rsm_band tools/rsm_band.cu)
target_include_directories(rsm_band PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rsm_band PRIVATE CUDA::cudart CUDA::cufft)

add_executable(liband_darwin tools/liband_darwin.cu)
target_include_directories(liband_darwin PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(liband_darwin PRIVATE CUDA::cudart CUDA::cufft)
target_link_libraries(chirp2d PRIVATE CUDA::cudart)

add_executable(weight_precision tools/weight_precision.cu)
target_include_directories(weight_precision PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(weight_precision PRIVATE CUDA::cudart)

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

# M1 gates: Yee vacuum dispersion / Esirkepov continuity / dual-path Langmuir.
foreach(t yee_vacuum esirkepov_continuity yee_langmuir yee_tiled yee_vs_darwin_lowfreq boundary_vacuum boundary_whistler deltaf_equilibrium deltaf_growth deltaf_consistency deltaf_landau yee_mirror_bounce deltaf_mirror_hold cold_fluid_dispersion dipole_profile mirror2d cold_fluid_oblique)
  add_executable(test_${t} tests/test_${t}.cu)
  target_include_directories(test_${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(test_${t} PRIVATE CUDA::cudart CUDA::cufft)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()

# L-shell M0: checkpoint format round-trip.
add_executable(test_checkpoint_format tests/test_checkpoint_format.cu)
target_include_directories(test_checkpoint_format PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_checkpoint_format PRIVATE CUDA::cudart)
add_test(NAME checkpoint_format COMMAND test_checkpoint_format)

# L-shell M0 gate: magnetized Boris (ExB drift, drift-frame mu, oblique B0).
add_executable(test_magnetized_boris tests/test_magnetized_boris.cu)
target_include_directories(test_magnetized_boris PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_magnetized_boris PRIVATE CUDA::cudart CUDA::cufft)
add_test(NAME magnetized_boris COMMAND test_magnetized_boris)

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

# RSM gate V1 (docs/RSM_MODEL_DEFINITION.md §6.1 R1): the phase load must be
# N^{-1/2} shot noise, not the coherent y = 0.5 loader artifact.
add_executable(test_rsm_phaseload tests/test_rsm_phaseload.cu)
target_include_directories(test_rsm_phaseload PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_rsm_phaseload PRIVATE CUDA::cudart)
add_test(NAME rsm_phaseload COMMAND test_rsm_phaseload)

add_executable(test_shell_split tests/test_shell_split.cu)
target_include_directories(test_shell_split PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_shell_split PRIVATE CUDA::cudart)
add_test(NAME shell_split COMMAND test_shell_split)

# RSM gate V2: spectral m=1 cold oblique dispersion/polarization/E_par vs
# Stix (same solver as test_cold_fluid_oblique = the R5 cross-check).
add_executable(test_rsm_cold_dispersion tests/test_rsm_cold_dispersion.cu)
target_include_directories(test_rsm_cold_dispersion PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_rsm_cold_dispersion PRIVATE CUDA::cudart)
add_test(NAME rsm_cold_dispersion COMMAND test_rsm_cold_dispersion)

# RSM gate V3: particle-coupling conservation (kick factor-2, modal
# continuity, divB1, energy ledger through the anisotropy instability).
add_executable(test_rsm_conservation tests/test_rsm_conservation.cu)
target_include_directories(test_rsm_conservation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_rsm_conservation PRIVATE CUDA::cudart)
add_test(NAME rsm_conservation COMMAND test_rsm_conservation)

# REFRESH Gates 1+2 (docs/REFRESH_DESIGN.md): thermal-bath null test (bath
# invisible at f = f0) + lambda_R geometric insensitivity.
add_executable(test_refresh_null tests/test_refresh_null.cu)
target_include_directories(test_refresh_null PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_refresh_null PRIVATE CUDA::cudart)
add_test(NAME refresh_null COMMAND test_refresh_null)

# ---- ctest labels (L-shell plan M0): unit / physics / regression ----
set_tests_properties(smoke_device fft_roundtrip poisson_sine particle_load
                     charge_conservation single_particle magnetized_boris
                     checkpoint_format diagnostics rsm_phaseload
                     PROPERTIES LABELS "unit")
set_tests_properties(langmuir two_stream weibel darwin_magnetostatic
                     chirp1d_mirror chirp1d_dispersion chirp1d_growth
                     yee_vacuum esirkepov_continuity yee_langmuir
                     yee_vs_darwin_lowfreq boundary_vacuum boundary_whistler
                     deltaf_equilibrium deltaf_growth deltaf_consistency deltaf_landau
                     PROPERTIES LABELS "physics")
set_tests_properties(conservation_longrun PROPERTIES LABELS "regression")

# ---- 2d-reborn (PLAN_2D_REBORN): the reborn 2D instrument -------------------
# P0 scaffold: linedipole background V0 gate + deck pre-flight runner.
# Everything additive; the legacy stack above is untouched by construction.
add_executable(warden2d tools/warden2d.cu)
target_include_directories(warden2d PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(warden2d PRIVATE CUDA::cudart)

add_executable(test_dipole2d_orbit tests/test_dipole2d_orbit.cu)
target_include_directories(test_dipole2d_orbit PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_dipole2d_orbit PRIVATE CUDA::cudart)
add_test(NAME dipole2d_orbit COMMAND test_dipole2d_orbit)
set_tests_properties(dipole2d_orbit PROPERTIES LABELS "unit")

add_executable(test_cold2d_dispersion tests/test_cold2d_dispersion.cu)
target_include_directories(test_cold2d_dispersion PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_cold2d_dispersion PRIVATE CUDA::cudart)
add_test(NAME cold2d_dispersion COMMAND test_cold2d_dispersion)
set_tests_properties(cold2d_dispersion PROPERTIES LABELS "physics")

add_executable(test_absorber2d tests/test_absorber2d.cu)
target_include_directories(test_absorber2d PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_absorber2d PRIVATE CUDA::cudart)
add_test(NAME absorber2d COMMAND test_absorber2d)
set_tests_properties(absorber2d PROPERTIES LABELS "physics")

add_executable(test_deltaf2d_hold tests/test_deltaf2d_hold.cu)
target_include_directories(test_deltaf2d_hold PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_deltaf2d_hold PRIVATE CUDA::cudart)
add_test(NAME deltaf2d_hold COMMAND test_deltaf2d_hold)
set_tests_properties(deltaf2d_hold PROPERTIES LABELS "physics")

add_test(NAME deltaf2d_hold_aniso COMMAND test_deltaf2d_hold aniso)
set_tests_properties(deltaf2d_hold_aniso PROPERTIES LABELS "physics")

add_executable(test_li2019_reborn tests/test_li2019_reborn.cu)
target_include_directories(test_li2019_reborn PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_li2019_reborn PRIVATE CUDA::cudart)
add_test(NAME li2019_reborn COMMAND test_li2019_reborn)
set_tests_properties(li2019_reborn PROPERTIES LABELS "regression")

add_executable(test_ckpt2d tests/test_ckpt2d.cu)
target_include_directories(test_ckpt2d PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_ckpt2d PRIVATE CUDA::cudart)
add_test(NAME ckpt2d COMMAND test_ckpt2d)
set_tests_properties(ckpt2d PROPERTIES LABELS "unit")

add_executable(bench2d tools/bench2d.cu)
target_include_directories(bench2d PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench2d PRIVATE CUDA::cudart)
