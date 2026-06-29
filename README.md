# ArcWarden

A GPU-accelerated **2D3V spectral Particle-in-Cell (PIC)** plasma simulator, built with
C++20 + CUDA for a single GPU. Supports both an **electrostatic** spectral solver and a
full **spectral Darwin** electromagnetic model (magnetoinductive, no radiation).

The Arc Warden is a hero from DotA who commands plasma and the electromagnetic field.

**Author:** Donglai Ma
**Email:** donglaima96@gmail.com · dma96@atmos.ucla.edu · donglai.ma@kla.com

---

## Overview

ArcWarden simulates kinetic plasma dynamics in 2D configuration space and 3D velocity
space (2D3V), with periodic boundaries. The field solve is **spectral** (FFT-based), which
gives exact derivatives and no grid dispersion. Two field models share one particle engine
and main loop, selected at compile time:

- **Electrostatic** — Poisson solve `E_L = −i k ρ_k/(ε₀k²)`. Validated on cold Langmuir
  oscillations, two-stream and bump-on-tail instabilities, long-run energy/charge conservation.
- **Spectral Darwin (EM)** — longitudinal `E_L`, magnetic `B` from the current, and the
  transverse inductive field `E_T` (UPIC `mpdbeps2` lineage; the radiative `∂E_T/∂t` is
  dropped, so there is no light-wave CFL and the FFT solver remains the engine). Validated on
  a magnetostatic current sheet and the **Weibel instability** growth rate.

It reproduces Simulation 1 of An et al., *"Unified view of nonlinear wave structures
associated with whistler-mode chorus"* (PRL/arXiv:1901.00953) — an external whistler pump
field Landau-traps tail electrons and excites Langmuir waves.

## Features

- 2D3V, periodic, single-GPU; `float` hot path, `double` diagnostics/reductions.
- Compile-time policy config (`SimConfig`): dims, CIC shape, B0, deposit strategy, field model.
- **Deposit:** tiled + chunk-pool-sorted charge/current deposit with bounded shared-memory
  footprint (scales to large 2D grids); selectable global-atomic and shared-tile policies.
- **Push:** magnetized Boris, fused gather/push with a shared field tile.
- **Spectral engine:** cuFFT R2C/C2R, k-space Poisson/Darwin solves, particle-shape form factors.
- External **pump field** + background **B0** driver for prescribed-wave experiments.
- Text input **decks** (`decks/*.ini`) and a multi-species loader — no recompile per setup.
- Python plotters for phase space, ω–k dispersion, k–t spectra, and movies.

## Requirements

- CUDA (developed on CUDA 13.x, RTX 5090 / sm_120; any recent NVIDIA GPU + CUDA toolkit),
  cuFFT, CMake ≥ 3.18, a C++20 compiler. Python 3 + numpy + matplotlib (and ffmpeg) for plots.

## Build

```bash
mkdir build && cd build
cmake ..
make -j
ctest            # run the validation suite (12 tests)
```

## Usage

Run a physics setup from a text deck (electrostatic by default, `--em` for Darwin):

```bash
./build/run_deck decks/bump_on_tail.ini out_dir          # electrostatic
./build/run_deck decks/bump_on_tail.ini out_dir --em     # spectral Darwin
```

Standalone experiments / benchmarks:

```bash
./build/deposit_bench 256 256 64          # deposit + push microbenchmark across grids
./build/whistler_pump 65536 7500 4096 1 6 # An et al. (2019) Simulation 1
python3 scripts/plot_whistler_kt.py       # k-t spectrum, etc.
```

## Physics / numerics

- **Shape:** cloud-in-cell (CIC); one stencil shared by deposit and gather.
- **Time integration:** leapfrog + Boris pusher (magnetized).
- **ES field solve:** spectral Poisson, k=0 and Nyquist zeroed.
- **Darwin field solve:** `B_k = i μ₀ (k×J)/k²`; transverse `E_T` from acceleration-density
  (`dcu`) and momentum-flux (`amu`) moments via the resummed transverse Green's function
  `1/(ε₀c²k² + n₀)` with k-space transverse projection.
- **Normalization:** ω_pe = 1, m_e = 1, |e| = 1, ε₀ = 1.

See `ARCHITECTURE.md` for the code/class map and `gpu_darwin_pic_plan.md` for design rationale.

## Project structure

```
include/pic/   header-only library (templated on the compile-time config)
src/           entry point (smoke-test / device banner)
tests/         CTest unit + physics validation (one executable each)
tools/         standalone experiments, benchmarks, deck runner
decks/         text input decks (*.ini)
scripts/       Python plotters (matplotlib)
sessions/      saved development-session records
```

## Validation

`ctest` runs 12 checks: FFT round-trip, spectral Poisson, particle load, charge
conservation, single-particle Boris, diagnostics, cold Langmuir, two-stream growth rate,
long-run conservation, and the Darwin gates — magnetostatic B-from-J and the Weibel
instability growth rate.

## Acknowledgements

Builds on the component structure of the UCLA/UPIC spectral PIC framework (V. K. Decyk).
The whistler-chorus study reproduced here is An, Li, Bortnik, Decyk, Kletzing & Hospodarsky
(2019).
