# ArcWarden — Code & Class Structure

A 2D3V electrostatic spectral GPU PIC (particle-in-cell) plasma simulator.
C++20 + CUDA, single GPU. Namespace `arc`. Headers in `include/pic/`,
device/host translation units in `src/`.

This document maps the code: the directory layout, the core abstractions, and
how data flows through one PIC step. For the build/step history see
`es_pic_step1_coding_plan.md`; for the design rationale see
`gpu_darwin_pic_plan.md`.

---

## 1. Directory layout

```
include/pic/      header-only library (most logic lives here, templated on Cfg)
src/              .cu/.cpp translation units (entry point + a few definitions)
tests/            CTest unit/physics tests (one executable each)
tools/            standalone experiments & visualizers (movies, deck runner)
decks/            text input decks (*.ini) — physics setups as data
scripts/          Python plotters (matplotlib) for the dumped CSV frames
learning_step*.md per-step teaching notes
```

## 2. Normalization contract (the glue)

All physics uses ω_pe = 1, m_e = 1, ε₀ = 1 (set in `RunParams`). A macro-particle
`i` represents `w[i]` physical particles. Charge density deposited per particle is
`q·w[i]/(dx·dy)`; kinetic energy is `Σ w[i]·½(ux²+uy²+uz²)`. q/m is global
(`RunParams::qm`, electrons = −1); per-particle q/m is a future (ions) extension.

## 3. The PIC main loop

`Simulation::step()` runs the standard cycle (`include/pic/simulation.hpp`):

```
zero rho
deposit  (particles → rho)        Depositor
solve    (rho → φ → E in k-space)  ElectrostaticSpectralSolver + SpectralEngine
push     (gather E → Boris update) Pusher
migrate  (periodic wrap + recell)  Particles
diagnose (energy/charge, gated)    Diagnostics
```

`init()` loads particles, computes the t=0 field, and rolls velocities back a
half step for the leapfrog start.

---

## 4. Class reference (by layer)

### 4.1 GPU infrastructure — `cuda_utils.hpp`, `device_array.hpp`
| Type | Role |
|---|---|
| `CudaError` | exception carrying `file:line`; thrown by `CUDA_CHECK`/`CUFFT_CHECK` |
| `CudaStream` | RAII stream (move-only) |
| `CudaEvent` | RAII event + timing |
| `DeviceView<T>` | POD `{ptr, size}` view; passed to kernels **by value**, `__device__ operator[]` |
| `DeviceArray<T>` | owning device buffer (move-only): `cudaMalloc/Free`, `zero()`, `view()` |

**The Views pattern:** every owning container (`Particles`, `Sources`, `Fields`)
has a `*Views` POD counterpart of `DeviceView`s. Owners live on the host; their
`.views()` are copied into kernels by value. This keeps kernels free of host
types and ownership.

### 4.2 Configuration — `config.hpp`, `species.hpp`
| Type | Role |
|---|---|
| `SimConfig<Dim,VelDim,Shape,Real,HasB0,DepositPolicy>` (alias `Cfg`) | compile-time config: dims, shape order, precision (`float`), B0 capability, deposit policy. Drives template specialization. |
| `RunParams` | runtime scalars: `dt`, `qm`, `eps0`, `wpe`, `B0[3]`, `ppc`, `nsteps`, `dump_every`, `weight`, normalization, plus legacy single-species load flags (`two_stream`, `bump_on_tail`, `beam_*`, `perturb_*`, `noisy_load`, `rng_seed`). Passed to kernels by value. |
| `ShapeOrder`, `NormMode` | enums (CIC shape; ω_pe=1 normalization) |
| `AtomicGlobalDeposit` / `SharedTileDeposit` / `CellOwnedDeposit` | deposit policy tags (only the first is implemented) |
| `Species` | **host** description of one population: `name, density, ppc, uth[3], ufl[3]` |
| `SpeciesList` | `std::vector<Species>` — a full plasma is just a list |

### 4.3 Geometry — `grid.hpp`
| Type | Role |
|---|---|
| `Grid` | real-space mesh: `nx,ny,dx,dy,Lx,Ly`, `idx()`/`idx_periodic()`. Single source of geometry truth. |
| `KGrid` | Fourier mesh for the R2C layout: `nkx=nx/2+1`, `kx/ky/k2` |
| `CicStencil` | 4-cell CIC weights for a position; shared by deposit and gather (`cic_stencil()`) |

### 4.4 Spectral engine — `fft.hpp`, `spectral.hpp`, `solver_es.hpp`
| Type | Role |
|---|---|
| `CufftPlan2D` | RAII cuFFT R2C/C2R plans for `(ny,nx)` |
| `SpectralWorkspace` | k-space scratch: `rho_k, phi_k, Ex_k, Ey_k` |
| `SpectralFormFactor` | `s(k)`, `g(k)=1/(ε₀k²)` shape/Green's factors |
| `SpectralEngine` | owns plan + workspace + KGrid; wraps forward/inverse + normalization |
| `ElectrostaticSpectralSolver` | `solve(sources,fields,…)`: rho → φ_k=rho_k/(ε₀k²) → E_k=−ik φ_k → E (k=0 & Nyquist zeroed) |

### 4.5 Sources & fields — `sources.hpp`, `fields.hpp`
| Type | Role |
|---|---|
| `Sources` / `SourceViews` | charge density `rho` (+ reserved Darwin currents) |
| `Fields` / `FieldViews` | `Ex, Ey` (+ reserved Darwin/EM fields) |

### 4.6 Particles — `particles.hpp`
| Type | Role |
|---|---|
| `Particles` | SoA owner: `x,y` (cell units), `ux,uy,uz`, **`w` (per-particle weight)**, `cell`, `n`. `allocate*/views/initialize/migrate`. |
| `ParticleViews` | POD device view of the above |
| `SpeciesInit` (detail) | POD per-species slice descriptor for the init kernel |
| `particle_init_kernel` (detail) | legacy single-species load (uniform weight); honors `two_stream`/`bump_on_tail`/`perturb`/`noisy_load` |
| `species_init_kernel` (detail) | multi-species load: fills one species' slice with its drift/width/weight |
| `radical_inverse`, `rng_uniform`, `hash_u32` (detail) | quiet (van der Corput) vs noisy (hashed RNG) sampling primitives |

Two load paths, selected by which `initialize` overload `Simulation` calls:
- `initialize(rp, grid, stream)` — single Maxwellian / two-stream / bump-on-tail
  via `RunParams` flags (used by the unit tests and the movie tools).
- `initialize(SpeciesList, grid, rp, stream)` — general multi-species (used by
  decks). Each species → a contiguous slice loaded by `species_init_kernel`.

### 4.7 Operators — `depositor.hpp`, `pusher.hpp`
| Type | Role |
|---|---|
| `Depositor<Cfg>` | `charge()`: scatter `q·w[i]/area` into `rho` via CIC (atomic-global policy). Reads **per-particle** weight. |
| `Pusher<Cfg>` | `boris()` (gather E → Boris velocity+position update) and `half_step_back()` (leapfrog −dt/2 init). q/m global. |

### 4.8 Diagnostics — `diagnostics.hpp`
| Type | Role |
|---|---|
| `DiagSample` | one record: `step,time,ke,ee,total,charge,max_e` |
| `Diagnostics` | `compute()` (per-particle-weighted KE, field energy, total charge) and `maybe_compute()` (gated by `RunParams::dump_every`; `<=0` ⇒ every step). Optional CSV log. |

### 4.9 Orchestration — `simulation.hpp`
| Type | Role |
|---|---|
| `Simulation<Cfg>` | owns Grid, Particles, Sources, Fields, SpectralEngine, solver, Diagnostics, stream. Two constructors: single-species (`RunParams` path) and multi-species (`SpeciesList` path). `init()/step()/run()`. |

### 4.10 Input deck — `deck.hpp`
| Type | Role |
|---|---|
| `Deck` | parsed setup: `nx,ny,Lx,Ly`, `RunParams`, `SpeciesList`, `dump_every`, `outdir` |
| `load_deck(path)` | INI parser: `[grid]`/`[time]`/`[plasma]`/`[species <name>]`, `#`/`!` comments, `uth/ufl` as 3-vectors. Validates grid + ≥1 species. |

---

## 5. Ownership & data-flow summary

```
HOST (owns)                         DEVICE (views passed by value)
  Particles  ── .views() ─────────►  ParticleViews  ─► kernels
  Sources    ── .views() ─────────►  SourceViews
  Fields     ── .views() ─────────►  FieldViews
  Grid/RunParams (POD)  ──────────►  copied by value into every kernel
  SpeciesList (host only) ─► loop launches species_init_kernel per species
```

Precision is `float` end-to-end on the device (particles, rho/E, FFT);
`RunParams` scalars are `double` host-side config. See the precision note in
`config.hpp` (`Cfg` is templated on `Real`, but storage/FFT sites currently
hardcode `float`).

## 6. Tools, decks, scripts

| Path | What it does |
|---|---|
| `src/main.cpp` | `arcwarden` smoke binary (device report) |
| `tools/run_deck.cu` | `./run_deck <deck> [outdir]` — run any deck, dump phase frames + `energy.csv` |
| `tools/two_stream_movie.cu` | two-stream phase-space movie + NVML GPU report |
| `tools/bump_on_tail_movie.cu` | bump-on-tail movie + NVML report (legacy single-species path) |
| `decks/bump_on_tail.ini`, `decks/two_stream.ini` | example setups (ppc ∝ density ⇒ equal weight) |
| `scripts/plot_phase_movie.py` | two-stream phase-space images / begin-vs-end / mp4 |
| `scripts/plot_bump_on_tail.py` | bump-on-tail phase space + f(v) bump→plateau panel / mp4 |

## 7. Extension points (designed-for, not yet built)

- **Magnetized ES**: `B0≠0` Boris rotation path + backward-half-Boris init already
  exist (`Pusher`, `Cfg::has_b0`); needs setup + validation.
- **Darwin / EM**: reserved current/field members in `Sources`/`Fields`; spectral
  workspace and form factor are the insertion points for `J`, transverse
  projection, and spectral correction.
- **Multiple ion species / per-particle q/m**: add a per-particle `qm` array
  alongside `w` and thread it through `Pusher` (deposit already per-particle).
- **fp64**: generalize the `float`-hardcoded storage/FFT sites to `Cfg::real`.
