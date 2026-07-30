# ArcWarden

A GPU-accelerated **Particle-in-Cell (PIC)** plasma simulator, built with C++20 + CUDA
for a single GPU (developed on RTX 5090 / sm_120). One particle engine — fused
gather–Boris–move–scatter kernels, tiled charge-conserving deposit, amortized spatial
sort — drives **three field engines**:

1. **2D EM (Yee / Esirkepov)** — full-Maxwell FDTD, charge-conserving current deposit,
   Umeda absorbing boundaries, linearized cold-electron fluid, δf and full-f particles.
   Measured **1.55×10¹⁰ particle-steps/s**, **2.59× faster than OSIRIS-CUDA** on the same
   GPU, same physics, matched diagnostics.
2. **2D spectral Darwin** — magnetoinductive (radiation-free, no light-wave CFL), UPIC
   `mpdbeps2` lineage, with UPIC-style trial-push **time-centering** (`darwin_tc`) that
   removes the O(dt) free-mode damping of naive Darwin schemes. 2.2× cheaper
   time-to-solution than tiled Yee on radiation-free problems.
3. **1D gcPIC-style field-aligned chorus code** (`chirp2d`, ny = 1) — hot kinetic
   electrons + linearized cold fluid + analytic background **B₀(s)** (parabolic or dipole
   with the gyro-averaged mirror force folded into the Boris rotation), triggering
   antenna or shot-noise self-seeding, hybrid/damping wave absorbers, checkpoint/resume,
   and a knob-free high-latitude **boundary-refresh thermal bath** (electron resupply at
   the particle's own bounce-flux rate).

The Arc Warden is a hero from DotA who commands plasma and the electromagnetic field.

**Author:** Donglai Ma
**Email:** donglaima96@gmail.com · dma96@atmos.ucla.edu

---

## Key results

Five published results reproduced end-to-end (figures + numbers:
**[docs/MILESTONES.md](docs/MILESTONES.md)**; methods paper draft in `paper/`):

| Result | Reference | Engine | Headline number |
|---|---|---|---|
| Whistler-driven nonlinear trapping (Langmuir / EAW+unipolar / holes+bipolar, sims 1–3) | An et al., PRL 122, 045101 (2019) | Darwin (Yee cross-check) | δB/B₀ saturation 0.105 vs 0.108 (2.7%) across solvers |
| Oblique whistler → Landau beams → electron-acoustic waves ("case 7") | Ma et al., PoP 31, 022304 (2024) | 2D EM | Panel-level match to OSIRIS at **2.59× less wall time** (2080 s vs 5393 s, 156M particles, 207k steps) |
| Rising-tone chorus element with triggered chirp | Tao et al., GRL 44 (2017) | 1D gcPIC (δf) | Chirp 0.25→0.7 Ωe; ∂ω/∂t = 4.2×10⁻⁴ vs theory 5.3×10⁻⁴ Ωe²; phase-space hole tracked following u_R(ω) |
| Two-band chorus with the 0.5 f_ce gap | Li et al., Nat. Commun. 10, 4672 (2019) | Yee 1D (15° oblique) | gap/LB carved 1.07 → **3×10⁻⁴** over 950 gyro-periods, Li's four stages; Darwin reproduces it only after the `darwin_tc` fix — the O(dt) damping diagnosis |
| Three chorus regimes (incoherent / discrete elements / hiss-like), full-f from shot noise | Chen et al., PoP 33, 072105 (2026) | 1D gcPIC (full-f) | All three regimes; Case II riser 0.21→0.53 Ωe at δB/B = 0.030 (paper 0.027); giant run (ppc = 110k, 425M markers) resolves the quiet phase + repetitive element train |

Beyond reproduction:

- **lre scan (Chen Case II at ×20/×10/×5 dipole compression):** self-start threshold
  ∝ lre⁻⁴ (×20 suppressed / ×10 threshold-gated / ×5 refill-gated element complexes,
  period ∝ bounce time); sweep rate = 0.72–0.73× Omura's Eq. (88) at both lre —
  amplitude-controlled and lre-free.
- **Performance:** tiled ρ+J deposit 14.9 Gdep/s (6.4× vs global atomics), fused EM push
  31.5 Gpush/s; flat→tiled→fused-migrate ablation 4.5→15.1→16.1 ×10⁹ p-steps/s.
- **RSM (flagship, in progress):** a reduced spectral model adding one complex oblique
  harmonic F₁(x)e^{ik₁y} to the 1D box (∂y → ik₁ exact) — parallel chirping (m=0) and the
  oblique Landau channel E∥ (m=±1) in one full-f run
  (`docs/RSM_MODEL_DEFINITION.md`).

## Features

- 2D3V (and 1D3V field-aligned), single-GPU; `float` hot path, `double` diagnostics.
- **Deposit:** Esirkepov charge-conserving (Yee) / moment deposits (Darwin); 16×16 tiled
  shared-memory path with amortized physical sort (`tile_sort`), global-atomic fallback.
  At high ppc the tiled path is mandatory (flat atomics measured 52× slower at ppc 110k).
- **Push:** relativistic/non-relativistic Boris, fused with gather + deposit + migration;
  background-B₀ mirror force via effective-field rotation (energy-conserving).
- **Boundaries:** periodic; Umeda field-damping layers; hybrid (field + layer-particle
  transverse damping) absorber; specular particle reflection.
- **Species:** multi-species text decks; bi-Maxwellian, bi-kappa (`kappa_v`), loss-cone
  subtracted (Chen Eq. 1) loaders; mirror-equilibrium (E,μ) loading in B₀(s); δf
  (nonlinear two-weight) and full-f representations.
- **Checkpoint/resume** (`--ckpt=N` / `--resume`): M0 schema, streamed payload, atomic
  rename; RSM state included.
- **Refresh bath** (`[refresh]`): stateless thermal-bath shell at λ_R, one f0 redraw per
  outward crossing (probability-gated, flux-weighted u∥) — the knob-free "infinite
  train" electron resupply (`docs/REFRESH_DESIGN.md`).
- Text input **decks** (`decks/*.ini`) — every experiment is a deck, never a new main().
- Python plotters: spectrograms/STFT, ω–k, k–t, phase-space videos, f(v∥) evolution,
  element-period/sweep-rate analysis.

## Requirements

CUDA 13.x (any recent GPU; developed on sm_120), cuFFT, CMake ≥ 3.18, C++20 compiler.
Python 3 + numpy + scipy + matplotlib for analysis.

## Build

```bash
mkdir build && cd build
cmake ..
make -j
ctest            # validation suite (22 gates)
```

## Usage

```bash
# 2D EM (Yee): oblique whistlers + EAW, Ma PoP 2024 case 7
./build/eaw2d_yee decks/eaw_case7.ini out_eaw

# 2D spectral Darwin: An et al. 2019 sims (unified runner picks the model from the deck)
./build/arcsim decks/an2019_sim3.ini out_an3

# 1D gcPIC chorus: Tao 2017 triggered element (δf) / Chen 2026 full-f from shot noise
./build/chirp2d decks/chirping_1d_tao_trig.ini out_tao
./build/chirp2d decks/chen2026_case2_giant.ini out_chen --fullf --ckpt=50000

# Li 2019 two-band gap (1D oblique Yee) and the RSM oblique-harmonic box
./build/liband_yee decks/li2019_band.ini out_li
./build/rsm_band  decks/rsm_band_li.ini out_rsm
```

Overrides on any runner: `--ppc= --nsteps= --amp= --fullf --ckpt= --resume`.

## Physics / numerics

- CIC shape shared by deposit and gather; leapfrog + Boris.
- Yee: FDTD Maxwell + Esirkepov J (Gauss-law exact to roundoff); binomial J smoothing.
- Darwin: `B_k = iμ₀(k×J)/k²`, transverse E from acceleration-density and momentum-flux
  moments via the resummed transverse Green's function; `darwin_tc` trial-push
  time-centering (dt-independent growth rates — see `docs/DARWIN_UPIC_COMPARISON.md`).
- Cold fluid: linearized electron fluid carrying the whistler branch (half-kick /
  exact-rotation / half-kick), `cold_model = full` for the 3-component oblique response.
- Dipole B₀(s): exact arc-length inversion fit to an even polynomial at deck-finalize;
  mirror force gyro-averaged into the Boris rotation.
- Normalization: ω_pe = 1, m_e = 1, |e| = 1, ε₀ = 1, c set per deck.

See `ARCHITECTURE.md` for the code map; `docs/` for per-study reproduction logs
(`CHEN2026_REPRODUCTION.md`, `TAO2017_REPRODUCTION.md`, `EAW_CASE7_REPRODUCTION.md`,
`GAP_PLAN.md`, `DARWIN_UPIC_COMPARISON.md`, `PROFILE_BASELINE.md`).

## Project structure

```
include/pic/   header-only library (Yee, Darwin, cold fluid, RSM, refresh, checkpoint)
tests/         CTest gates (unit / physics / regression), one executable each
tools/         runners: arcsim, chirp2d, eaw2d_yee, liband_yee, rsm_band, mirror2d, ...
decks/         text input decks (*.ini) — one per experiment
scripts/       Python analysis + plotting
docs/          reproduction logs, design records, MILESTONES.md
paper/         methods-paper draft (ArcWarden vs OSIRIS)
```

## Validation

`ctest` runs 22 gates: FFT/Poisson/deposit/pusher units, cold Langmuir, two-stream,
Weibel, Darwin magnetostatics, checkpoint format, δf growth, RSM ladder (phase load,
cold-plasma oblique dispersion vs Stix ≤0.8%, particle-coupling conservation), and the
refresh-bath null test. Shared-file changes are additionally gated by a statistical
non-regression run pair (`scripts/regress_case2.py`, both deposit paths).

## Acknowledgements

Builds on the component structure of the UCLA/UPIC spectral PIC framework (V. K. Decyk).
Reproduced studies: An et al. 2019; Ma et al. 2024; Tao et al. 2017; Li et al. 2019;
Chen et al. 2026. OSIRIS-CUDA comparison runs use OSIRIS 4.4.4.
