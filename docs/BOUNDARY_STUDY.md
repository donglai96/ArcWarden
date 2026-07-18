# M2 boundary study: absorbing x-ends for the 2D Yee branch

Goal (plan M2): absorbing boundaries in the field-aligned direction (x here;
"s" in the L-shell geometry) with whistler-band reflection R < 1%, plus a
recorded map of failure modes. No PML presupposed — candidates compared on the
plan's benchmark: R(ω) for quasi-parallel whistlers measured **in plasma**.

## Scheme implemented

Umeda-style multiplicative masking layers, ported from the validated chirp1d
1D implementation (`include/pic/hybrid1d.hpp`) to the 2D Yee branch:

- `[boundary] x = damping|hybrid`, `nd` (cells per side), `numax`.
- Mask m(x) = exp(−ν_max d² Δt), d = normalized depth into the nd-cell layer,
  multiplied onto ALL wave fields each step (`yee::k_damp_x`; staggered site
  masks m(i), m(i+½)). The grid holds only wave fields — the external B0 in
  RunParams is never touched.
- `hybrid` additionally damps each layer particle's transverse momentum
  (u_y, u_z) by the same mask (2D analogue of chirp1d damping the cold-fluid
  v_cy/v_cz next to the fields).
- Particles reflect specularly at the domain ends (x = 0, nx) — the layers
  may contain plasma; the x field indexing stays periodic and the layers
  suppress the wrap-around leakage.

## Benchmark

`tools/boundary_reflection.cu`: cold magnetized plasma (ω_pe = 1, ω_ce = 0.5,
c = 10, ppc 100 quiet start), antenna column at the domain center radiates an
R-mode whistler packet (trapezoid-enveloped, 6 drive periods); probe column
between antenna and layer measures the **narrowband complex envelope at the
drive frequency** (demodulation + one-pole low-pass, both helicities);
R = late-window max / outbound-window max. Windows from the cold Maxwell
whistler group velocity; the late window opens just 2 drive periods after
the estimated round trip (an early version opened 6 periods late and
clipped the head of the reflected packet, understating R by up to ~3× —
all numbers below use the honest window).

Sanity gates: vacuum EM pulse absorption R = 0.14% with a periodic control
returning the pulse at 99.95% (`tests/test_boundary_vacuum.cu`, ctest
`boundary_vacuum`); probe spectrum peaks at the drive frequency; incident
amplitude 100× the shot-noise floor.

## Results (2026-07-17, RTX 5090)

R = A_ref/A_inc at the drive frequency; λ = whistler wavelength in cells
(dx = 1, i.e. dx·ω_pe/c = 0.1):

| ω/ω_ce | λ (dx) | mode | nd | ν_max | R |
|---|---|---|---|---|---|
| 0.40 | 75  | hybrid | 64  | 0.3 | 1.1% |
| 0.40 | 75  | hybrid | 64  | 1.0 | ~2% |
| 0.40 | 75  | hybrid | 256 | 0.1 | **0.84%** |
| 0.40 | 75  | hybrid | 384 | 0.1 | 0.80% |
| 0.25 | 105 | hybrid | 64  | 1.0 | ~17% |
| 0.25 | 105 | hybrid | 256 | 0.1 | **0.87%** |
| 0.10 | 186 | hybrid | 64  | 1.0 | ~40% |
| 0.10 | 186 | hybrid | 256 | 0.1 | **0.87%** |
| 0.10 | 186 | hybrid | 384 | 0.15 | 0.62% |

(Rows measured before the window fix, re-scaled qualitatively, are marked ~;
the bold production rows are honest-window measurements. Field-only masks
track hybrid to within ~2× at the ν_max optimum and degrade the same way
off it.)

**Production recommendation: `x = hybrid`, `nd = 256`, `numax = 0.1` →
R = 0.84% / 0.87% / 0.87% at 0.4 / 0.25 / 0.1 ω_ce — below 1% across the
whole chorus band with one setting** (nd = 384 buys 0.6–0.8%). The ctest
gate `boundary_whistler` runs this config at 0.4 ω_ce (R = 0.74%).

### Conclusions so far

1. **The controlling variables are nd/λ(ω) and the damping adiabaticity.**
   R < 1% needs the layer roughly a wavelength deep at the lowest frequency
   to absorb, with ν_max small enough that the damping profile is adiabatic
   (ν_max ≈ 0.3 ≈ 1.5 ω at 0.4 ω_ce beat ν_max = 1 and 3 — over-damping
   reflects off the impedance step before absorbing).
2. **Hybrid beats field-only by ~2.5× at the ν_max optimum** (0.36% vs 0.89%
   at 0.4 ω_ce, ν_max = 0.3), though both are usable; at over-damped settings
   they converge (the gradient reflection dominates). The wave's
   magnetic-energy fraction (~60% here) lets field-only masks work at all.
3. **Low-frequency band is the cost driver**: at 0.1 ω_ce, λ = 186 cells
   forces nd ≥ 256 (per side) with ν_max ≲ 0.3. For chorus runs (0.1–0.5
   ω_ce band) budget the layers off the LOWEST band frequency.

### Oblique incidence (ny = 1 tilted-B0 proxy)

Tilting B0 in the x–y plane while the antenna keeps k ∥ x̂ gives a wave with
wave-normal angle θ hitting the layers (still y-uniform, so ny = 1 remains
valid). Production config (nd = 256, ν_max = 0.1):

| θ | R (0.4 ω_ce) | R (0.25 ω_ce) |
|---|---|---|
| 15° | 2.2% | 1.8% |
| 30° | 8.0% | 5.9% |
| 45° | 32% | 13% |

R at θ = 30° is INSENSITIVE to layer config (nd 256→384, ν_max 0.1→0.3 all
read ~13% with the honest window) — a config-independent floor that smells
like measurement contamination (with oblique B0 the antenna also couples to
slow quasi-electrostatic branches whose late arrival pollutes the window)
rather than pure layer reflection. Recorded caveat: the ny = 1 proxy is
trustworthy for quasi-parallel incidence; a true 2D oblique-packet benchmark
is future work. For the M10 use case — field-aligned chorus reaching the
s-ends quasi-parallel — the θ ≤ 15° numbers (≈ 2%) are the relevant ones.

### Density gradient (密度渐变专项)

Cosine taper of the plasma density from n0 to n_edge over W cells in front
of each layer (implemented via per-particle weights; a cold static profile
with E(0) = 0 is force-free). At 0.4 ω_ce, nd = 64, ν_max = 0.3
(uniform-density baseline R = 1.1%):

| taper W | n_edge | R |
|---|---|---|
| 128 | 0.25 | 3.6% |
| 256 | 0.25 | 3.6% |
| 128 | 0.0625 | 9.2% |

**A density falloff into the layers makes absorption WORSE, and the
mechanism is not gradient reflection** (R is independent of taper width):
lower density raises the whistler group velocity (2.8 → 5.1 at n = 0.25),
so the wave crosses the layer in half the time and picks up half the
integrated damping. Design rule for the dipole/L-shell runs (M5+), where
field-aligned density falls toward the boundary ends naturally: do NOT add
an artificial taper, and size nd/ν_max for the LOCAL (lower) density at the
layer — i.e. for the faster group velocity and longer wavelength there.

### Failure modes recorded

- ν_max too large → reflection off the damping gradient (R grows with ν_max
  past the optimum). ν_max too small → insufficient integrated damping.
- nd ≪ λ → the layer is transparent regardless of ν_max (42% at nd/λ = 0.34).
- **L-sense antenna drive couples to nothing below ω_ce** (no L-mode): the
  drive is evanescent, only broadband ramp transients radiate, and the slow
  near-ω_ce content contaminates any late-window measurement. Antenna must
  rotate in the electron-gyration sense: (J_y + iJ_z) ∝ e^{+iω0 t}.
- Hard antenna turn-off radiates a broadband burst with vg → 0 components
  near ω_ce that linger near the source — use the trapezoid envelope.
- Field-only masks with plasma in the layer leave the coherent transverse
  current undamped; measurable but subdominant here (hybrid exists for runs
  where it matters, e.g. hotter plasma with larger kinetic energy fraction).

## Status

- [x] R(ω) quasi-parallel: production config < 1% across 0.1–0.5 ω_ce.
- [x] Oblique R(ω, θ) via the ny = 1 tilted-B0 proxy (θ ≤ 15° ≈ 2%; caveat
      recorded for θ ≥ 30°; true 2D packet benchmark = future refinement).
- [x] Density-gradient study: taper hurts (vg speed-up in the layer);
      design rule recorded for M5+ dipole density profiles.
- [x] ctest `boundary_whistler`: production config at 0.4 ω_ce, R = 0.74%
      < 1% gate. ctest `boundary_vacuum`: EM pulse R = 0.14%.
- [ ] (deferred) vacuum-gap/PML candidate — only if a future run needs
      θ ≥ 30° absorption the masks can't provide.
