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
whistler group velocity.

Sanity gates: vacuum EM pulse absorption R = 0.14% with a periodic control
returning the pulse at 99.95% (`tests/test_boundary_vacuum.cu`, ctest
`boundary_vacuum`); probe spectrum peaks at the drive frequency; incident
amplitude 100× the shot-noise floor.

## Results (2026-07-17, RTX 5090)

R = A_ref/A_inc at the drive frequency; λ = whistler wavelength in cells
(dx = 1, i.e. dx·ω_pe/c = 0.1):

| ω/ω_ce | λ (dx) | mode | nd | ν_max | R |
|---|---|---|---|---|---|
| 0.40 | 75  | field  | 64  | 0.3 | 0.89% |
| 0.40 | 75  | field  | 64  | 1.0 | 1.3% |
| 0.40 | 75  | hybrid | 64  | 0.3 | **0.36%** |
| 0.40 | 75  | hybrid | 64  | 1.0 | 1.6% |
| 0.40 | 75  | hybrid | 64  | 3.0 | 2.3% |
| 0.40 | 75  | hybrid | 128 | 1.0 | 1.1% |
| 0.40 | 75  | hybrid | 256 | 0.1 | 0.85% |
| 0.25 | 105 | field  | 64  | 1.0 | 20% |
| 0.25 | 105 | hybrid | 64  | 1.0 | 17% |
| 0.25 | 105 | hybrid | 128 | 0.3 | 0.88% |
| 0.25 | 105 | hybrid | 256 | 0.3 | 0.44% |
| 0.25 | 105 | hybrid | 256 | 0.1 | 0.68% |
| 0.10 | 186 | field  | 64  | 1.0 | 42% |
| 0.10 | 186 | hybrid | 64  | 1.0 | 39% |
| 0.10 | 186 | hybrid | 256 | 0.1 | 0.75% |
| 0.10 | 186 | hybrid | 256 | 0.3 | 3.2% |
| 0.10 | 186 | hybrid | 256 | 1.0 | 12% |
| 0.10 | 186 | hybrid | 384 | 0.15 | 0.65% |

**Production recommendation: `x = hybrid`, `nd = 256`, `numax = 0.1` → R =
0.75% / 0.68% / 0.85% at 0.1 / 0.25 / 0.4 ω_ce — R < 1% across the whole
chorus band with ONE setting.** (If the low band is not present, nd = 64,
ν_max = 0.3 suffices at 2.5× less grid: R = 0.36% at 0.4 ω_ce.)

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

## TODO

- [ ] Oblique incidence R(ω, θ) (2D: seed oblique packets or tilt B0).
- [ ] Density-gradient variant (plan: 密度渐变反射专项).
- [ ] Vacuum-gap + PML candidate for comparison if masking can't reach the
      target at acceptable nd (currently it can).
- [x] ctest `boundary_whistler` (tests/test_boundary_whistler.cu): 0.4 ω_ce,
      hybrid nd=64 ν_max=0.3, gate R < 1% (measured 0.79%).
