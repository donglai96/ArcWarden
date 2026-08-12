# Learning Note — Step 14: Physics Validation (`two_stream` / long-run conservation)

> Goal: understand the two *physics* acceptance tests that certify the code is not
> just numerically self-consistent but actually reproduces kinetic plasma behavior:
> the **two-stream instability** (a growth rate you can compare to an analytic
> dispersion relation, plus phase-space vortices) and **long-run conservation** (the
> code doesn't silently heat or leak charge over thousands of steps). No new physics
> machinery — these exercise the Step-13 loop at its limits.

---

## 0. What Step 14 is, in one paragraph

Steps 1–13 built and unit-tested every component, ending with a linear Langmuir
oscillation. Step 14 pushes into **nonlinear kinetic physics** and **long-time
robustness**. Two tests: `test_two_stream` drives the classic beam instability and
measures its exponential growth rate against theory, then checks for particle
trapping (phase-space vortices); `test_conservation_longrun` runs a thermal plasma for
1000 steps and verifies charge and energy don't drift. The only new code is a
`two_stream` load option in `RunParams` + the init kernel.

---

## 1. The two-stream instability — the physics

Two interpenetrating electron beams streaming through each other (velocities `±v0`)
are **unstable**: any small density ripple grows exponentially, feeding on the beams'
relative kinetic energy. It's the textbook kinetic instability and a rite of passage
for any PIC code, because its growth rate has a clean analytic prediction.

For two equal cold beams (each density `n0/2`, total `n0`), the dielectric function is

```
1 = (ω_pe²/2) [ 1/(ω - k v0)²  +  1/(ω + k v0)² ]
```

Solving for complex `ω` (instability ⇔ `ω² < 0` ⇔ growth) gives instability for
`k v0 < ω_pe`, with the **maximum growth rate**

```
γ_max = ω_pe / (2√2) ≈ 0.3536 ω_pe   at   (k v0)² = (3/8) ω_pe².
```

(Derivation: with `u=ω²`, `b=(kv0)²`, `ω_pe=1`, the dispersion becomes
`(u-b)² = u+b`, so `u = [(2b+1) ± √(8b+1)]/2`; the `−` root goes negative for
`0<b<1`, and maximizing `−u₋` gives `b=3/8`, `γ=√(1/8)=1/(2√2)`.)

---

## 2. Designing the test around the analytic answer

To compare against `γ_max`, we make the **fundamental box mode** `k1 = 2π/Lx` *be* the
most unstable mode: set `k1 v0 = √(3/8) ω_pe`. With `v0=1`, `ω_pe=1`, that fixes
`Lx = 2π/√(3/8)`. Then only the fundamental is unstable (the next harmonic `k2=2k1`
has `(k2 v0)² = 1.5 > 1`, stable), so we get clean single-mode exponential growth.

The two beams come from the new load option: in the init kernel, when `two_stream` is
set, the within-cell particle index parity picks the beam sign,
`drift = (s&1) ? -vd : +vd`. The two beams are **co-located and balanced** per cell
(32 + 32 of `ppc=64`), preserving the quiet start. A tiny single-k seed
(`perturb_amp=0.005`) starts the mode in the linear regime.

**Measuring the rate.** The field energy `EE ∝ E²` grows like `exp(2γt)`, so `ln(EE)`
is a straight line during the linear phase. The test finds the saturation point
(`argmax EE`), least-squares-fits the slope of `ln(EE)` over the growing window, and
reports `γ = slope/2`. Result: **γ = 0.3631 vs analytic 0.3536 — 2.7% off**, well
within PIC + finite-`dt` + fit error.

---

## 3. Phase-space vortices (the nonlinear signature)

Exponential growth can't continue forever: when the field is strong enough to
**trap** particles in its potential wells, the instability *saturates* and the trapped
populations roll up into **vortices** (BGK holes) in `(x, vx)` phase space — the
hallmark image of the saturated two-stream instability.

Quantifying a vortex automatically is awkward, so the test uses a robust proxy: the
two cold beams start with *every* particle at `vx = ±v0`, so the fraction with
`|vx| < v0/2` is exactly **0**. Trapping drags particles off the beams into the gap,
so that fraction grows. Measured: **0.345** at saturation — a third of the particles
have been pulled into the trapping region, the unambiguous signature of vortex
formation. The test also dumps `(x, vx)` to `two_stream_phase.csv` so the vortices can
be seen directly in a scatter plot.

---

## 4. Long-run conservation — the robustness test

A PIC scheme can be subtly wrong in a way no single short test catches: slow
**numerical heating** (energy creeping up because the grid under-resolves the Debye
length) or charge leaking. `test_conservation_longrun` guards against both with a
warm, **Debye-resolved** plasma (`vth=1` ⇒ `λ_D = vth/ω_pe = 1`, and `dx ≈ 0.098 ≪
λ_D`) run for 1000 steps:

- **Charge** must be exactly `N·q·weight` and constant. Measured `−39.4784` with
  relative variation `3.9e-8` — flat to float roundoff (deposition conserves charge
  identically and particle count is fixed).
- **Total energy** must stay bounded. Drift `(max−min)/mean < 0.05%` over `t=50` —
  the symplectic leapfrog+Boris with a resolved Debye length barely heats.

Why Debye resolution matters: when `dx > λ_D`, the grid can't represent the shortest
shielding scale and aliasing pumps energy into the particles (the classic PIC grid-
heating instability). Keeping `dx ≪ λ_D` is the standard cure, and this test is what
tells you you've done it.

---

## 5. How we verified Step 14

- **Two-stream:** `γ = 0.3631` vs `0.3536` (2.7%); trapped fraction `0.345` (from 0);
  phase space dumped. ✓
- **Conservation:** charge `−39.4784`, rel-variation `3.9e-8`; energy drift `<0.05%`
  over 1000 steps. ✓
- CTest 10/10; `compute-sanitizer` clean (including the two-stream load path).

---

## 6. Concepts you learned in Step 14 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Two-stream instability | counter-streaming beams grow a mode | benchmark with an analytic growth rate |
| Dispersion relation | `1 = (ω_pe²/2)[1/(ω∓kv0)²]` | predicts `γ_max = ω_pe/(2√2)` |
| Most-unstable mode | `(k v0)² = 3/8 ω_pe²` | size box so fundamental = that mode |
| Growth from `ln(EE)` | `EE ∝ e^{2γt}` | slope/2 of log field energy = γ |
| Particle trapping | field wells capture particles | saturation; phase-space vortices |
| Trapped fraction proxy | `|vx|<v0/2` grows from 0 | quantitative vortex signature |
| Numerical (grid) heating | energy creep when `dx>λ_D` | long-run test catches it |
| Debye resolution | `dx ≪ λ_D = vth/ω_pe` | keeps the scheme non-heating |
| Charge invariant | `N·q·weight`, constant | strongest conservation check |

---

## 7. One-sentence summary

Step 14 validates the *physics*: the two-stream instability grows at the measured
`γ = 0.3631` against the analytic `ω_pe/(2√2)=0.3536` (2.7%) and saturates into
phase-space vortices (trapped fraction 0→0.345), while a Debye-resolved thermal plasma
conserves charge to 4×10⁻⁸ and total energy to <0.05% over 1000 steps — proving the
code reproduces real kinetic behavior and stays robust over long runs.
