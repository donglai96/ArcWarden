# Delta-f weight accumulator precision study (M3 close-out)

**Question.** The delta-f weight update `wd += dt·qm·(1−wd)·S` runs every step
in FP32. Over production-length runs (~1e5 steps, M10 scale) does FP32
accumulation roundoff degrade the weights — and would a compensated (Kahan) or
FP64 accumulator buy anything?

**Answer: no. Plain FP32 stays the production default (`df_wprec = 0`).**
The measured FP32-vs-FP64 divergence is *identical* to the Kahan-vs-FP64
divergence at every checkpoint in both test cases, which proves the dominant
error is **not** accumulation roundoff (Kahan suppresses exactly that term by
~2⁻²⁴ and changes nothing). What remains is per-increment FP32 arithmetic —
the drive `S` and the `(float)(dt·qm)` product — whose rounding differences
feed back through the delta-f deposit into the fields and orbits and grow as
ordinary marker-trajectory divergence. That error channel is shared by every
accumulator choice, is 3+ orders of magnitude below marker (1/√ppc) noise, and
leaves ensemble observables untouched.

## Setup

Tool: `tools/weight_precision.cu` (`./weight_precision [nsteps=100000]
[snap=5000]`). Three accumulators via `RunParams::df_wprec` +
`Particles::enable_deltaf(s, wprec)`, flat deposit path (the tile sort does
not scatter the aux arrays; `sort_by_tile` throws if asked):

| mode | update | aux storage |
|---|---|---|
| 0 FP32 (production) | `wd += inc` | — |
| 1 FP32 + Kahan | compensated sum, `__fadd_rn` (reassociation-proof) | `wc` float |
| 2 FP64 reference | double `wdd`, mirrored to `wd` for the deposit | `wdd` double |

Two cases, nx=256 ny=1 ppc=1600 dt=0.02, 1e5 steps (t = 2000/ωpe), identical
quiet load + identical mt19937 1e-6 grid-noise seed across precisions;
particle order is stable on the flat path so per-particle comparison against
the FP64 run is valid. Divergence metric: `rms(wd − wd⁶⁴)/rms(wd⁶⁴)`.

- **equil** — iso Maxwellian (uth 0.03752, wce 0.25): no instability, wd sits
  at its noise floor (~3e-6). Cleanest window on pure roundoff.
- **growth** — the `test_deltaf_growth` gate config (bi-Max A=4, mode m=2 at
  k=1.636, γ_theory=2.872e-3): wd grows exponentially; the physics metric is
  the fitted γ per precision.

## Results (2026-07-18, RTX 5090)

Figure: `docs/figs/weight_precision.png`. CSVs:
`build/weight_precision_{equil,growth}.csv`.

**Growth:** γ fitted over the last third = **2.6474e-3 in all three
precisions — identical to 5 significant digits** (the 7.8% offset from theory
is the known ppc=1600 marker-resolution limit, see test header). Per-particle
divergence stays at 2e-5–4.6e-5 relative for 1e5 steps while rms wd grows two
decades; Kahan is marginally lower late (3.4e-5 vs 4.6e-5) — physically
irrelevant. Σwd matches FP64 to ~0.1% at t=2000.

**Equilibrium:** relative divergence grows 5e-6 → 1.4e-3 over 1e5 steps —
but Kahan tracks FP32 within ±20% at every snapshot (sometimes *above* it),
the signature of trajectory chaos seeded by per-increment rounding, not of
accumulator drift. In absolute terms the wd fields still agree to ~5e-9 at a
noise floor of 3e-6. No systematic bias: Σwd (FP32) matches Σwd (FP64) to
~0.2% throughout.

## Decision

- **`df_wprec = 0` (plain FP32) is the production configuration.** No
  measurable physics difference at 1e5 steps; the error that does exist is
  not fixable by a wider accumulator because it enters through the FP32 drive
  arithmetic all modes share.
- Kahan (`df_wprec = 1`) buys nothing measurable; FP64 (`df_wprec = 2`) costs
  8 B/particle + bandwidth for no observable change. Both remain available as
  study/debug knobs (flat path only).
- If a future regime pushes wd increments far below FP32 ulp of wd (e.g. wd
  saturated near O(1) with tiny residual drive over ≫1e5 steps), rerun this
  tool at that configuration before revisiting the default.
