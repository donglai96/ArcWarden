# Learning Note — Step 12: Diagnostics (`diagnostics.hpp`)

> Goal: understand **what a PIC code measures to know it's working** — kinetic
> energy, field energy, total charge, max|E| — how those map to the ωpe=1
> normalization, why the reductions must be done in *double*, and the decoupling
> that keeps `Diagnostics` independent of `Simulation`. This is the instrumentation
> layer; Step 13 turns it into a conservation curve.

---

## 0. What Step 12 is, in one paragraph

A simulation that runs is not the same as a simulation that's *correct*. Diagnostics
computes the handful of scalars that reveal correctness — energies (which should
exchange but sum to a bounded total), total charge (which must stay constant), and
max|E| (a health/units check) — and logs them to CSV each dump step. It's one class,
`Diagnostics`, in `include/pic/diagnostics.hpp`, namespace `arc`, deliberately
decoupled from the rest. The test validates every reduction against a hand-computed
state.

---

## 1. The four scalars and their formulas

Under the ωpe=1 normalization (`config.hpp`: eps0=me=|e|=1), with a macro-particle
representing `weight = n0·dx·dy/ppc` physical particles of mass 1:

| Scalar | Formula | Meaning |
|---|---|---|
| Kinetic energy `ke` | `weight · ½ · Σ_particles (ux²+uy²+uz²)` | energy of motion |
| Field energy `ee` | `½ · eps0 · Σ_cells (Ex²+Ey²) · dx·dy` | energy stored in E |
| Total `total` | `ke + ee` | the conserved quantity |
| Charge `charge` | `Σ_cells rho · dx·dy` | must stay constant |
| `max_e` | `max_cells √(Ex²+Ey²)` | units/health check |

Two normalization subtleties worth pinning down:

- **Why `weight` in `ke`.** Each macro-particle stands for `weight` real particles of
  mass `me=1`, so its kinetic energy is `½·(me·weight)·v²`. The `weight` converts
  "per macro-particle" to "physical energy," and it's the *same* `weight` that scales
  the deposited charge (Step 10) — keeping particle and field energies in the same
  units so their sum is meaningful.
- **Why `dx·dy` in `ee` and `charge`.** Field energy density is `½eps0 E²` *per unit
  area*; summing over cells and multiplying by the cell area `dx·dy` gives the total.
  Same for charge: `rho` is a density, so `Σ rho · dx·dy` is total charge.

---

## 2. Why double precision matters here

The fields and particles are stored in **float** (Step 4: that's the hot-path
precision). But conservation is judged by *small drifts* — you want to see whether
total energy wanders by 0.1% over thousands of steps. If you accumulated a sum of
thousands–millions of float terms in float, the rounding of the accumulation itself
would be larger than the drift you're trying to detect, masking the signal. So the
reductions accumulate in **double** (plan §17). The data stays float; only the *sums*
are double. This is a general numerical principle: reduce in higher precision than you
store.

> v1 does the reduction by copying the (modest) arrays to the host and summing in a
> `double` loop — simple and obviously correct. Diagnostics run only every
> `dump_every` steps, so the copy cost is amortized; a fused on-GPU double reduction
> is a later optimization, not a correctness requirement.

---

## 3. Decoupling: no `Simulation&` in the signature

```cpp
DiagSample compute(long step, ParticleViews p, const Sources& src,
                   const Fields& f, const RunParams& rp, cudaStream_t s);
```

Diagnostics takes exactly the views/containers it reads — particles, sources, fields,
params — and **not** a `Simulation&` (design §9). If it took the whole `Simulation`,
the two classes would depend on each other (Simulation owns Diagnostics, Diagnostics
references Simulation) — a circular dependency that makes both harder to test and
compile. By passing only the data it needs, `Diagnostics` can be unit-tested with a
hand-built state (which is exactly what the test does) and `Simulation` just hands it
`particles_.views(), sources_, fields_, p_`.

`compute()` always runs and appends to an in-memory `history` (handy for tests and for
post-run analysis); `maybe_compute()` is the loop-facing wrapper that runs only on
dump steps (`step % dump_every == 0`) and writes the CSV row. The CSV columns are
`step,time,ke,ee,total,charge,max_e`.

---

## 4. A subtlety the test caught

`compute()` records to history but does **not** write the CSV — that's
`maybe_compute()`'s job. The first version of the test called `compute()` directly
and then tried to read the CSV, which was empty. The fix mirrors real usage: call
`maybe_compute()` (which computes *and* writes), then read the result from
`diag.last()`. A small thing, but it's the kind of API-contract detail a test is
supposed to surface.

---

## 5. How we verified Step 12

Each reduction has a closed form for a uniform state, so the test sets one and checks:

- Fields `Ex=0.3, Ey=-0.4` uniform → `EE = ½·eps0·(0.3²+0.4²)·Ncell·dx·dy = 1.0` ✓,
  `max|E| = 0.5` ✓.
- `rho = 2.0` uniform → `charge = 2.0·Ncell·dx·dy = 16.0` ✓.
- Maxwellian-loaded particles → `KE = weight·½·N·3vth² = 12.0` analytically, measured
  `11.99` (the quiet-start std is 0.9999, not exactly 1). ✓
- `time = step·dt`, `total = ke+ee` ✓, and the CSV row begins `7,...` ✓.

CTest 7/7; `compute-sanitizer` clean.

---

## 6. Concepts you learned in Step 12 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Energy bookkeeping | `ke` (particles) + `ee` (field) | total is the conserved quantity |
| `weight` in `ke` | macro→physical energy conversion | particle & field energy in same units |
| `dx·dy` factor | density → total (energy, charge) | correct extensive quantities |
| Reduce in double | sum float data in double | drift signal not lost to rounding |
| Decoupling | take views, not `Simulation&` | no circular dependency; unit-testable |
| `compute` vs `maybe_compute` | always vs dump-gated + CSV | history for analysis, CSV for runs |
| CSV schema | `step,time,ke,ee,total,charge,max_e` | one row per dump, plot conservation |

---

## 7. One-sentence summary

Step 12's `Diagnostics` computes the conservation scalars — kinetic energy
`weight·½Σu²`, field energy `½eps0ΣE²dx dy`, total charge `Σρ dx dy`, and max|E| — in
**double** to resolve small drifts, decoupled from `Simulation` (it takes views, not a
back-reference), logging a CSV row per dump step; every reduction is verified against a
hand-computed uniform state.
