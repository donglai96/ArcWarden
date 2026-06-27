# Learning Note — Step 13: The Simulation Main Loop (`simulation.hpp`)

> Goal: see **how all the pieces come together** into one PIC cycle, why the step
> order is exactly `zero → deposit → solve → push → migrate → diagnose`, why
> initialization needs a half-step velocity rollback, and how the first *end-to-end
> physics* — a cold Langmuir oscillation at ω_pe — validates the entire code at once.
> This is the payoff step: everything built in Steps 1–12 runs together.

---

## 0. What Step 13 is, in one paragraph

`Simulation<Cfg>` is the conductor. It owns the grid, particles, sources, fields, the
spectral engine, the solver, diagnostics, and the one CUDA stream, and runs the PIC
cycle each step. It's in `include/pic/simulation.hpp`, namespace `arc`. The test
seeds a cold plasma with a tiny density ripple and confirms it oscillates at exactly
the plasma frequency — the textbook check that a PIC code is wired correctly.

---

## 1. The PIC cycle — one step, in order

```cpp
void step(long n) {
    sources_.zero(stream_);                                  // 1. clear rho
    Depositor<Cfg>::charge(particles_, sources_, grid_, p_, stream_);   // 2. scatter q -> rho
    solver_.solve(sources_, fields_, spectral_, p_, stream_);           // 3. rho -> E (full FFT Poisson)
    Pusher<Cfg>::boris(particles_, fields_, grid_, p_, stream_);        // 4. gather E, advance u & x
    particles_.migrate(grid_, stream_);                                 // 5. periodic wrap + recell
    diag_.maybe_compute(n, particles_.views(), sources_, fields_, p_, stream_);  // 6. measure
}
```

Why this exact order:

1. **zero** — deposit *accumulates* with atomicAdd (Step 10), so `rho` must start
   clean each step.
2. **deposit** — particles' charge becomes the grid source `rho`.
3. **solve** — the entire `rho → E` (forward FFT, k-space Poisson, inverse FFT,
   normalization) is *inside* `solver_.solve`. There are deliberately **no stray FFT
   calls in the loop** (design §10) — the spectral engine is borrowed, not scattered.
4. **push** — gather `E` to each particle and advance velocity then position (Boris).
5. **migrate** — particles that crossed the periodic edge wrap back and their `cell`
   is recomputed. This is separate from push so each kernel stays simple.
6. **diagnose** — record energies/charge (gated by `dump_every`).

Each stage was built and tested in isolation (Steps 6–12); Step 13 only sequences
them. Everything runs on **one stream**, so the operations are ordered without
host-side synchronization between them — and it gives a single capture target for a
later CUDA Graph optimization.

---

## 2. The leapfrog start (why `init()` rolls velocities back)

Leapfrog integration stores position and velocity **staggered in time**: `x` at whole
steps `t=0, dt, 2dt, …` and `u` at half steps `t=-dt/2, dt/2, …`. But we *load*
particles with both `x` and `u` at `t=0`. If we just started stepping, the velocity
would be half a step ahead of where leapfrog assumes it is, putting a systematic
offset on the energy curve from step one (it would look like the scheme doesn't
conserve energy, when really the *start* was wrong).

So `init()` does:

```cpp
particles_.initialize(...);          // load x, u both at t=0
... allocate sources/fields ...
sources_.zero; deposit; solve;       // compute E at t=0
Pusher<Cfg>::half_step_back(...);    // roll u back to t=-dt/2 using E(0)
```

The rollback is the Step-11 `half_step_back` (a `-dt/2` velocity-only Boris update),
which needs the t=0 field — hence it runs *after* the first solve. This is exactly why
that rollback was deferred from Step 9 to live with the Pusher's gather.

---

## 3. Ownership & the template-on-Cfg design

`Simulation<Cfg>` owns everything by value: `Grid`, `Particles`, `Sources`, `Fields`,
`SpectralEngine` (which itself owns the FFT plan + k-space workspace), a *concrete*
`ElectrostaticSpectralSolver`, `Diagnostics`, and one `CudaStream`. Two design notes:

- **Concrete solver, not a variant (yet).** With a single solver, a plain member is
  cleanest; the `std::variant` static-polymorphism only earns its keep when a second
  (Darwin) solver appears (design §7.2). The `solve(...)` signature is already the one
  the variant would use, so that upgrade is frictionless.
- **`Cfg` is a template parameter.** The hot kernels (`Depositor<Cfg>`, `Pusher<Cfg>`)
  are selected at compile time, so swapping the deposit policy or enabling the B-field
  is a type change at the `Simulation<Cfg>` instantiation — the loop body never moves.

---

## 4. The physics: cold Langmuir oscillation

This is the canonical first end-to-end test. Displace a cold, uniform electron plasma
slightly and it rings: the displaced charge creates a restoring electric field, the
electrons overshoot, and the system oscillates at the **plasma frequency**
`ω_pe = √(n e²/(ε0 m))`. Under our normalization (`n0=1, e=1, ε0=1, m=1`),
`ω_pe = 1`. Crucially, for a *cold* plasma this frequency is **independent of the
perturbation wavelength** — so the answer is exactly 1 regardless of grid size.

The test seeds it with the single-k density perturbation from Step 12
(`perturb_amp=0.3`, ~3% density, linear regime — no wave breaking) and a cold load
(`vth=0`), then runs 600 steps and reads the diagnostics history. Three independent
checks:

1. **Frequency.** The field energy `EE ∝ E²` oscillates at `2ω_pe`, so its *peaks* are
   spaced `π/ω_pe`. Measured peak spacing → **ω_pe = 0.9996** (0.04% off 1.0).
2. **Energy exchange.** Kinetic and field energy should slosh back and forth in
   antiphase. Pearson correlation **corr(KE,EE) = −0.9988** — almost perfectly
   anti-correlated.
3. **Boundedness.** Total energy stays within **5.1%** (no secular blow-up) — the
   leapfrog+Boris scheme is non-dissipative and the half-step start keeps the curve
   centered.

Passing all three means deposit, the spectral Poisson solve, gather, the Boris push,
migration, the normalization contract, *and* the leapfrog start are all simultaneously
correct. One number (0.9996) exercises the whole stack.

---

## 5. Concepts you learned in Step 13 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| PIC cycle order | zero→deposit→solve→push→migrate→diag | each stage feeds the next; zero before accumulate |
| FFT not in the loop | full rho→E inside `solver_.solve` | one owner of the spectral engine |
| Single stream | all ops ordered on one CudaStream | no host sync between stages; graph-ready |
| Leapfrog stagger | x at whole steps, u at half steps | accuracy of the symplectic integrator |
| Half-step start | roll u back to −dt/2 after first solve | no systematic energy offset at step 0 |
| Concrete vs variant solver | one solver now, variant later | simplest that works; upgrade is frictionless |
| `Simulation<Cfg>` | compile-time config threads through | swap policy/B-field without touching the loop |
| ω_pe | cold plasma oscillation frequency | end-to-end correctness in a single number |
| EE at 2·ω_pe | energy ∝ field² | peak spacing π/ω_pe gives the frequency |

---

## 6. One-sentence summary

Step 13's `Simulation<Cfg>` sequences the whole PIC cycle on one CUDA stream
(zero→deposit→solve→push→migrate→diagnose) with a leapfrog half-step velocity rollback
at init, and the first end-to-end physics test — a cold Langmuir oscillation —
validates the entire stack at once: measured **ω_pe = 0.9996** vs the exact 1.0,
near-perfect KE↔EE energy exchange (corr −0.9988), and bounded total energy.
