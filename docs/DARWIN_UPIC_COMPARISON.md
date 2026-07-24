# Why our spectral Darwin damps free whistlers and UPIC's does not

**Date:** 2026-07-23.
**Trigger:** the full-length Li-2019 gap run on our Darwin branch
(`build/li2019_band_darwin_dt005`, 1000 tau_g at wpe·dt = 0.05) produced NO
gap / NO plateau / NO band splitting — waves stalled at dB/B0 ~ 1.5e-3 vs
1.9e-2 on the Yee arm, in quantitative agreement with the measured O(dt)
free-mode damping (gamma deficit ~0.7e-3 at dt = 0.05, ~4 e-folds over the
run = e^4 ≈ 55x amplitude deficit). Li/Bortnik/An 2019 grew the same
instability to saturation on a UPIC-lineage 1D Darwin code at wpe·dt = 0.2.
This document pins down the algorithmic difference, from a line-level read
of UPIC-2.0 (`../UPIC-2.0/mbeps1`, Decyk's mdbeps1 Darwin family) against
our `include/pic/solver_darwin.hpp` + `simulation.hpp` Darwin path.

**TL;DR — two structural differences, one of which is the O(dt) smoking gun:**

1. **Time centering (THE smoking gun).** UPIC's Darwin deposits run a
   *trial Boris push inside the deposit kernel*: dcu, amu (and cue in the
   iteration) are formed from the *centered* pair v(t−dt/2), v(t+dt/2)_trial
   — so every source feeding the field solve sits exactly at time t. Our
   deposits use the raw pre-push v(t−dt/2) — every Darwin source (J → B,
   amu → Π, dcu → E_T) lags by up to half a step. A transverse force lagged
   by ~dt/2 acting on an eigenmode e^{−iωt} is a phase-lagged force →
   numerical damping ∝ ω·dt. This is exactly the O(dt) deficit we measured,
   and it is invisible to a *driven*-pump validation (the an2019 gate),
   which enforces the phase externally.

2. **Self-consistency of the E_T response.** UPIC iterates the transverse
   solve to a fixed point in which the *actual* E_T force (magnetized, via
   the trial Boris rotation with B(t) included) enters the acceleration,
   with the resummation shift added back explicitly each sweep — at
   convergence the shift cancels and E_T is exact. We do a ONE-SHOT solve
   that *never* lets E_T act on the acceleration, replacing its self-term
   by the *unmagnetized, mean-density* resummed denominator. That is a
   dt-independent operator error (response mismatch at whistler
   frequencies, where the true E_T response is the magnetized
   conductivity, not free acceleration).

Also relevant but secondary: UPIC applies a Gaussian particle-shape factor
s(k) = exp(−(k·ax)²/2) inside every solve (noise/heating control; we have
no smoothing on the Darwin branch — cf. the +23%/1000 tau_g grid heating we
measured at dt = 0.2).

---

## 1. The UPIC mdbeps1 Darwin step (verified, file:line)

Main loop: `mbeps1/mdbeps1.py` lines 204–643. Per step (positions x(t),
velocities v(t−dt/2) entering):

1. Deposit cue (current), qe (charge) at x(t) with v(t−dt/2).
2. Poisson → fxe = E_L(t).
3. **Predictor** `darwin_predictor13` (`sd1.py:908–982`): B from cue;
   E_total guess = E_L(t) + cus(previous step's E_T); deposit dcu, amu via
   `wmgdjpost1`; **shift-back** `mascfguard1`: dcu += q2m0·cus_old; solve
   `mepois1` → cus (E_T).
4. **Iteration** k = 1..ndc (`darwin_iteration`, `sd1.py:984–1065`):
   redeposit cue, dcu, amu via `wmgdcjpost1` **with the updated
   fxyze/byze**; recompute B(t) from the re-deposited cue; shift-back;
   re-solve cus. Convergence variable = cus. Default ndc = 1 (examples),
   design doc says "converges in about 2 iterations".
5. Push with fxyze = fxe + cus and byze, all at time t; v → v(t+dt/2),
   x → x(t+dt).

### The deposit kernel is a trial Boris push (`libmdpush1.f`, GDJPPOST1L, lines 70–297)

Verbatim structure (comments + code):

```fortran
! qci = qm*dvj/dt, where dvj = (vj(t+dt/2)-vj(t-dt/2))/dt        [dcu]
! qci = qm*vj*vk, where vj = 0.5*(vj(t+dt/2)+vj(t-dt/2)) ...      [amu]
...
acx = vx + dx          ! half kick with E(t) (current guess incl. E_T)
...                    ! full Boris rotation with byze(t) (incl. external B)
dx = (rot1*acx + ...)*anorm + dx     ! = v(t+dt/2)_trial
...
ox = 0.5*(dx + vx)     ! centered velocity  v̄(t)
vy = dti*(dy - vy)     ! centered acceleration (v⁺−v⁻)/dt at time t
samu(...) += (ox*oy)*w ! momentum flux  Π(t)
sdcu(...) += vy*w      ! acceleration density  dcu(t)
```

The particle array is NOT updated — this is a *virtual* push used purely to
time-center the sources. In the iteration variant (`wmgdcjpost1`) the
current cue is ALSO re-deposited with centered velocities, so **B(t) is
time-centered too**.

### The transverse solve (`libmfield1.f`, EPOIS13, lines 762–863)

```
ffe(k)  = affp·s(k) / (k² + wp0·ci²·s(k)²),   s(k) = exp(−(k·ax)²/2)
E_T(k)  = −ci²·ffe(k)·[s(k)]·dcu_perp(k)
```

with wp0 recomputed from the ACTUAL density each step
(`calc_shift13`, `sd1.py:161–184`: wp0 = (max ωp² + min ωp²)/2), and the
shift term added back to dcu in real space every sweep (`mascfguard1`:
dcu += q2m0·cus_old). UPICModels.pdf p.12–13 states the plain iteration is
UNSTABLE for kc < ωpe; the shift stabilizes it; convergence requires
max(ωp²(x)) < 1.5·wp0². At the fixed point the added and resummed wp0
terms cancel exactly, so converged E_T contains the full (magnetized, warm)
response — the resummation is a *convergence accelerator, not a physics
substitution*.

## 2. Our Darwin step (verified, file:line)

`simulation.hpp::darwin_fields` (lines 103–119) per step:

1. Deposit ρ at x(t); J and amu with **raw v(t−dt/2)**
   (`depositor.hpp` charge_current_sorted, deposit_amu).
2. `solve_el_b`: E_L(t) from ρ; **B from the stale J** → B is effectively
   at t−dt/2 (mixed with x(t)).
3. Form E_L + pump (E_T explicitly EXCLUDED); `deposit_dcu`
   (`depositor.hpp:272–324`): dcu = qm²·w·(E + v×B) with **raw v(t−dt/2)**
   and the mixed-time B.
4. `solve_et` ONE SHOT: E_T = −green_et·(dcu − ∇·Π)_T with
   green_et = μ₀/(k² + μ₀ωpe²) — **unmagnetized, fixed mean-density n0, no
   s(k), no iteration, no shift-back** (`spectral.hpp:85–91`). The header
   comment in `darwin_fields` says "ndc fixed-point sweeps" but the code
   performs exactly one pass (the deck's `ndc` is not consumed by this
   loop).
5. Push with E_L(t) + E_T(one-shot) + B(stale).

### Consequences, itemized

| Source | UPIC time level | Ours | Effect |
|---|---|---|---|
| J → B | t (centered v̄, re-deposited each sweep) | t−dt/2 | B half-step stale in push and in dcu's v×B |
| amu → Π | t (v̄⊗v̄) | t−dt/2 (v⁻⊗v⁻) | ∇·Π lags |
| dcu | t (finite-difference of trial Boris) | mixed (E at t, v at t−dt/2) | acceleration lags ~dt/2 |
| E_T self-term | exact at fixed point (magnetized, warm, local density) | unmagnetized cold mean-density resummation only | dt-independent response error |
| E_T ← sources | iterated (ndc≥1 + predictor uses previous E_T) | one shot, previous E_T unused | lag + no self-correction |

The lagged sources are the dt-proportional channel: for a whistler
eigenmode, every Darwin field the particle feels is built from information
~dt/2 old, i.e. a force with phase lag ωdt/2 → damping rate
γ_num ~ O(ω·dt)·(mode-structure factor). Measured: deficit
(γ_Yee − γ_Darwin) = 2.78e-3, 2.17e-3, 0.70e-3 at wpe·dt = 0.2, 0.1, 0.05 —
consistent with an O(dt) law at the small-dt end (with higher-order
contributions at dt = 0.2). The driven an2019 gate could not see any of
this because a pump enforces the wave phase externally — free-running
eigenmodes are the exposing test.

## 3. Why Li 2019 worked on UPIC at wpe·dt = 0.2

Their sources are time-centered by the trial-push deposit and their E_T is
the converged self-consistent field, so the leapfrog symmetry is intact and
the residual error is O(dt²) — at dt = 0.2 that is ~(ωdt)² ≈ 1e-4-level on
γ, negligible against γ_kinetic = 2.6e-3. Our scheme loses the symmetry at
O(dt), which at dt = 0.05 still eats 27% of the growth rate — enough to
stall the instability marginally below the plateau-carving threshold for
1000 tau_g (the observed no-gap outcome).

## 4. Fix path (backlog: "Darwin time-centering fix"; gate: γ = 2.6e-3 at wpe·dt = 0.2)

1. **Trial-Boris deposit kernel** (the core fix): rewrite `deposit_dcu` to
   gather E_total-guess + B(t) + B0, run the virtual Boris update
   v⁻ → v⁺_trial (do NOT write back), and deposit
   dcu = qm·w·(v⁺−v⁻)/dt, amu = qm·w·v̄⊗v̄, cue = qm·w·v̄ in one pass
   (UPIC's wmgdcjpost1 shape). B then re-solved from the centered cue.
2. **ndc iteration + shift-back**: loop {form E_total; trial-deposit;
   re-solve B, E_T with dcu += ωp0²·E_T_old before the resummed solve};
   ndc = 2 default. This also upgrades the resummation from "unmagnetized
   substitution" to "exact at convergence".
3. Optional: Gaussian s(k) shape factor in the spectral solves (UPIC's ax)
   — addresses the separate +23%/1000 tau_g Darwin grid-heating channel.
4. Re-run gates: (a) free whistler γ at dt = 0.2 vs Yee's 2.59e-3;
   (b) the twin-arm Li gap run — if the fix is right, the Darwin arm should
   reproduce the two-band + gap at dt = 0.2 in ~25 min of wall time.

## 5. Methods-paper relevance (task #13)

This closes the loop on the twin-arm story with a mechanism, not just a
measurement: (i) driven-pump validations are structurally blind to
time-centering damping of free modes — free-eigenmode γ against a trusted
arm is the necessary gate; (ii) the UPIC Darwin's trial-push deposit is the
load-bearing detail that makes radiationless PIC leapfrog-symmetric, and it
is easy to lose in a reimplementation while passing every driven test;
(iii) with the deficit law γ_num(dt) measured, the stalled-instability
outcome of a 1000 tau_g run was predictable to within a factor ~1 in
amplitude (e^4 ≈ 55x observed) — a clean quantitative validation of the
diagnosis.

---

# ADDENDUM (same day): fix IMPLEMENTED and GATE PASSED

The UPIC scheme was ported behind a feature flag (`[field] tc = true`,
`RunParams::darwin_tc`; legacy path bit-identical when off — regression
tests pass):

- `deposit_dcj_centered_kernel` (depositor.hpp): the trial-Boris fused
  deposit — dcu = (qm w/area)(v⁺−v⁻)/dt, amu = v̄⊗v̄ (deviatoric),
  cue = v̄ on corrector sweeps (WithCue) → centered B(t).
- `DarwinSpectralSolver::solve_b` + `shift_dcu` (dcu −= ωp0²·E_T_guess,
  the mascfguard analog; ωp0² = rp.n0 = the green_et constant, so the
  fixed point is exact).
- `Simulation::darwin_fields_tc`: predictor (retained E_T = UPIC's
  cus_old) + ndc corrector sweeps.

## Gate results (Li two-band deck, 200 tau_g, m = ±7 mode-resolved gamma)

| scheme | wpe·dt | gamma (early windows) | saturation dB/B0 |
|---|---|---|---|
| legacy | 0.2  | −1.9e-4 (never leaves noise) | — |
| tc     | 0.2  | 0.94–1.08e-3 | 1.68 % |
| tc     | 0.1  | 0.94e-3 (windows coincide with dt = 0.2) | 1.71 % |
| Yee    | 0.02 | 0.86–1.27e-3 (same estimator, same windows) | 1.9 % |

- The two tc runs are dt-INDEPENDENT window-by-window → the O(dt) damping
  channel is CLOSED.
- Measured with the same estimator, tc Darwin matches the Yee arm
  mode-for-mode within ~10–20 % at 10× the Yee timestep.
- ESTIMATOR NOTE: the earlier "Yee gamma = 2.59e-3" reference came from a
  different measure; same-estimator Yee per-mode values are 0.9–1.3e-3
  (2.1e-3 transient pre-saturation). The structural conclusions of the
  main document are unchanged (legacy dead at dt = 0.2, deficit shrinking
  with dt), but cross-scheme numbers must use one estimator.
- Cost: tc ≈ 3× legacy per step (1 predictor + 2 corrector fused deposits
  + extra transverse solves) — a bargain against the ≥16× step-count
  penalty the legacy scheme would need.
- Full-length tc gap run (1000 tau_g at dt = 0.2) launched as the final
  twin-arm redemption test: two-band + 0.5 fce gap expected if the fix is
  complete.

## FINAL: twin-arm redemption (full-length tc gap run)

`build/li2019_band_darwin_tc_full` — 1000 tau_g at wpe·dt = 0.2, darwin_tc:
**two-band + 0.5 fce gap REPRODUCED** (gap/LB = 6.8e-4 vs Yee arm 3e-4;
UB present, gap/UB = 0.02; Landau plateau carved in the Vp band). Figure:
docs/figs/li2019_band_darwin_tc_kw.png. The same run that produced NOTHING
on the legacy scheme at 4× finer dt (li2019_band_darwin_dt005) now
completes at dt = 0.2 in ~1 h. Diagnosis → mechanism (UPIC trial-push
centering) → fix (darwin_tc) → validation (dt-independent gamma, Yee
match, gap reproduction): loop closed.
