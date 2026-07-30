# Reduced Spectral Mirror PIC (RSM-PIC) — Model Definition

Status: v1.0 "constitution", 2026-07-24. This document DEFINES the model.
All implementations (full-orbit first, gc-delta-f later, continuous-k_perp
extensions eventually) conform to it; the physical objective in Section 1
does not drift with code details. Amend this document deliberately, by
dated sections, never silently.

Companion documents: docs/GAP_PLAN.md (flagship campaign log; the RSM model
is its minimal-flagship arm), docs/CHEN2026_REPRODUCTION.md (parallel-limit
anchor), GAP_PLAN G2.1 sections (oblique-limit anchor).

---

## Contribution statement

The novelty of this model is NOT "a cheaper way to run 2D". It is:

> For the first time, parallel nonlinear chorus generation and oblique
> Landau processing are placed in one self-consistent full-f
> Vlasov–Maxwell system as two INDEPENDENTLY CONTROLLABLE degrees of
> freedom.

Every existing simulation is structurally on one side of this divide:
codes that chirp (KO2007, Hikishima, DAWN, gcPIC, our 1D) are 1D parallel
with E_par = 0 and cannot write a Landau plateau; the code that gapped
(Li 2019, 15-deg tilted periodic box) freezes ALL modes at one oblique
angle in uniform B and cannot chirp. RSM-PIC is the minimal system in
which both channels coexist and are coupled through the same particle
population — and in which the oblique channel has an off switch.

No gap is inserted by hand. No plateau, no shell, no tuned second
component is loaded. The system remains

    +----------------------+
    |   Vlasov - Maxwell   |
    +----------------------+

and the ONLY structural addition relative to the validated 1D parallel
mirror model is: a weak oblique harmonic is permitted to act on the
electrons over long times. If a gap appears, the system produced it.

---

## 1. Scope

**Objective.** Develop a reduced full-f PIC model that bridges the gap
between existing one-dimensional parallel chorus simulations and fully
two-dimensional oblique simulations.

The purpose of the model is **not** to reproduce every property of
magnetospheric chorus. Instead, the goal is to test one specific
physical hypothesis:

> **Can a weak oblique wave population self-consistently modify the
> electron distribution through Landau resonance while preserving
> discrete parallel chorus, thereby producing a self-generated spectral
> gap near 0.5 f_ce?**

Any critique of the form "the model lacks X" where X is outside this
hypothesis (upper-band amplitude, MLT structure, ducting, ...) is a
statement about scope, not about validity. See Section 2 for the
explicit lists and Section 8 for what success does and does not require.

The question this model attacks is the one left open between the two
existing lineages: not "how is the upper band generated", but

> **why, once chirping exists, is there still no gap** — i.e. what is
> the minimal additional physics that lets a chirping system carve one.

---

## 2. Physical assumptions

### Retained physics

- mirror geometry (inhomogeneous |B| along the field line; the chirping
  inhomogeneity factor in Omura's sweep-rate theory)
- nonlinear cyclotron trapping (full-orbit particles: resonant islands,
  phase bunching, hole/hill dynamics emerge automatically)
- discrete chorus elements (rising tones with amplitude-controlled sweep;
  validated in the parallel limit against our Chen PoP 2026 reproduction)
- full-f evolution of the kinetic species (no delta-f linearization in
  the first implementation; plateau formation, tail depletion, and
  cumulative distribution reshaping are all retained)
- finite E_par (carried by the oblique harmonic; identically zero in the
  parallel limit)
- Landau resonance (n = 0), acting on the KINETIC species' tail at
  v_par = Vp = omega/k_par (~0.08–0.10 c in the reference parameters)
- finite Larmor radius effects and all cyclotron harmonics n (full-orbit
  push; no gyro-averaging anywhere)
- self-consistent field–particle coupling in ALL retained modes (the
  parallel and oblique channels exchange energy through the shared
  particle population; the m=+1 x m=-1 beat onto m=0 is also retained —
  see Section 3)
- WNA evolution ALONG the field line at fixed k_perp: k_perp is
  quantized and conserved, but k_par evolves in the inhomogeneous B, so
  the local wave normal angle of the oblique harmonic changes with
  "latitude" exactly as in the paraxial limit of ray tracing. (Li's
  fixed tilt has no analog of this.)
- magnetic-tube (flux-tube) geometry: one field line, absorbing wave
  ends, trapped/precipitating particle dynamics along it

### Neglected physics

- continuous WNA spectrum (only m = 0, +-1 harmonics; the k_perp cascade
  m=1 -> m=2 -> ... is truncated)
- mode conversion requiring k_perp continuity (e.g. oblique whistler ->
  quasi-electrostatic branch evolution across a continuum of angles)
- CROSS-FIELD refraction / ray bending in y (k_perp frozen by
  quantization). NB: the along-field half of refraction IS retained (see
  above); only the transverse half is cut. State both halves when citing
  this assumption.
- cross-field density variation (no ducting)
- thermal-cold Landau response: the cold population is a T = 0 fluid
  (cold_model=full); Landau resonance lives exclusively on the kinetic
  hot/warm species. A delta-f kinetic cold extension is the designated
  upgrade if cold Landau absorption proves relevant (GAP_PLAN G3).
- radial diffusion, drift-shell splitting, MLT dependence, multiple
  L-shells (single flux tube)
- ion kinetic effects (ions = immobile neutralizing background on
  whistler timescales)

Reviewer protocol: "you have no MLT / no ducting / no continuous WNA" ->
Yes. Not intended. See Section 1.

---

## 3. Wave-vector content: parallel + one oblique harmonic

A genuine 2D simulation retains a filled half-plane of wave vectors:

```text
k_perp
  ^
  | x x x x x x x x x x x x x x x
  | x x x x x x x x x x x x x x x
  | x x x x x x x x x x x x x x x
  +-------------------------------->  k_par
```

RSM-PIC retains three lines:

```text
k_perp
  ^
  | . . . . . . . . . . . . . . .   <- m = +1  (k_perp = +k1)
  |
  +-- . . . . . . . . . . . . . -->  m =  0   (parallel; k_perp = 0)
  |
  | . . . . . . . . . . . . . . .   <- m = -1  (k_perp = -k1)
```

Each line is a full 1D continuum in k_par; only k_perp is truncated to
{0, +-k1}. Reality of the fields ties m = -1 to the conjugate of m = +1.

> This is not intended to reproduce the full WNA spectrum. Rather, it
> isolates the weakest possible extension beyond the purely parallel
> model.

Wave–wave coupling bookkeeping (be precise when asked):
- retained: parallel <-> oblique coupling THROUGH THE PARTICLES (the
  mechanism under test); the second-order beat m=+1 x m=-1 -> m=0.
- truncated: m=1 x m=1 -> m=2 and the entire higher-k_perp cascade.
If the gap mechanism secretly requires that cascade, this model returns
a null — which is Case A of Section 9 and is itself the result.

Choice of k1 (reference parameters, Section 6): k1 places the oblique
harmonic at the Li-validated wave normal angle in the lower band.
theta(omega) = atan(k1 / k_par(omega)); with k1 = 0.16 wpe/c the
harmonic sits at ~15 deg at k_par = 0.6 (lower band) and ~9 deg at
k_par = 1.0 (omega = 0.5 Omega_e). k1 is a declared knob of the model
(scan range: WNA 10–25 deg in the lower band), not a hidden fit.

---

## 4. Why not full 2D

Not "because 2D is expensive". The reason is control.

In a full 2D mirror box, wave propagation, WNA broadening, mode
coupling, transverse refraction, Landau damping and cyclotron trapping
all switch on SIMULTANEOUSLY, entangled in one output. When such a run
produces (or fails to produce) a gap, the attribution question — WHICH
ingredient carved it — is exactly as open as before the run. This is
the standing weakness of the short 2D literature runs: they demonstrate
phenomena, not mechanisms.

RSM-PIC adds ONE degree of freedom to the validated parallel model: the
oblique channel, with amplitude/presence R_obl and placement k1. Every
experiment is an A/B against R_obl = 0. This is a controlled model, not
a cheap model. (That it is also ~an order of magnitude cheaper than 2D
is what makes the long cumulative-gap timescales and parameter scans
affordable — a consequence, not the rationale.)

---

## 5. Relation to Li and Lu

- **Li 2019 (NatComm, tilted uniform box)** answers: can an oblique wave
  population, given E_par and long times, self-consistently carve a gap?
  (Yes — and we have reproduced it in-house: gap/LB -> 3e-4 over
  ~1000 tau_g, plateau x10 in [0.08, 0.10] c.) But: uniform B, periodic
  recirculation, all modes frozen at 15 deg — it cannot chirp, and it
  demonstrates the loop with incoherent waves, not chorus.
- **The Lu-group series (Ke 2017 ... Ke 2025, 2D mirror gcPIC)** answers:
  does mirror geometry produce chirping chorus in 2D? (Yes.) But: cold
  population is a T = 0 fluid in every run, so the Landau channel on
  cold electrons is absent by construction, and NO run in the series
  shows a gap or a two-band structure.

No existing model answers whether the two can coexist. RSM-PIC is the
minimal intersection: Lu's mirror + Li's oblique channel, one particle
population coupling both, everything else stripped.

|                       | chirping | E_par/Landau | gap shown | controlled A/B |
|-----------------------|----------|--------------|-----------|----------------|
| 1D parallel (all)     | yes      | no (=0)      | never     | —              |
| Li 2019 tilted        | no       | yes (15 deg) | yes       | no             |
| Lu 2D mirror series   | yes      | fluid-cold   | never     | no             |
| **RSM-PIC**           | yes(L0)  | yes (m=+-1)  | HYPOTHESIS| **yes: R_obl** |

---

## 6. Mathematical structure

The loop is standard PIC, unchanged:

```
Particles  ->  Current deposition  ->  Field solve  ->  Particle push
    ^                                                        |
    +--------------------------------------------------------+
```

What changes is ONLY the field/current representation. Fields:

    F(x, y, t) = F_0(x, t) + 2 Re[ F_1(x, t) e^{i k1 y} ]

i.e. one real 1D field set (m = 0) plus one complex 1D field set
(m = +1); m = -1 is its conjugate. Particles remain FULL ORBIT in
(x, y, vx, vy, vz); y enters the wave coupling only through the phase
k1 y_p.

- Gather:  F(x_p, y_p) = F_0(x_p) + 2 Re[ F_1(x_p) e^{i k1 y_p} ]
- Deposit: J_m(x) = sum_p w_p v_p S(x - x_p) e^{-i m k1 y_p},  m = 0, 1
- Mirror background: applied ANALYTICALLY in the pusher, never through
  the truncated wave field. Two conforming choices:
  (i) PARAMETRIZED gc mirror-force term on full-orbit particles with
  y-independent B0x(x) (Lu-lineage "mc term", our b0_prof 1/2). The
  background is then exactly y-homogeneous, so every linear operator in
  the loop (Yee curls, absorbing masks, binomial filter, cold-fluid
  rotation about b-hat) is diagonal in the k_perp harmonic index m —
  the truncation is EXACT: masking the deposited current is sufficient,
  nothing repopulates discarded modes. First implementation uses (i).
  (ii) RESOLVED div-B = 0 slab (prof=3: B0x = B0(1 + a x~^2),
  B0y = -2 a B0 x~ y~). B0y ~ y weakly couples m -> m+-1 through the
  background (small at small Ly); this is the natural rung toward the
  full 2D model, not the first implementation.
- Cold electrons: T = 0 fluid (cold_model=full, symmetric staggered
  operators, oblique dispersion gated to 0.003–0.46 %), diagonal in m.
- Boundaries: x = absorbing for waves (masks), with the cavity-gain
  budget checked (gamma L / v_g >= ~5 e-folds of runway); y = periodic
  with Ly = 2 pi / k1 implicitly.

Cost: ~3x a 1D run (three 1D field sets + one complex exponential per
particle per gather/deposit) — NOT ny-times anything.

Two conforming implementations (the model is representation-agnostic):

- **(A) Direct spectral**: complex F_1(x) arrays, phase-factor
  gather/scatter as above. Cleanest; new deposit/gather kernels needed.
- **(B) Filtered narrow-y grid**: the existing (gated) 2D code with
  small ny (>= 8) and a per-step k_y projection that zeroes every mode
  with m not in {0, +-1} in the deposited current (and once in the
  initial seed). Mathematically the same truncation (deposit-then-
  project = project-then-deposit up to the shape-factor transfer
  function). Source-verified facts (2026-07-24): particles ALREADY
  store (x, y) and full 3V (particles.hpp Particles struct), the 2D
  Esirkepov deposit and 2D Yee solver already exist, and the oblique
  cold-plasma response (v_par response, E_par shielding) is
  cold_model=full, gated by test_cold_fluid_oblique to 0.003–0.46 %
  with zero numerical damping. The ONLY new code is the ky projection
  kernel + a deck flag. Cost ny x 1D with small ny.

First implementation: **(A)** — amended 2026-07-24, user decision: the
code stays 1D. ny = 1 everywhere; the oblique content lives purely in
the complex spectral amplitudes F_1(x). (B) remains a conforming
alternative and the rung toward full 2D. Consequences of (A):
- the y-derivative is EXACT (d/dy -> i k1): no grid dispersion, no
  aliasing, no resolution rule in y at all;
- k1 is a free deck parameter, untied to any box width;
- particle y is a pure phase coordinate (already stored; advanced
  ballistically, used only as e^{+-i k1 y_p} in gather/deposit);
- with the flag off, the m = 0 path is LITERALLY the unmodified legacy
  1D code — no new kernels launch, no arrays allocate: the
  non-regression guarantee holds by construction.
Numerical care point of (A): charge conservation for the m = 1 system
(continuity: dt rho_1 + dx J1x + i k1 J1y = 0). E_par accuracy is the
entire point of the oblique channel, so the m = 1 longitudinal field
must be Gauss-consistent: direct (shape x phase) deposit plus a
per-step 1D complex Gauss correction of E1 (cheap tridiagonal/spectral
solve), verified in the dispersion/polarization benchmark before any
physics run. This section is the one to keep open while coding.

### 6.1 Implementation risk register (2026-07-24 design review; static
### audit by the user's reviewer, claims source-verified on our side)

R1 — PHASE COORDINATE, the trap that would have fired first: at ny = 1
every loader places ALL particles at y = 0.5 (particles.hpp, verified).
A direct e^{-i k1 y_p} deposit is then a COHERENT sum — a huge
artificial oblique seed instead of N^{-1/2} shot noise, i.e. instant
fake E_par. Required: a dedicated phase state
theta_p = k_perp y_phys (mod 2 pi), loaded UNIFORM in [0, 2 pi)
independently of gyrophase, advanced as dtheta/dt = k_perp u_y / gamma.
If p.y storage is reused, it is REDEFINED under the rsm flag as
theta/2pi with rsm-only advance semantics — never real y-cell
semantics.

RESOLVED 2026-07-24 (implementation, rsm_oblique.hpp): p.y is reused
WITHOUT redefinition. At ny = 1 the legacy pusher already advances y by
v_y dt / dy and wraps it mod 1, and the m = 0 gather/deposit are
invariant to its value (every y-row is the same row; the Esirkepov Jy
still encodes v_y through Delta-y, which the phase view preserves). With
the deck contract Ly = 2 pi / k1 (enforced by the [rsm] finalize block
and RsmState::init), cell-unit y IS theta/2pi and the legacy advance IS
dtheta/dt = k_perp u_y / gamma exactly — no new particle array, no
sort/migrate/checkpoint plumbing, no pusher change for the phase.
Float32 phase step ~2e-3 cells/step at k1 = 0.16: accumulation error is
a ~1e-7-relative random walk, negligible over 3e5 steps. Only the R1
remedy remains a real kernel: rsm_theta_init randomizes y uniform in
[0, 1) on RNG stream 11 (loaders use 0-5) after the load. Gate V1
(tests/test_rsm_phaseload.cu) PASSED: pinned load coherence |S|/N =
1.000000 (trap demonstrated), randomized E|S_rho|^2/N = 1.06/0.87 and
E|S_jz|^2/(N<uz^2>) = 0.92/0.73 at N = 2^18/2^20 (Exp(1) statistics,
32 seeds, bounds [0.5, 1.6]).

R2 — FULL m = 1 STATE: complex Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz AND
rho_1. Monitor the solenoidal constraint Dx B1x + i k_perp B1y = 0
every diagnostic step. The legacy white-noise By/Bz seed is ILLEGAL at
k_perp != 0 (violates div B = 0); the oblique seed must be a
divergence-free, Gauss-consistent eigenmode or be projected first.

R3 — CHARGE CLOSURE, the most dangerous numerical point: discrete
continuity (rho1^{n+1} - rho1^n)/dt + Dx J1x + i k_perp J1y = 0.
Preferred: modal/worldline Esirkepov (analytic time integral of
S(x - x_p(t)) e^{-i theta_p(t)} along the substep — the quasi-3D
azimuthal-PIC construction); fallback: strictly matched complex Gauss
projection. Either way the correction MUST include the cold-fluid
longitudinal charge rho_c1: cold compression IS the E_par screening
physics, and a hot-only projection overestimates E_par — the easiest
way to FAKE a Landau plateau and a gap.

RESOLVED 2026-07-24 (k_rsm_push_esirkepov): worldline Esirkepov, NO
projection anywhere. The per-step charge change factorizes EXACTLY:
S1 e1 - S0 e0 = ebar (S1 - S0) + Sbar (e1 - e0), ebar = (e0+e1)/2,
Sbar = (S0+S1)/2, e = e^{-i theta}. First term -> 1D Esirkepov prefix
sum W -> J1x links with phase ebar; second term -> J1y(i) =
i qw Sbar_i (e1-e0)/(k1 dt dV) — the analytic worldline integral
(equals qw vy_tilde Sbar e^{-i theta_bar}, vy_tilde =
2 sin(k1 vy dt/2)/(k1 dt) -> vy). Continuity holds to float roundoff
BY CONSTRUCTION — gate V3 measured 4.9e-5 (atomics roundoff). Because
there is no projection step, the cold-inclusive-Gauss trap cannot
fire: cold rho_c1 never needs reconstructing (cold J1c feeds Ampere
directly, same as m0).

R4 — ONE PUSH: gather the TOTAL field E0 + 2 Re[E1 e^{i theta}],
B0 + dB0 + 2 Re[B1 e^{i theta}], then a single Boris rotation; deposit
J0 and J1 from the SAME worldline (two deposit passes over one
trajectory are fine; two pushes are not — they break time-centering,
energy, and trapping).

RESOLVED 2026-07-24 (k_rsm_push_esirkepov): one staggered gather of
F0 + one 1D complex gather of F1 -> total field -> single Boris
kick-rotate-kick (same op sequence as yee_advance_particle; m1 dB
rides in the wave dB so b0_prof mirror branches see the total field);
then m0 esirkepov_scatter AND the modal m1 deposit from the same
(x0,y0)->(x1,y1) worldline. Gate V3-A: deterministic kick matches
qm dt 2 Re[E1 e^{i theta}] to 1e-6 (float32); V3-D: energy ledger
W = KE + Wm0 + 2 Wm1 conserved to 3.6% of transfer through a full
anisotropy-instability run with m1 growth to 2W1 ~ 2.2.

R5 — COLD FLUID: the m = 1 channel requires the full 3-component
complex cold response (new code). test_cold_fluid_oblique gates the
REAL-SPACE 2D path only; the spectral path needs its own
dispersion/polarization gate, cross-checked against BOTH the exact
Stix solution and the real-space cold_full implementation (two
independent codes must agree). The m = 0 arm of an RSM run keeps its
legacy transverse cold fluid (validated for parallel chorus).

R6 — FACTOR-2 LEDGER: Ampere uses the Fourier coefficient J1 (no
factor 2); y-averaged energy and work are W = W0 + 2 W1 and
P = J0.E0 + 2 Re(J1* . E1). All RSM energy/work diagnostics are new
code — the real-field diagnostics must not be silently reused, or
growth rates, R_obl, and gap depth are all off by 2.

Semantics and secondary items:
- m = -1 is the REALITY CONJUGATE of m = +1, not a counter-propagating
  wave. Directional (one-way/two-way) splits are by sign(k_par) or Sx,
  never by "switching off m = -1".
- Fixed k_perp is NOT fixed WNA: theta_WNA = atan(k_perp/k_par)
  evolves as the chirp moves k_par. Calibrate k_perp to the target
  band through the linear dispersion when comparing to Li's 15 deg.
- Multi-mode extension only commensurate: k_m = m k0, so one theta
  serves all modes (deposit e^{-i m theta}). Incommensurate k_perp
  lists have no single periodic phase coordinate — that regime is
  effectively true 2D and out of scope for this arm.
- Normalization: there is NO y box. Weights are per-unit-transverse-
  length line densities; the phase period is 2 pi. (Supersedes the
  earlier "Ly = 2 pi / k1 implicitly" wording.)
- jfilter at ny = 1 filters x only; under RSM it must act identically
  on Re/Im of J1 AND on rho1, so the FILTERED continuity still closes.
- Boundaries: the x absorbing layer is unvalidated for oblique waves,
  and the hybrid boundary damps transverse particle momentum — long
  plateau/gap runs require a boundary A/B scan (BOUNDARY_STUDY.md).
- Checkpoint: complex m = 1 fields, cold m = 1 state, and the theta
  stream must enter the checkpoint schema (current manifest covers
  real Yee state only).
- Loader: the mirror loader rejects kappa_v (verified) — "mirror +
  kappa tail" needs loader work before Level 2; the uniform-B Level 1
  is unaffected (call it a Li-TYPE benchmark, not a Li replication).

Validation ladder (supersedes the earlier B1/B2 list):
- V0: RSM-off statistical regression on the Chen case-2 path, BOTH
  deposit paths (flat + tile_sort) — executed 2026-07-24, PASS.
- V1: phase-load test — <rho1>, <J1> zero-mean, amplitude scales as
  N^{-1/2}.
- V2: uniform B, periodic x — spectral cold dispersion, polarization,
  E_par/E_perp vs exact AND vs real-space cold_full.
- V3: single- and few-particle tests — complex continuity, Gauss,
  div B1, and the energy / field-particle-work / boundary-loss ledger
  closing.
- V4: uniform-B Li-type plateau (both channels, no mirror).
- V5: mirror + chorus coexistence (Level 2).
- V6: notch forensics BEFORE any gap claim: m0/m1 interference on a
  single fixed-phase lineout can fake a notch. A Level-3 claim is made
  only on y-averaged modal spectra (P0 and 2 P1 separately + total),
  with E_par,1(x,t), Gauss residual, and direction-resolved power and
  field-particle work in evidence.

Non-regression requirement (binding): the k_y projection is a SEPARATE
kernel behind an off-by-default deck flag; no signature or call-path
change to any kernel used by the validated parallel chirping path. With
the flag off, the Chen 2026 case-2 reproduction must pass the
statistical-equivalence gate scripts/regress_case2.py (bit-identity is
unattainable: GPU float atomics make even same-binary reruns differ —
measured 2026-07-24; the gate is therefore candidate-vs-reference
envelope <= 3x the same-build run-to-run envelope).

Reference parameters (code units c = wpe = 1): wpe/Omega_e = 5
(wce = 0.2), hot kappa-tail species per the Li-repro loader (0.6 c cap),
nh ~ 0.006–0.01, cold_nc ~ 0.99, mirror a per chirping runs. Landau
band: Vp = omega/k_par in [0.08, 0.10] c for omega in [0.25, 0.5]
Omega_e — the plateau window validated in the Li reproduction.

---

## 7. Particle representation (why not guiding-center)

The proposed model is independent of the particle representation. Both
guiding-center particles and full-orbit particles are compatible with
the reduced spectral geometry.

The first implementation adopts full-orbit particles because:

1. the existing one-dimensional full-orbit code is already available
   and validated in both limits (Section 8, Levels 0–1);
2. finite-Larmor-radius effects, cyclotron harmonics, and nonlinear
   phase trapping emerge automatically, with no gyro-averaging
   assumption to defend at omega ~ Omega_e (where gyro-averaging is
   precisely the approximation that fails);
3. this minimizes the number of additional approximations present
   during the first validation stage — the truncation to three k_perp
   harmonics is then the ONLY reduction in the model.

Future implementations may migrate to a delta-f guiding-center (or
delta-f full-orbit) formulation for computational efficiency, once the
full-f full-orbit version has established the reference behavior.

---

## 8. Success criteria

Graded levels, each gated against an IN-HOUSE reproduction, not only
against literature figures. Declare the level reached; never blur.

- **Level 0 — parallel limit reproduces the chirping anchor.**
  R_obl = 0 (oblique channel off), mirror on. Discrete rising tones;
  sweep rate within the validated band of our Chen PoP 2026
  reproduction (0.72–0.73 x Omura Eq. 88 at the same amplitude);
  threshold/saturation consistent with the lre-scan matrix.
- **Level 1 — oblique limit reproduces the gap-loop anchor.**
  a = 0 (uniform B), oblique channel on. Landau plateau rises ~x10 in
  the Vp band [0.08, 0.10] c; cumulative spectral-notch deepening on the
  Li clock (gap metric falling monotonically by ~800 tau_g), consistent
  with our Li 2019 tilted reproduction (gap/LB -> 3e-4 at 900–950
  tau_g). Documented deviation: Li's tilt freezes ALL modes at 15 deg,
  whereas here a parallel continuum coexists with the 15-deg harmonic.
- **Level 2 — coexistence.** Both on. Chirping persists (element
  occurrence and sweep rate within ~2x of Level 0) AND the oblique
  channel is demonstrably active (plateau forms in the Vp band).
- **Level 3 — self-generated gap.** A LOCAL MINIMUM of wave power near
  0.5 f_ce that deepens cumulatively in time (gap/LB <= 1e-1 and
  falling), with FINITE power above the notch. The finite upper
  shoulder is part of the definition: a spectrum that merely cuts off
  at 0.5 is not a gap. Report gap/LB(t) exactly as in the Li-repro
  analysis so the numbers are comparable.
- **Level 4 — distinct upper band** of observational-grade amplitude.

> Level 4 is not required for the proposed hypothesis.

The hypothesis of Section 1 is confirmed at Level 3. Levels 0 and 1 are
prerequisites for interpreting anything above them: no Level-2 claim
without both limits passing first.

---

## 9. Interpretation of outcomes (failure means what)

Every outcome is physics. Pre-registered readings:

- **Case A — no gap (Levels 0–2 pass, 3 fails).** The weakest oblique
  extension is insufficient: a continuous WNA spectrum (or the
  higher-k_perp cascade, or transverse refraction) is REQUIRED for gap
  formation. This is a strong, publishable constraint on every proposed
  gap mechanism. Before declaring Case A, exclude the three confounds:
  (i) runtime — the gap is CUMULATIVE, carving over 300–900 tau_g in
  the Li reproduction; runs shorter than ~1000 tau_g cannot claim a
  null; (ii) oblique amplitude below the plateau-writing threshold —
  scan R_obl upward; (iii) k1 misplaced — scan WNA 10–25 deg.
- **Case B — gap forms, no upper band (Level 3, not 4).** Gap carving
  and upper-band generation are separate mechanisms; the upper band
  needs fuel (low-energy anisotropy) or physics (continuous spectrum)
  not present here. Consistent with the hypothesis; sharpens the
  two-shoulder question.
- **Case C — chirping destroyed by the oblique channel (Level 2
  fails).** The oblique amplitude that suffices for Landau processing
  is incompatible with coherent trapping at these parameters — an
  interference constraint between the two channels; map the R_obl
  boundary.
- **Case D — gap AND chirping coexist (Level 3 with Level 2 intact).**
  The hypothesis is confirmed; the minimal sufficient physics for a
  chirping system to carve its own gap is one weak oblique harmonic.

---

## 10. Model hierarchy

```
        Real magnetosphere
                |
                v
        2D mirror PIC  (Lu series; Ke17 replication arm in GAP_PLAN)
                |
                v
   >> Reduced spectral mirror PIC  (THIS MODEL) <<
                |
                v
        Parallel mirror PIC  (chirping codes; our 1D — Level-0 anchor)
                |
                v
        Uniform tilted PIC  (Li 2019; our G2.1 repro — Level-1 anchor)
```

RSM-PIC is **not an approximation of the 2D model**. It is the bridge
model between the Li lineage and the Lu lineage: the minimal
self-consistent system containing both of their essential degrees of
freedom, with the oblique one under experimental control. The ladder is
two-way: results propagate UP (a Level-3 gap here predicts what the 2D
run must show and where) and DOWN (a Case-A null sends the question
back to the continuous-spectrum rung with a sharpened requirement).

This document is the project constitution. Implementations, decks, and
run campaigns cite it; they do not silently redefine it.
