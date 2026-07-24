# Boundary-refresh ("infinite train") experiment — design record (2026-07-22)

Status: DESIGN ONLY, no code yet. User-approved constraints: must be a
FEATURE (zero impact on existing code paths), and all discriminating
experiments must keep clean discrete elements (no noise-floor/ppc tricks).

## Question

Closed-box repetitive elements die out and their period stretches
(x10: 894 -> ~1500/Oe then extinction) because the free-energy reservoir
drains. Real chorus repeats for hours at steady period because
gradient-curvature drift feeds the source region fresh hot electrons.
Hypothesis: with drift modeled as a knob-free exchange, the element train
becomes an endless limit cycle with constant period.

## Why not taud (and who did what in the literature)

- taud/τ_D-style injection = an IMPOSED clock; "period follows injection
  rate" is circular. Precedent: USTC gcPIC family — Lu et al. 2021,
  Chen (Huayue) et al. 2022 GRL (τ_D = 5000), Kong et al. 2023 JGR
  (τ_D scan: slow injection -> discrete elements, fast -> hiss-like
  continuum). Cite as the contrast, not the method.
- Chen et al. 2026 PoP attribute their single-element (non-repetitive)
  Case II to "particle injections are not included" — but OUR closed box,
  same physics, no injection, produced 5-6 repeating elements with
  period ∝ T_b. Repetition needs bounce refill, not injection; injection
  sets how LONG the train runs. This is a direct correction to their
  attribution (their δf 400 ppc vs our full-f 110k ppc noise floor is the
  likely cause of the difference).
- Nunn's VHS / Demekhov BWO: short equator-only boxes where electrons
  stream through and enter fresh — refresh by construction, but no mirror
  points in the box, hence no bounce clock to test.
- Knob-free refresh in a full-field-line box containing mirror points:
  nobody has done it (KO2007/2016, Hikishima, Tao DAWN, Chen 2026 all
  reflect = closed).

## Design: high-latitude thermal bath, stateless

Rule: any hot marker in the bath region (|lambda| > lambda_R = 15 deg)
that is moving OUTWARD gets its velocity redrawn from f0(local) every
step; position untouched; inward movers untouched.

- Stateless: pure function of current (x, u) — no crossing detection, no
  history flag, no x0. Lives in a standalone post-push kernel.
- Outward movers are repeatedly reset to f0 until a redraw points inward;
  the particle then re-enters the physical region as a fresh f0 sample.
  Refresh rate = the particle's own bounce traffic through lambda_R:
  the supply clock IS the bounce clock. No imposed timescale.
- Sampler: draw equatorial loss-cone sample, transport to local x via
  (E,mu) conservation, reject if it cannot reach x (same math as the
  mirror-equilibrium loader; self-contained copy in the new header, do
  NOT touch the loader). Bounded retries (<=16), keep old velocity on
  failure (count it).
- lambda_R = 15 deg: outside the generation region (equator +-10), inside
  the particle wall (~24-26.6 deg). Geometric choice, not a physics knob;
  insensitivity check at 12/18 deg. Known bias (state honestly): alpha_eq
  >~55 deg particles mirror below lambda_R and are never refreshed; the
  resonant fuel (alpha_eq ~45 deg) all is. lambda_R does NOT change WHEN
  particles return (their own orbit does), only WHETHER they are washed —
  so it cannot masquerade as a clock.
- Pitfalls that shaped this design: refresh at the WALL only washes the
  loss cone (trapped majority never reaches the wall); refresh at the
  EQUATOR kills phase bunching mid-formation.
- This is STEADY-STATE exchange (inflow = outflow, n_h fixed): the 1:1
  exit-swap is the fixed-N approximation of absorb + independent
  bath-flux injection; the difference is second order (waves change pitch
  angle/energy of resonants, barely the number flux through lambda_R),
  and variable-N machinery on GPU is expensive. Net injection (substorm
  onset, n_h(t) ramp in the bath f0) is a reserved FUTURE extension —
  do not mix into this experiment. Precipitation mode (absorb loss cone
  at wall + count) is a separate switch to build in the same area later.

## Feature isolation (user requirement: do not touch existing code)

- ALL new code in include/pic/refresh.hpp: RefreshCfg (enabled=false
  default), device sampler, __global__ refresh_kernel, host wrapper,
  refresh.csv diagnostics (redraw count, injected energy).
- Touches to existing code: ONE guarded call in the step loop
  (if (refresh.enabled) apply_refresh(...)) AFTER push; ONE additive
  [refresh] parse block in deck.hpp (absent section -> disabled).
- push kernel, loader, deposit, boundaries, energy.csv format: zero edits.
  CTest 10/10 must pass untouched.

## Gates (in order; a failed gate stops the line)

- Gate 0 (bit-identity): new binary, feature compiled but disabled, same
  seed, x10 deck 5000 steps -> probe/bline outputs byte-identical to the
  current binary. compute-sanitizer clean.
- Gate 1 (null test): refresh ON during noise-level window: n(x), Tperp(x),
  Tpar(x) statistically unchanged (bath must be invisible when f = f0);
  refresh.csv net energy ~ 0; redraw count ~ analytic bounce flux through
  lambda_R.
- Gate 2 (geometry): short runs (t~4000) at lambda_R = 12/15/18 deg:
  early element times/amplitudes agree within noise.
- Gate 3 (main run): x10 refreshed vs closed, same seed, t_end 10000
  (extend 20000 to prove "endless"). CLAIM LIMITED TO: train does not die,
  period stationary (ACF peak does not drift with sliding window — the
  894->1500 stretch disappears), amplitudes steady. Gate 3 alone canNOT
  identify the clock (see below).
- Gate 4 (THE JUDGE — clock identification): x7 refreshed. At x10 the two
  clocks are DEGENERATE: t_ign = ln(B_th/floor)/gamma0 =
  ln(1.24e-3/5e-5)/2.6e-3 ~ 1230/Oe sits inside the T_b/2 band 720-1260.
  The lre lever splits them with OPPOSITE SIGNS: bounce clock ∝ lre
  (x7 -> x1.43 ~ 1430), ignition clock = ln(B_th/floor)/gamma with
  B_th ∝ lre^-4 (x7: th ~ 3.2e-4, ratio-to-floor 6 -> clean elements,
  clock -> x0.6 ~ 715). Sign of the period shift decides; both runs give
  publication-clean discrete elements. (x8 = conservative alternative:
  smaller split, cleaner.) NOTE: the closed-box x10->x5 data already
  shows the sign (period ROSE 3x when lre doubled while ignition clock
  predicts a DROP) — x7 refreshed is the clean version.
- Gate 5 (optional kill shot, pick one): timed seed injection right after
  an element (ignition-gated -> immediate retrigger + phase reset;
  refill-gated -> refuses until T_b/2), or the fuel-gauge diagnostic
  (equatorial velocity moments; at ignition instants B_w aligns on B_th
  if threshold-gated, A/resonant flux aligns on a critical level if
  refill-gated). Both keep elements pristine.
- REJECTED discriminators (user: wants clean elements): ppc/noise-floor
  scans (grainy spectrograms), x5-as-judge (complex/sub-riser mush;
  x5 refreshed demoted to optional robustness check).

## Analysis / outputs

- Money figure: sliding-window ACF period vs time, closed vs refreshed.
- Energy closure: refresh.csv injection rate vs wave radiated power in
  steady state.
- Re-run sweep-rate check in steady state (expect the same 0.7x Eq. 88).
- Cost: code + Gates 0-1 ~ half day; Gate 2 ~ 1.5 h; Gate 3 ~ 3.5-7 h;
  Gate 4 ~ 3.5 h. Land checkpoint (M7 pull-forward) FIRST.

## Related: Tao/Zonca/Chen 2025 letter (user will supply PDF)

"What drives chorus wave frequency chirping?", Phys. Plasmas 32, 100703
(2025): modified PIC with J_B feedback removed still chirps -> multi-wave
selective amplification (TaRA) vs Omura single-wave J_B picture.
Implications for us: (1) our sweep-rate 0.72-0.73x Eq.88 result does NOT
discriminate (both frameworks predict dw/dt ∝ Bw, similar coefficient) —
cite both, avoid J_B-causal language; (2) threshold/self-start/period/
gating results are mechanism-agnostic (B_th = hole-formation condition,
needed by both); (3) candidate new experiment: J_B ablation in ArcWarden
(project hot current onto wave-B, subtract in field update) — does
REPETITION + threshold scaling survive without J_B? Nobody has run the
ablation in a dipole with repetitive elements. No arXiv preprint found;
AIP blocks fetch — user will drop the PDF in docs/ for a deep read.
