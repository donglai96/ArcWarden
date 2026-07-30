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

## IMPLEMENTATION RECORD (2026-07-26) — landed, gates PASS

Code: include/pic/refresh.hpp (all logic), [refresh] deck block
(enable/lambda_deg/shell), RunParams::refresh/refresh_lambda/refresh_shell,
chirp2d wiring (RefreshState + refresh.csv: step,time,redraws,dE,fails).
simulation_maxwell.hpp UNTOUCHED (runner-level call after sim.step() —
stronger isolation than planned). Gate test: tests/test_refresh_null.cu
(ctest refresh_null; runs a zero-dynamics sampler probe + Gate 1 + Gate 2).

FINAL FORM (differs from the design sketch in three load-bearing ways —
each forced by a measured Gate-1 failure, all five variants kept in the
refresh.hpp header as methods material):
1. THIN SHELL s(15deg) < |s| < s(15deg)+15, not the whole |lambda|>15deg
   region. Wide bath + hybrid absorber churns +16x hot KE per 450/wpe
   (absorber re-drains every refreshed u_perp); wide bath +
   trapped-conditioning perp-piles the deep bath (closing trapped cone:
   Tpar -45% in ONE application) and pumps the whistler instability.
2. PROBABILITY GATE q = |u_par| dt / w per step => expected redraws per
   outward crossing EXACTLY 1, velocity-independent, still stateless.
   Redrawing every step while inside the shell (~250x per crossing)
   re-randomizes the shell current each step = white-noise whistler
   antenna: +38% hot KE per 450/wpe radiated even with reflecting walls.
3. FLUX-WEIGHTED u_par (|u| Rayleigh, random sign; u_perp stays the
   loader's density-weighted local (E,mu) form). The swap fires once per
   CROSSING (a flux event); the density-weighted Gaussian left a -6.8%
   shell-adjacent Tpar dip (classic boundary-injection result).
Also: refresh REJECTS bnd_x = 2 (hybrid) decks — the particle absorber
makes the closed steady state deviate from f0 and any f0-restoring bath
fights it (variants 1,2,4 above + thin-shell-unconditioned all fail on
hybrid). Gate-3 decks use x = damping (particles reflect specularly),
which is also closer to Chen 2026's own bare-reflection boundary.

GATE RESULTS (RTX 5090, chen2026-case2-like deck, 3000 steps, ppc 400):
- Sampler probe (zero dynamics, one application): every bin moment within
  +-0.4% (distribution-identity with the mirror loader confirmed).
- Gate 1 (bath ON vs OFF, same seed): interior n 0.18% / Tpar 0.69% /
  Tperp 0.16% (all shot-noise); net dE = -1.2e-6 = 3e-6 of hot KE;
  redraws 27/step = the bounce flux; 0 draw failures.
- Gate 2 (lambda_R 12/15/18): interior insensitive, worst 0.66%.
- Gate 0 (feature compiled+disabled, statistical non-regression,
  scripts/regress_case2.py): tiled PASS outright; flat PASS with pooled
  3+3-run envelope (WE 0.97x, WB 1.45x, bline 1.00x of same-build
  envelope; the 2-run WE/WB envelope is unreliable — energies are sparse
  scalars, always pool >=3 runs).

GATE 3 LAUNCHED (this session): decks/chen2026_case2_refl.ini (closed) vs
decks/chen2026_case2_refresh.ini (shell bath), both x=damping, same seed,
t_end 10000/We0, outputs build/gate3_closed + build/gate3_refresh.
Question: does the element train persist with stationary ACF period?

## GATE 3 RESULT (2026-07-26) — refill sustains the drive; discreteness
## lost to the cavity boundary (both arms), claim scoped accordingly

Runs: build/gate3_closed vs build/gate3_refresh (same seed, x=damping,
t_end 10000/We0, ppc 800). Figures: gate3_{closed,refresh}/gate3_*_final.png
(standard 4-panel), gate3_compare.png (A/B overlay).

VERDICT — the refill hypothesis holds on this base:
- WB(t): identical through the growth phase (ratio 1.0 at t=2000), then
  monotonic divergence as the closed box drains: 1.7x (4000), 2.1x (6000),
  2.6x (8000), 5.0x (9800/We0).
- Envelope: closed decays 0.0045 -> 0.0037 (-18%, peak at t=1765);
  refreshed flat 0.0043 -> 0.0044 with its RUN PEAK AT t=9528 — activity
  undiminished at end-of-run.
- ACF period of the envelope: closed stretches 436 -> 927/We0 (first-half
  vs second-half medians — the fuel-drain clock slowdown, same signature
  as the hybrid x10 894 -> ~1500); refreshed stationary 727 -> 748 (+3%).
- Supply: 55 redraws/step steady (vs 27 in the noise-window null test —
  the bath responds to wave-driven pitch-angle traffic, no knob); total
  injected energy 9.8e-3 = 2.4% of hot KE over the full run.

SCOPE CAVEAT (honest): with x=damping both arms produce CONTINUOUS
broadband whistler turbulence (dense counter-propagating packets in the
h-t maps), NOT the discrete element trains of the hybrid decks — the
field-only damping layers reflect too much (boundary_reflection.cu R~1;
the df7 lesson that reflectivity sets the timeline applies). So Gate 3 as
run demonstrates "refresh turns a draining box into a statistically
steady driven state" (limit-cycle throughput), but the design's original
"endless DISCRETE element train with period ~ T_b" claim remains open —
it needs a wave-absorbing, particle-clean boundary. Candidate next steps:
(a) reproduce df7's partial-reflection (R~10%) configuration full-f and
rerun both arms; (b) revisit a hot-current-aware field absorber that does
not touch f (so refresh stays compatible); (c) accept hybrid for the
closed arm only and compare refresh on the damping base against closed
on BOTH bases. Gate 4 (x7 lre clock split) unaffected — it can run on
whichever base recovers discreteness.

## PRECIPITATION MODE LANDED (2026-07-27) — refresh + real precipitation,
## smoke-tested, ready for the production A/B

User call after the Gate-3 giant closed-arm forensics (free energy never
tapped on the damping base; hybrid's absorber both sinks u_perp AND
enables wave absorption but is refresh-incompatible): build the complete
loss+refill loop as refresh + REAL precipitation.

Scheme ([refresh] precip = true, requires bnd_x = 1 + refresh enabled;
all in refresh.hpp, same kernel, velocity-only => charge continuity safe,
fixed N):
- WALL STRIPS (one cell at x = 0 / Lx): any marker there gets u_perp
  ZEROED, u_par kept; removed energy counted (refresh.csv dE_precip).
  Only true loss-cone markers (alpha_eq < asin(b_wall^-1/2) = 39.5 deg)
  ever reach the strips — the hybrid layers' 39.5-43.5 deg buffer band is
  untouched. Ghost keeps incident u_par => mu = 0, no mirror trapping,
  streams across the box in ~T_b/5, minimal density pile-up.
- RECYCLING: the ghost crosses the FAR shell outward and the ordinary
  bath rule swaps it for a fresh trapped-f0 sample — precipitation
  removes the energy at the wall, the bath re-injects it at the
  bounce-flux rate. Net budget = dE_injected - dE_precip, both logged.
- Counting: dE ledger takes every zeroing event; the precips counter
  only fires above u_perp^2 > 1e-6 (strip-dwelling ghosts re-accrete
  ~1e-8 of wave-noise u_perp between zeroings — 12x count inflation
  before the threshold, energy unaffected).

SMOKE RESULTS (ppc = 50 throwaway, decks/refresh_precip_smoke.ini):
- test_refresh_null still PASS (precip off = untouched path).
- compute-sanitizer memcheck: 0 errors.
- Loop closure (t to 3000/We0 = 2.3 ghost transits): injection rate
  ramps 1e-4 -> 1.9e-3 per 300/We0 interval while the initial loss-cone
  drain decays 5.8e-3 -> 2.3e-3; late-window injection/precipitation =
  0.82 and still converging — the pipeline fills over the transit time
  exactly as predicted. Late rates: 1.2 first-arrival precips/step,
  3.0 redraws/step.
- CAVEAT: ppc-50 noise floor is enormous (WB ~ 5e-4) => the smoke's
  precip flux is noise-scattering-driven; production numbers need giant
  scale. NOT run (user: smoke only).

Production next step (when user green-lights GPU): the three-arm giant
set on x = damping — closed / refresh / refresh+precip — answers both
"does refill sustain the drive" and "does precipitation restore the
element-forming free-energy cycling that hybrid provided", with the full
energy budget (bath in, wall out, wave field) closed in refresh.csv.

## DESIGN RULING (2026-07-27, user): the bath refills the SUBTRACTED
## (empty-cone) f0 — and that is the physical choice

Question: should redraws sample the loss-cone-subtracted f0 or the full
one? Ruling: SUBTRACTED. Drift-resupplied particles arrive from
neighboring flux tubes that also touch the atmosphere; tau_drift /
tau_bounce ~ 1e3, so the supply is pre-emptied — cone occupancy is set
by the LOCAL balance (wave scattering in vs precipitation out), never by
the supply. The implementation already complies (redraw inherits the
species' dist/lc_rho/lc_kappa verbatim). The R-test's cone-filled
isotropic bath is an unphysical drift model — mechanism-study use only.
COROLLARY: with subtracted refill + precip, refresh.csv dE_precip is a
pure WAVE-DRIVEN precipitation flux meter — chorus elements should pulse
it (the chorus <-> pulsating-aurora correspondence, free diagnostic for
the 3-arm run). Known small inconsistency, accepted: Chen's smooth
rho=1/kappa=0.3 subtraction vs the box's sharp 39.5-deg geometric cone —
the system relaxes to its own self-consistent cone-edge shape;
mirror-before-wall conditioning exists in git history if ever needed.
