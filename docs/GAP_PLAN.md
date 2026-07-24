# FLAGSHIP: chirping chorus + self-consistent 0.5 fce power gap in one box
(plan drafted 2026-07-23; nobody has ever shown both together — codes that
chirp (KO/DAWN/gcPIC/ours) never gap, the code that gapped (Li 2019 tilted
periodic box) never chirps)

## Why every existing attempt was structurally unable

- 1D parallel (all chirping codes incl. ours today): E_par = 0, Landau
  channel absent -> no plateau loop possible. KO2016 said it: oblique needed.
- Single hot bi-Maxwellian (everyone): upper-band resonant shelf
  (0.1-2.6 keV at wpe/We=5) has no anisotropic fuel; three suppression
  factors (resonant-shell measure ~ V_R f(V_R) -> 0, KP threshold
  A > w/(We-w) -> 4 at 0.8, dilution of a 10-20 keV Maxwellian at low v).
  No upper band -> no gap (a gap needs two shoulders).
- Short runs (USTC 2D): gap is CUMULATIVE (plateau builds over many
  elements); one-or-two-element runs can't carve it.
- Small latitude range: obliquity (the Landau pen) develops at lambda ~
  10-25 deg; domains ending at 15-20 deg absorb the wave before it writes.
- Li 2019: has the loop but imposed 15-deg tilt, uniform B, periodic
  recirculation, incoherent waves. Demonstrates the loop, not chorus.

## Physics hypothesis being tested (mine, falsifiable)

Plateau loop with hard kinematic anchor + directionality:
LB oblique E_par (Landau, co-moving, written at lambda ~ 10-25 deg) carves
plateau spanning [Vp(w_LB_min), Vp(0.5)]; Vp is EXTREMAL at 0.5 ->
plateau top edge universal; |V_R(0.5)| = Vp(0.5) degeneracy -> cyclotron
image of the plateau top lands exactly at 0.5; plateau (A<1) cyclotron-DAMPS
counter-propagating waves in (0.5, ~0.55). Soft wall (needs plateau ->
6% crossing events, Chen 2023 plateau correlation), hard anchor (always 0.5).
Northward LB walls the southward 0.5 band and vice versa -> gap requires
bidirectional equatorial source ("one-way street" test is unique to this
mechanism). Upper band is separately fueled by low-energy anisotropy
(continuous kappa tail, NOT a tuned Fu-style second component).

## G0 — Analytics first (paper+pencil + scripts, ~1 week, START NOW)

- G0.1 Scar transport map (THE missing calculation in the whole field):
  Landau scar written at lambda_w in local (vpar,vperp) -> (E,mu) transport
  to equator -> equatorial image vs |V_R(w)| curve. Predicts gap position
  shift + width from geometry. Deliverable: formula + figure. Reuses the
  mirror-load mapping math. Either fortifies or corrects the 0.5 anchor —
  both outcomes are paper sections.
- G0.2 Linear growth spectrum calculator for candidate f0 (bi-Max sum /
  kappa, anisotropy extending to 0.1 keV): confirm UB fuel exists AND LB
  growth (chirping recipe) unchanged; pick deck f0. Extends
  omura_threshold.py branch machinery.

## G1 — Infrastructure (each piece feature-isolated + gated)

- G1.1 Loader v2: kappa / multi-component / energy-stratified marker
  weights (statistics on the 0.1-2.6 keV shelf), (E,mu) mirror mapping for
  arbitrary f0. Shared prerequisite with M5b (loss-cone df0). Gate: n(x),
  T(x) match analytics; quiet-start noise unchanged for legacy decks.
- G1.2 2D mirror background: Bx = B0(1+a x^2), By = -B0 a x y
  (divergence-free to first order; Ke 2017 geometry) + 2D equilibrium load.
  Gate: single-particle (E,mu) conservation over many bounces; loaded
  plasma quiescent for >= 2 bounce times.
- G1.3 Diagnostics pack: f(vpar,vperp; lambda bins; t) moment dumps
  (plateau monitor), wave-normal-angle map, E_par field output, per-band
  power tracking (LB / gap 0.50-0.56 / UB), resonance-tagged J.E ledger
  (bin particle work by vpar near +Vp = Landau vs near -|V_R| = cyclotron
  — settles Li-vs-Chen-vs-Omura division of labor in one figure).
- G1.4 Checkpoint (M7 pull-forward) — long runs, mandatory. Do FIRST.
- NOTE boundary: CLOSED box (specular walls) is CORRECT here — reflection
  conserves (E,mu) so the plateau survives bounces and accumulates.
  Refresh bath would ERASE the scar every crossing (design tension with the
  infinite-train experiment logged in REFRESH_DESIGN.md — different decks,
  different papers, do not mix).

## G2 — Stepping stones (cheap, each independently reportable)

- G2.1 Li-2019 replication in tilted uniform B (rotate B0 by 15 deg in our
  2D code; their box = 1024 cells, 8M particles = MINUTES on the 5090):
  gate = band splitting + plateau with our kappa loader.
  BONUS: run the one-way-street pre-test here (their box is inherently
  bidirectional; k-filter one direction) — first discriminator data before
  any 2D-dipole cost.
- G2.2 2D mirror short run: verify oblique conversion vs latitude matches
  cold-plasma ray tracing; E_par appears where expected; no numerical
  surprises. Hours.

## G3 — Flagship run: chirping + gap in one box

Deck sketch: 2D mirror, lambda range +-27-30 deg (compression chosen via
our Omura-threshold calibration so the system sits in the CLEAN
threshold-gated repetitive regime — the lre-scan expertise directly reused:
gap needs MANY elements), wpe/We = 5, f0 from G0.2 (kappa-like, anisotropy
down to 0.1 keV), duration 20000-30000/We (>= 10-20 elements).

Size reality on one 5090 (32 GB — memory is the binding constraint, not
time): ny 256 (Ly ~ 65-130 c/wpe covers k_perp for theta <= 45-60 deg),
nx ~ 2500-3000 -> ~0.7M cells; full-f ppc ~1200-1500 -> ~1e9 markers
~ 26-30 GB. Noise floor ~ 4-5e-4 -> th/floor only ~2-3: marginal.
Fallback ladder if too dirty: (a) energy-stratified weights buy the shelf
statistics without global ppc; (b) delta-f hot + antenna-triggered element
trains (self-start not essential for gap physics — justify openly);
(c) shrink lambda range to +-25 deg. Runtime ~ 5e14 p-steps ~ 9 h. 
Checkpoint every 20k steps.

Success criteria (the money figures):
1. Risers stall at 0.5 (or at the G0.1-predicted shifted anchor) while UB
   power appears above ~0.55-0.6 -> first-ever chirping+gap coexistence.
2. Three-curve time-lagged causality: oblique LB power(t) -> plateau
   density(t) -> gap depth(t), in that order.
3. Gap BREATHES at the element repetition period (plateau written per
   element, eroded between) — connects the gap physics to our period
   physics; nobody has ever looked.
4. Resonance ledger: 0.5-band losses booked to counter-moving cyclotron
   electrons; LB oblique losses booked to co-moving Landau electrons.

## G4 — Causality experiments (one panel each; reuse masks/feature pattern)

1. ONE-WAY STREET (direction-selective k-mask on LB): surviving direction's
   0.5 band heals -> unique confirmation of counter-propagation closure.
   No other mechanism predicts direction sensitivity.
2. Plateau surgery: pre-seed plateau at t=0 (gap appears immediately) /
   relax plateau away mid-run (gap heals). Sufficiency + necessity.
3. Kill-the-parent: spectral damping of LB in flight -> plateau supply
   stops -> gap heals; also kills cascade hypothesis if UB survives.
4. 1D control: same f0, ny=1 -> no gap expected (Landau channel absent).
5. Width-scaling scan: wpe/We in {3, 5, 8} -> gap WIDTH vs the G0.1/VR-map
   formula (position pins at ~0.5 for everyone; WIDTH discriminates).

## G5 — Paper + observational hooks (UCLA collab)

Title shape: "Rising-tone chorus digs its own spectral gap". Sim core +
two data hooks for the Bortnik/Li/An group next door: (a) UB-LB element-
level cross-correlation statistics (cascade -> phase-locked 2x chirp;
two-source -> independent; plateau -> shared source, independent timing);
(b) phase continuity ACROSS the gap in the 6% crossing events (wall =
amplitude filter vs process killer) — our phase-forensics toolkit applies.

## Risk register

- Memory ceiling -> fallback ladder above; worst case the flagship runs
  delta-f + antenna (state openly in paper).
- Gap does not form -> negative result WITH the G0.1 analytics saying
  where it should have been = still the first serious 2D search, and the
  1D/2D + f0 controls say WHY. Publishable either way (KO2016 precedent).
- Plateau erased by boundary: closed box preserves it (specular, E,mu
  conserving) — verified design choice, see note in G1.
- UB fuel under-resolved -> stratified weights (G1.1) before bigger ppc.

## Order of battle (what starts today)

1. G0.1 scar-transport analytics (pure math + existing mapping code)
2. G1.4 checkpoint landing (also unblocks M5b)
3. G1.1 loader v2 (shared with M5b)
then G2.1 (minutes-scale Li replication) as the first physics smoke test.

---
# EXECUTION LOG

## G0 — COMPLETE (2026-07-22 evening)

G0.1 (scripts/gap_scar_transport.py, docs/figs/gap_scar_transport.png):
- Degeneracy verified exactly: Vp_eq(0.5) = |VR_eq(0.5)| = 0.0995c.
- THE ANCHOR SURVIVES TRANSPORT, and we now know why: wall sharpness
  ~ 1/scar-spread; spread -> 0 as lambda_w -> 0 (1.6% at 2 deg, 49% at
  15 deg). Near-equator writing pins a sharp wall at 0.50-0.53; high-
  latitude writing only adds a diluted shoulder BELOW 0.5 (walls sliding
  to 0.35-0.43 but smeared thin).
- CORRECTED my own guess on asymmetry: assembled erosion profile =
  graded LB decay toward 0.5 (transported-shoulder) + deep notch
  0.50-0.53 + clean above ~0.55. Compare to VAP spectra.
- Table: lam_w = 2/5/10/15/20/25 deg -> ridge wall 0.499/0.492/0.468/
  0.433/0.389/0.340; spread/Vp(0.5) = 0.016/0.09/0.29/0.49/0.64/0.73.

G0.2 (scripts/whistler_growth.py, docs/figs/gap_growth_design.png):
- Solver gated: cold limit 1e-17; KP marginal exact (0.342); Case II
  bi-Max gamma_max 1.67e-3 vs loss-cone giant-run measured 2.6e-3
  (loss-cone enhancement, consistent).
- FINDING: warm-component route is expensive — the 10-keV core actively
  DAMPS the UB (-1.7e-4), so warm fuel needs n>=0.010 AND R>=4 (Li 2019's
  R=4 fit now makes sense). 
- FINDING: single-component sweet spot exists: R=2.5-3.0, nh=0.005-0.007
  -> LB peak 2.3-3.2e-3 (discrete-regime drive) AND UB growth positive
  (+2-3.6e-4 out to w~0.65). Linear spectrum is GAPLESS -> perfect for
  the emergence experiment (any gap must be dynamically carved).
- Flagship f0 candidate: single loss-cone/bi-Max R=2.5-3, nh~0.005-0.007
  (calibrate compression via omura_threshold.py for clean risers).

## G1.4 checkpoint — LANDED (2026-07-22 evening)

- include/pic/checkpoint_io.hpp (all new): M0 schema + payload streaming,
  64M-element chunks, atomic .tmp+rename; manifest built from what is
  allocated (full-f/delta-f/wprec all round-trip).
- Touches: simulation_maxwell.hpp +2 additive accessors (step_count/
  set_step_count; restore forces tile re-sort); chirp2d.cu flags
  --ckpt=N / --resume (probe.bin truncate+append, energy.csv append,
  bnoise/wdnoise skipped on resume).
- GATE PASSED: code is run-to-run nondeterministic (atomic deposit), so
  the gate is statistical: resume-vs-straight rms diff 2.96e-8 <=
  straight-vs-straight 3.12e-8 (zero added error); WB continuous to 5+
  digits across the resume; probe record count exact. CTest 32/32.
- NOTE: choose --ckpt multiple of tile_sort (25) to keep the sort
  schedule aligned across resume.

## NEXT (in order)
- G1.1 loader v2: kappa + multi-component + energy-stratified weights +
  (E,mu) mapping for arbitrary f0 (read initialize_mirror carefully
  first; shared prerequisite with M5b delta-f loss-cone df0).
- G1.2 2D mirror B0 (divergence-free) + 2D equilibrium load + gates.
- G1.3 diagnostics pack (plateau monitor, WNA map, E_par, band power,
  resonance-tagged J.E ledger).
- G2.1 Li 2019 replication (tilted uniform B0 — needs the B0-direction
  feature from G1.2 work + kappa loader) + one-way-street pretest.
- G2.2 2D mirror short run vs ray tracing.

## G1.1(partial) + G2.1(first run) — 2026-07-22 late night

LANDED:
- kappa_v bi-kappa loader (Species/deck/species_init_kernel): Gaussian *
  sqrt(kappa/W), W ~ chi2_{2k-1} shared across components; integer-nu
  validation + noisy-only guard. GATE PASS: f(vpar) core max rel err 2.6%
  vs Student-t analytic; kappa=1.5 fat tail at 0.10c is 4000x Maxwellian.
- tilted B0 was ALREADY there ([background] theta_deg, x-z plane; push uses
  full B0[3]; M10 case-7 validated) — zero work needed.
- tools/liband_yee.cu (new, +CMake target): all-PIC multi-species runner,
  uniform tilted B0, periodic, dumps bline(By,Bz,Epar)/probe/fhist(vpar
  plateau monitor)/energy. decks/li2019_band.ini (their 1024 cells, 8.4M
  markers, cold 80% k=4 10eV + warm 20% k=1.5 R=4, theta=15deg).
- scripts/plot_li2019_band.py (4-panel gate fig) + k-w analysis.

FIRST RUN (500k steps = 2000/We, ~7 min wall):
- Broadband whistler growth ✓ (saturates by t~700/We, delta-B rms ~ 0.13
  B0), parallel heating of the core visible in Delta-f map ✓.
- TWO-BAND SPLIT NOT YET: late-time gap-window/LB power = 0.97 (no gap).
  docs/figs/li2019_band_gate.png + li2019_band_kw.png.

OPEN ITEMS for the G2.1 gate (next session):
1. LONGER RUN: 2M steps = 8000/We (~25 min) — splitting may be slower than
   2000/We; watch gap/LB ratio vs t.
2. KAPPA=1.5 TAIL GUARD: nu=2 Student-t has INFINITE variance — rare
   superluminal draws break the nonrel push. Add umax truncation
   (redraw if |u| > ~0.7c) to the sampler.
3. CONVENTION CHECK: Li's kappa=1.5 sits exactly at the <v^2> divergence
   of the standard kappa pdf — their definition may be the "modified
   kappa" (theta = true thermal speed, (kappa-3/2) inside). Verify against
   their Methods before tuning further.
4. Plateau localization: current Delta-f shows BULK heating, not a clean
   plateau at [0.08, 0.10]c — could be overdrive (R=4, 20%) or display
   scale; check band-limited early window.

## G2.1 correction (2026-07-22, after user caught the wrong-looking run)

- FIRST DIAGNOSIS (grid heating) REFUTED by data: cold-core sigma grew
  only 13% over the whole run (0.0126 -> 0.0142c). Numerics are fine.
- REAL BUGS, found by reading the actual Li 2019 PDF (/tmp/li2019.txt;
  Table 2 confirms our deck params EXACTLY):
  1. TIME UNITS: their clock is tau_gyro = 2pi/We. The gap forms at
     T = 800 tau_gyro = 5030/We. Our 500k-step run = 318 tau_gyro — we
     stopped between their Fig 4c (parallel acceleration onset, 150 tau_g)
     and 4e/f (two-band + gap, 800 tau_g). The killed 2M-step run was the
     right length all along.
  2. DISPLAY: their spectra are k-omega FFTs over fixed windows
     (50-100 tau_g one-band 0.2-0.8 with UB growing FASTER — our early
     broadband actually matches their Stage 1); my probe STFT during
     exponential growth has spectral leakage below 0.2 (fake power).
     scripts/plot_li2019_kw.py now reproduces their Fig-4 format
     (k mode number vs omega/We, their windows).
  3. their dt = 0.032/fpe (we run finer, fine).
- Relaunched 1.6M steps = 1018 tau_gyro with the kappa tail guard (0.6c
  cap) in. Gate = late-window (700-800 tau_g) gap/LB << 1 with UB present.

## G2.1 twin-arm design (2026-07-22, user's call: Darwin-vs-Yee control)

User judgement: grid heating IS the concern for the delicate gap physics
even at the measured slow rate (dx/lambda_D_cold = 5 on the Yee arm), and
Li's code is UPIC-lineage Darwin — so run BOTH our arms on the identical
deck: (a) Yee/Esirkepov momentum-conserving (li2019_band, 1.6M steps
dt=0.02), (b) SPECTRAL DARWIN twin (same UPIC lineage as Li's actual code:
tools/liband_darwin.cu, decks/li2019_band_darwin.ini, dt=0.2 = Li's
0.032/fpe, ndc=2 smoothing, 157k steps). Identical output formats — the
same plot scripts run on both. This doubles as a methods-paper section
(Yee-vs-Darwin at marginal Debye resolution) and the definitive
grid-heating control: compare cold-core sigma(t), dB2(t), and the
late-window (700-800 tau_g) k-w gap on both arms.

## G2.1 GATE PASSED on the Yee arm (2026-07-22 night)

TRUE CULPRIT of the "wrong" first run: UNBOUNDED kappa=1.5 sampling
(nu=2 Student-t, P(|v|>0.6c)~1e-3) -> thousands of superluminal markers
poisoning the deposit from step 0. Fixed by the 0.6c tail cap; NOT grid
heating (cold core grew 13% only), NOT overdrive. Evidence: capped rerun
starts at WB 3e-6 vs poisoned run's 1.3e-3 at comparable time.

Capped 1.6M-step Yee run (1018 tau_g) REPRODUCES Li 2019 Fig 4 end to end
(docs/figs/li2019_band_yee_kw.png):
- Stage 1: one continuous whistler band, both directions, growth FASTEST
  near 0.5 (Vp extremal there -> strongest Landau self-digging; the anchor
  argument playing out dynamically);
- plateau rises ~10x inside the Vp band [0.08,0.10]c, core stays narrow;
- gap carves open from ~300 tau_g, deepens monotonically THREE DECADES:
  gap/LB = 1.07 (200-250) -> 0.31 -> 0.095 -> 0.048 -> 0.019 -> 0.006 ->
  0.002 (800-850) -> 3e-4 (900-950); gap/UB 0.007 late (UB real, distinct);
- saturation dB_rms/B0 = 1.9% (the expected percent level).
Analysis: scripts/plot_li2019_kw.py (Li Fig-4 format) + inline gap-vs-time.
Darwin twin (liband_darwin, same UPIC lineage as Li's code) in flight for
the numerics-independence control; compare cold-core sigma(t), gap(t).

Flagship implication: the plateau->gap engine WORKS in our Yee/Esirkepov
stack at Li's own parameters. Remaining distance to the flagship is
geometry (2D mirror, G1.2) + coherence (dipole chirping regime), not the
kinetic engine.

## G2.1 twin-arm verdict (2026-07-22 night) — INVERTED from expectation

- DARWIN arm: cold core +23%/1000 tau_g (grid heating ACTIVE), instability
  barely grows (dB/B0 = 0.12%), no gap. Root cause: our Darwin uses the
  SAME momentum-conserving CIC deposit as Yee — it is NOT the
  energy-conserving UPIC scheme Li used; "Darwin is immune" does not hold
  for our implementation. Wave-growth suppression itself needs a separate
  look (ES heating noise vs self-excited-whistler solver fidelity — the
  an2019 validation was DRIVEN pump, not free growth). Methods-paper
  material; NOT a flagship blocker.
- YEE arm: cold core -0.7% (NO heating — Esirkepov charge-conserving J +
  jfilter=3 suppress the aliasing channel), full Li replication (see
  previous entry).
- DECISION: flagship runs on the Yee stack, now doubly validated (kinetic
  engine + numerical health). User's twin-control proposal delivered the
  decisive evidence.

## Session close 2026-07-22/23 (user paused before the Darwin gap run)

- Darwin full-length gap run (dt=0.05, 628k steps, ~50 min) KILLED by user
  before completion; restart recipe: ./liband_darwin
  decks/li2019_band_darwin_dt005.ini li2019_band_darwin_dt005 --nsteps=628400
  (cadences now dt-scaled in the tool). Optional cross-confirmation only —
  the Yee replication + gamma(dt) convergence already close G2.1.
- dt-scan run data preserved: build/li2019_band_darwin_dt01_200tg,
  build/li2019_band_darwin_dt005_200tg (the O(dt) verdict data).
- Li 2019 Methods verification (user-supplied PDFs in docs/): their Eq. 2
  kappa = EXACTLY our sampling convention; their loader is a DETERMINISTIC
  90x90 geometric velocity-grid quadrature load up to 0.99c (they also
  carry ~4000 markers >0.5c in a nonrel code — our 0.6c cap is the same
  regularization, more conservative); dx NOT published anywhere (main or
  supplementary — supplementary is purely observational); our dx=0.05
  assumption stands on the k-mode-number range consistency with their
  Fig 4. Optional future: implement their quadrature load as a loader
  option (better fat-tail statistics; kin to stratified weights).

## G1.2 EXECUTION (2026-07-23) — 2D slab mirror background LANDED

- Implementation (feature-isolated behind b0_prof = 3 "mirror2d"; prof
  0/1/2 paths bit-identical): flux function Az = B0eq*(y-yc)*(1+a(x-xc)^2)
  gives Bx = B0eq(1+a x~^2), By = -2a B0eq x~ y~ — div B = 0 EXACTLY.
  NOTE the factor 2 vs the plan's cylindrical "-a B0 x y" guess: the slab
  solenoidality requires it AND it makes the resolved y-gyration supply the
  FULL gyro-averaged mirror force -mu dBx/dx analytically (Fx = -q vz By;
  the z half-orbit contributes nothing since Bz_bg = 0). No effective-field
  trick in the pusher for prof 3 — the real geometry does the work.
- Files: background_b0.hpp (b0y2d + header math), yee2d.hpp (prof-3 branch
  in yee_advance_particle), config.hpp (b0_yc), deck.hpp (profile=mirror2d,
  yc key, ny>1 validation), particles.hpp (initialize_mirror ny>1:
  uniform-in-y — |B| is x-only to paraxial order; int32 offset guard),
  tests/test_mirror2d.cu + CMake. Delta-f Tperp(x) drive and (E,mu) load
  formulas consume b(x) = b0x/B0eq and work for prof 3 UNCHANGED.
- Gate A (single particle) PASS, and it is the strongest orbit gate in the
  code: wb vs sqrt(a)*uperp/gamma to 1e-6 (nonrel) / 4e-6 (rel), mu spread
  6.6e-4 / 1.1e-3 vs 1% gate, |u|^2 drift < 1e-5. Residual mu ripple is the
  PHYSICAL first-order adiabatic oscillation O(2 a rho x~max) — rel case
  uses a = 2.5e-4 to keep it under the gate at rho = 0.53.
- Gate B (loaded plasma) exposed TWO FLAGSHIP DESIGN RULES (both with n(x)
  holding at <0.2% — velocity-space effects only):
  1. rho_thermal/dy >= ~2 MANDATORY: at rho/dy = 1 the magnetized
     finite-grid instability pumped Tperp x4.6 in 2 bounce periods
     (saturating at rho ~ 2 dy, textbook Birdsall-Langdon). Invisible in
     all our 1D mirror work (gyration never on the grid). Li-like decks
     (rho_warm/dy = 4) are safe; check EVERY species incl. cold.
  2. (E,mu) load is a VACUUM-FIELD equilibrium — omits diamagnetic
     dB/B ~ -beta_perp/2. Full-density warm load (beta_perp = 0.32)
     relaxed at the 16%-level budget: T drifts 18-35% per 2 T_b at
     ppc 400 (part shot-noise ~1/ppc: x4 ppc gave 51->18%). RULE:
     mirror-load only MINORITY hot species (beta_hot << 1); cold majority
     isotropic (mapping-neutral). Gate B now runs at density 0.1
     (beta_perp = 0.032) with 5% T-drift gates.
- y-boundary caveat (documented in background_b0.hpp): By is linear in y~,
  so it jumps by 2 a B0 x~ Ly across the periodic y-wrap; keep paraxial
  ordering (R * Ly/x~max << 1). Only gyro-orbits straddling the wrap see
  it (magnetic-only — cannot heat; pitch-scatters a boundary layer
  2*rho/Ly of markers). Closed-x (bnd_x = 1) also keeps passing particles
  off the x-wrap where By flips sign — production config, used by gate B.
- G1.2 GATE CLOSED (PASS): gate A wb err 1e-6/4e-6, mu spread 6.6e-4/1.1e-3;
  gate B (density 0.1, ppc 400, jfilter 3, bnd_x 1, 2 T_b = 129k steps):
  <dB2> half-ratio 0.73 (decaying), n(x) 0.07%, INTERIOR dTpar -4.2%
  dTperp +1.2% (< 5% gates). jfilter=3 is part of the production recipe
  (turned <dB2> growth 1.36 -> 0.73; T drift barely moved — see rule 3).
  DESIGN RULE 3 (measured): the periodic y-wrap By sign flip pitch-scatters
  straddling gyro-orbits — layer (2rho/Ly = 25% here) isotropized dTpar
  -10.5% dTperp +5.6% per 2 T_b at CONSTANT |u| (elastic, not heating);
  interior echo -4%/+1% via passing-particle gyro-center migration.
  Scales (2 a B0 x~ Ly)^2 * (rho/Ly). Escalation if ever needed:
  cos-modulated periodic variant Bx = B0(1 + a x~^2 cos(2pi y~/Ly))
  (smooth, anti-mirror strip at wrap) or specular y-walls.
  Legacy prof 1/2 regression: yee_mirror_bounce, dipole_profile,
  deltaf_mirror_hold all PASS (paths bit-identical by construction).

## G1.3 EXECUTION (2026-07-23) — diagnostics pack LANDED (smoke-verified)

- include/pic/gap_diags.hpp (ADDITIVE-ONLY: no simulation-loop changes):
  1. FvHist f(vpar,vperp | x-region) w-weighted device histograms — the 2D
     plateau/scar monitor, vpar w.r.t. LOCAL b-hat (mirror2d aware);
  2. WorkLedger resonance-tagged J.E: per-marker staggered-gather E, bins
     q w E_par v_par (Landau ch) and q w E_perp.v_perp (cyclotron ch) by
     (x-region, vpar), accumulate-and-reset windows — the
     Li-vs-Chen-vs-Omura division-of-labor figure;
  3. full-2D 6-component field snapshots (tool-side) -> offline WNA map +
     E_par(x,y).
- particles.hpp: initialize_mirror(SpeciesList) — multi-species (E,mu)
  loader; isotropic species degrade EXACTLY to uniform (mapping cancels),
  so cold-core + hot-minority decks compose. Single-species path
  bit-identical (absolute-index RNG streams; regression tests pass).
- tools/mirror2d.cu: the G2.2/G3 runner — bline (axis row, Epar = Ex
  on-axis), probes, fv/wl/f2d cadenced in PHYSICAL time, energy.csv,
  meta.txt; validates design rules at startup (rho/dy >= 2 warning,
  beta_perp warning for anisotropic species). decks/mirror2d_smoke.ini.
- scripts/plot_mirror2d_diags.py: 4-panel quicklook (fv log + scar Delta-f
  + equator ledger Landau-vs-cyclotron + WNA k-power map).
- SMOKE VERIFIED (4000 steps, 9.8M markers): fv captures 98.4% weight
  (rest = >0.6c tail), region marginals reproduce the warm-species (E,mu)
  wall dip (nfac 0.676 at R=2) on top of the uniform cold — the
  multi-species mapping is correct through the tool path; ledger signs
  correct (particles feed the growing noise fields); WNA map renders the
  discrete mode lattice. Figure: docs/figs/mirror2d_diags_smoke.png.
- NEXT: G2.2 — physics deck (wce down or dy up so BOTH species pass
  rho/dy >= 2 without waiver; whistler-unstable warm minority), verify
  oblique conversion vs latitude against cold-plasma ray tracing + E_par
  appearance; then G3 flagship deck design (G0.2 sweet-spot f0).

## G2.2 PREP (2026-07-23) — ray-tracing reference DONE, pre-registered

- scripts/ray_mirror2d.py: QL whistler ray tracing in the exact b0_prof=3
  geometry (w conserved to 1e-8). RESULT: the paraxial slab BARELY twists
  rays — theta <= 0.6 deg at |x~| = 50 even launched at y~ = Ly/4 (the
  Ly << Lx compression kills field-line curvature; rays follow field
  lines). Figure: docs/figs/ray_mirror2d.png.
- PHYSICS IMPLICATION (flagship): oblique power in the slab must come from
  the anisotropy instability's own theta-spectrum (2D box carries ALL ky
  modes — MORE physical than Li's single imposed 15-deg tilt), not from
  propagation. Dipole-strength WNA growth is NOT reproducible in a
  paraxial slab — accepted: our gap hypothesis (scar transport + Vp/VR
  degeneracy) lives in parallel propagation + (E,mu), and Li got the gap
  in uniform-B 1D.
- PRE-REGISTERED G2.2 GATE: linear-stage WNA map (f2d snapshots) must stay
  near-parallel per the ray curves (no anomalous oblique conversion);
  E_par at the theta-consistent thermal level; instability-generated
  theta-spectrum appears at saturation, not before.
- decks/mirror2d_g22.ini: wce=0.25 (wpe/We=4), R_wall=2, cold 0.94
  uth=0.04 (rho/dy=2) + warm 0.06 A=3 (uth 0.04/0.08, rho/dy=4),
  dx=dy=0.08, 2560x80, dt=0.05, jfilter 3, closed x. ~2e8 markers,
  ~30 min/160k steps on the 5090. Launch AFTER the Darwin run frees the
  GPU.

## Darwin cross-confirmation CLOSED (2026-07-23, full-length dt=0.05 run)

- build/li2019_band_darwin_dt005 (628k steps = 1000 tau_g, completed):
  NO gap, NO plateau, NO band splitting — waves stall at dB/B0 ~ 1.5e-3
  (Yee: 1.9e-2), f(vpar) unchanged through 800 tau_g, k-w windows show
  only the noise-lit whistler branch. Figure:
  docs/figs/li2019_band_darwin_dt005_kw.png (gap/LB metric meaningless
  at noise level).
- QUANTITATIVE match to the O(dt) damping model: residual numerical
  damping ~0.7e-3 at dt=0.05 cuts gamma 2.6e-3 -> 1.9e-3, a ~4-e-fold
  deficit over 1000 tau_g = e^4 ~ 55x amplitude — as observed. The system
  sits marginally below the plateau-carving threshold and quasi-stalls.
- VERDICT (final): our spectral Darwin cannot do free-growth gap physics
  at practical dt (dt <= 0.02 would still be ~12% off and ~4h/run).
  Flagship runs on Yee — the twin-arm story is now complete as a methods
  result: Yee reproduces Li 2019 end-to-end; Darwin's O(dt) free-mode
  damping suppresses the same physics, with the deficit predicted by the
  dt-scan gamma measurements.

## G2.2 run 1 (2026-07-23) — NULL by design error; DESIGN RULE 4

- build/mirror2d_g22 (150M markers, 320k steps = 636 tau_c) came out QUIET:
  WB flat at the noise floor, WE slowly decaying, WNA map = noise lattice
  (consistent with the ray-tracing pre-registration: no anomalous oblique
  conversion), Delta-f only a slow ~6% cold-core low-v restructuring.
  The whistler instability NEVER GREW — my deck error, and the lesson is
  a keeper:
- DESIGN RULE 4 (resonant-tail budget): growth needs V_R/v_par_th <~ 2.
  Here V_R(0.5 We) = 0.125c vs warm v_par_th = 0.04c -> resonant fraction
  e^(-4.9): gamma exponentially dead. THIS is why Li 2019 use kappa = 1.5
  (fat tail feeds the resonant shell at cold-ish bulk temperatures) — the
  flagship f0 must pass an explicit V_R/v_th (or kappa-tail) check per
  band, tied to the G0.2 sweet-spot scan.
- Silver lining: an accidental 150M-marker, 636-tau_c quiescence
  validation of the whole mirror2d + diagnostics stack at scale. Figure:
  docs/figs/mirror2d_g22_diags.png.
- G2.2 run 2 deck fix: warm uth 0.1/0.2/0.2 (V_R/v_th = 1.25, rho/dy = 10)
  density 0.03 (beta_perp = 0.038, rule 2 OK) -> decks/mirror2d_g22.ini
  updated in place (run-1 deck preserved in the run dir's deck_snapshot).

## Darwin branch REDEEMED (2026-07-23, darwin_tc)

- Root cause found vs UPIC-2.0 (docs/DARWIN_UPIC_COMPARISON.md): UPIC
  time-centers ALL Darwin sources via a trial-Boris deposit; our legacy
  scheme deposited raw v(t-dt/2) -> O(dt) free-mode damping. Ported behind
  [field] tc = true (legacy bit-identical off).
- GATES: gamma(m+-7) dt-independent (0.2 vs 0.1 coincide) and matches the
  Yee arm same-estimator; full-length tc run at dt = 0.2 reproduces
  two-band + gap (gap/LB 6.8e-4). The Darwin branch is now a USABLE second
  engine (3x/step cost, 10x coarser dt than Yee) — cross-confirmation of
  flagship results is back on the menu, and the methods-paper section
  writes itself (task #13).
