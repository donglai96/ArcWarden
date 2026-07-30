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

## Lu-group 2D gcPIC audit (2026-07-23, verified from PDFs in docs/)

- Ke 2017 JGR mirror field = OUR EXACT prof=3 FORMULA (their Eq.1:
  B0x = -2 xi x z B0eq, B0z = (1+xi z^2)B0eq — incl. the factor 2; field
  lines x0/(1+xi z^2)). Their hot off-equator profile (Chan 1994 force
  balance) = ALGEBRAICALLY IDENTICAL to our (E,mu) mapping. Geometry and
  loader independently validated by the field standard.
- COLD = MASSLESS PRESSURELESS FLUID in ALL their 2D runs (KO2007
  electron-hybrid closure) — the user's "back to fluid" call is the field
  standard. wpe/We = 5 everywhere.
- Boundaries: absorbing waves (Tao 2014 masks) + specular particles = our
  bnd_x scheme. Full-f, ppc 500-1000 (hot only!), NO smoothing documented.
- THE difference vs our G2.2: SCALE + ASPECT. Ke 2017 box ~ +-495 x +-354
  c/wpe (near-square; field-line tilt to ~35 deg IN the box -> real
  curvature, WNA grows to 25-50 deg); our ribbon 205 x 6.4 (max tilt 3.5
  deg). Same formula, different transverse extent = the curvature knob.
  Their gamma = 1.4e-2 We = 2.8e-3 wpe (same as our linear estimate!) but
  runway L ~ 700 c/wpe -> single-pass gain ~7 e-folds: convective design
  that WORKS with absorbing walls. Onset We t ~ 400, elements at
  amplitudes ~1e-2 B0, chirping in 3-25 deg latitude, 1-2 elements per
  run; repetition needs injection (Lu 2021 1D delta-f, tau_D ~ 2000/We).
- Hot: nh = 1-2%, Tperp/Tpar = 4.5-6, beta_par = 0.01, wpar ~ 0.14c,
  RELATIVISTIC push. A threshold for rising tones drops 1D->2D (7 -> 3.8).
- NO GAP, NO UPPER BAND anywhere in the series (lower band only; Ke 2025:
  upper band would need finer grids). FLAGSHIP NICHE CONFIRMED OPEN.
- G2.2 REFRAME: make it a Ke-2017-style run on ArcWarden — fluid cold
  (cold_model=full upgrade: vcx + symmetric stagger + local-bhat rotation),
  wide box (aspect toward theirs; fluid cold removes the cold-Debye grid
  constraint so dx can grow), hot nh~0.01 A=4.5-6 wpar 0.14c rel=1.
  Known target figures = clean gates; flagship then = + kinetic delta-f
  cold (Landau channel for the gap, which THEY cannot do with T=0 fluid).

## G2.2 v2 BUILD (2026-07-23 evening) — cold_model=full LANDED+GATED; Ke17 deck staged

- cold_model=full ([field] key; legacy transverse path bit-identical off):
  3-component cold fluid at NODES — vcx restores the longitudinal cold
  response (E_par shielding for oblique waves); SYMMETRIC staggered
  E-gather/J-scatter (real transfer cos(kd/2): zero phase error);
  EXACT Rodrigues rotation about the LOCAL analytic b-hat (mirror2d aware,
  reduces to the legacy sincos for B || x). Files: config/deck/yee2d
  (k_cold_fluid_full + k_cold_current_full + mask vcx)/simulation_maxwell
  (vcx_ alloc + dispatch)/checkpoint_io (vcx entry)/mirror2d (unlock).
- GATE test_cold_fluid_oblique (2-level convergence): fine level omega
  errors 0.003-0.46% vs exact Stix cold dispersion across theta=0-63 deg;
  coarse-level 0.35-1.4% residual converges (grid dispersion, documented);
  energy drift -0.04%/-0.10% over T=2000 -> ZERO numerical damping.
  Estimator lessons burned in: frequency-bin quantization (parabolic peak
  interp) + elliptic-polarization mirror-line sidelobes (Hann window).
  Legacy test_cold_fluid_dispersion untouched, passes.
- decks/mirror2d_ke17.ini staged (NOT run — user hold on long runs):
  Ke 2017 Table-1 faithful (xi=6.94e-6, dx=0.345/dy=0.966, wpe/We=5,
  cold_nc=0.99 fluid, hot 1% A=4.5 wpar=0.141c rel, dt=0.1, 150k steps
  = 3000/We; deviations documented in deck header: half transverse width,
  periodic y, jfilter=3). 300-step smoke: 275M markers, ~21 GB, clean.
- Targets when launched: onset We*t~400, elements ~1e-2 B0, chirping
  Gamma=3.5-7.7e-4 We^2, WNA < 25 deg (Ke Figs. 3-4, 8).

## RSM pivot (2026-07-24, user-directed): bridge model = new minimal flagship arm

- User insight chain: (a) 2D expensive -> add a few k_perp to 1D; (b) user
  correction: Li 2019 is OBLIQUE (15-deg tilted B0) — k_perp is the gap's
  admission ticket, NOT a robustness check (my "1D parallel repro" claim
  was wrong; the plan header itself says E_par=0 -> no loop); (c) System
  Design Document commissioned and written.
- docs/RSM_MODEL_DEFINITION.md v1.0 = PROJECT CONSTITUTION (scope +
  boxed hypothesis, retained/neglected physics, why {0,+-k1} only,
  controlled-not-cheap, Li/Lu bridge table, math structure, full-orbit
  rationale, success Levels 0-4, failure Cases A-D, model hierarchy).
  Key refinements over the outline: refraction split (along-field WNA
  evolution RETAINED via kx at fixed k_perp; transverse ray bending
  neglected); Level-3 gap defined as deepening LOCAL MINIMUM with finite
  upper shoulder (gap != cutoff); Levels 0/1 gate against IN-HOUSE
  reproductions (Chen 2026 chirp; Li 2019 tilted gap), not literature.
- Implementation audit (source-verified): particles already carry (x,y)
  + full 3V (particles.hpp Particles struct) with 2D Esirkepov deposit
  -> no eta_p phase variable needed; cold_model=full IS the oblique
  field response (gated 0.003-0.46%, zero damping); ONLY new code =
  per-step ky projection of J keeping m in {0,+-1} (+ initial-seed
  masking) + deck flag. Feature-isolated as always.
- Mirror choice for clean truncation: parametrized mc-term (b0_prof 1/2,
  y-homogeneous background -> ALL linear ops diagonal in m, truncation
  exact) FIRST; prof=3 resolved slab leaks m->m+-1 through B0y~y (weak
  at small Ly) and is the rung toward full 2D. TODO when implementing:
  verify the mc term is applied for ny>1 (chirp2d enforces ny==1 today).
- Benchmark ladder (maps to constitution Levels): (i) uniform-B single-
  oblique-mode KINETIC benchmark: omega(k) shift, E_par/E_perp vs Stix,
  Landau Vp, Li-type mid-energy reshaping (cold half already gated);
  (ii) Li-type plateau in RSM geometry = Level 1 (~1000 tau_g run);
  (iii) both channels on = Level 2; (iv) mirror chirp+gap = Level 3.
- Ke17 full-2D (decks/mirror2d_ke17.ini, staged+smoked) demoted from
  "next launch" to literature-anchor arm (constitution Section 10).

## RSM non-regression gate on the Chen 2026 case-2 path (2026-07-24)

- User requirement: RSM code must not affect the chirp2d path that
  reproduces Chen 2026 case 2.
- Bit-identity is IMPOSSIBLE at run level: two runs of the SAME binary
  on chen2026_case2.ini differ from step 1 (GPU float atomicAdd
  ordering); measured same-build envelope at 5000 steps: WE 4.2e-6,
  WB 1.3e-5, bline rms 3.6e-4, probe rms 1.9e-4.
- Gate = STATISTICAL EQUIVALENCE (scripts/regress_case2.py): candidate
  envelope <= 3x same-build envelope, three short runs (~s each).
- EXECUTED for the evening cold_full changes vs HEAD b4499e9 (worktree
  twin build, identical Release flags): cross-build WE 4.3e-6 /
  WB 9.1e-6 / bline 3.6e-4 / probe 2.1e-4 — indistinguishable from
  same-build noise. PASS: cold_full landing did not perturb the case-2
  physics. Run data: build/rsm_regress_20260724/.
- Constitution amended (Section 6): ky projection = separate kernel,
  off-by-default flag, no signature/call-path changes to the validated
  parallel path; flag-off must pass this gate before any RSM commit.

## RSM implementation switched to PURE SPECTRAL (2026-07-24, user decision)

- User: "i dont want to add ny, I said i want this to be 1d code, just
  have spectrum in ky". Constitution Section 6 amended: first
  implementation = (A) direct spectral. ny = 1 forever on this arm.
- Architecture: m=0 = untouched legacy 1D path; m=1 = NEW complex 1D
  arrays {E1,B1,J1,cold v1}(x) evolved with d/dy -> ik1 (exact, no y
  grid dispersion); particles keep stored y as pure phase coordinate;
  gather adds 2 Re[F1 e^{ik1 y_p}] in a SEPARATE rsm pusher kernel;
  deposit adds a k_deposit_j1 pass (shape x phase, complex atomics).
  Flag off -> no new kernel launches, no allocations: legacy path
  untouched BY CONSTRUCTION (+ statistical gate anyway).
- Care point: m=1 Gauss consistency (dt rho1 + dx J1x + ik1 J1y = 0);
  direct deposit + per-step 1D complex Gauss correction of E1;
  E_par/E_perp vs Stix is the acceptance test (B2 benchmark).

## Case-2 regression gate EXTENDED to the tiled-deposit path (2026-07-24)

- User clarified: THE successful Chen 2026 case-2 reproduction is the
  GIANT deck (ppc 110000 + tile_sort=25, build/chen2026_case2_giant) —
  which exercises the M9 TILED deposit, not the flat-atomics path my
  first regression covered; depositor.hpp was touched by darwin_tc.
- Reran the statistical gate on base deck + tile_sort=25 (ppc 800,
  5000 steps, HEAD worktree twin): all ratios 0.6-1.6 vs the 3x margin
  -> PASS. Both deposit paths now verified unperturbed by the evening
  changes. Data: build/rsm_regress_20260724/tiled_*.
- Note for future gates: run regress_case2.py protocol on BOTH decks
  (flat + tile_sort) — the giant/tiled configuration is the canonical
  reproduction path.

## RSM design review absorbed (2026-07-24, user's static code audit)

- User-side reviewer (no CUDA device; static read) delivered a 6-risk
  audit; all file:line claims SOURCE-VERIFIED here: (1) ny=1 loaders set
  y=0.5 for ALL particles -> direct e^{-ik1 y} deposit = coherent fake
  oblique seed (the trap my plan would have hit); (2) mirror loader
  rejects kappa_v; (3) cold_full off by default in the Chen deck.
- Constitution amended: Section 6.1 "Implementation risk register" =
  R1 theta_p phase state (uniform load, k_perp*uy/gamma advance),
  R2 full m=1 state incl rho1 + div-free oblique seed + divB1 monitor,
  R3 modal Esirkepov / cold-INCLUSIVE Gauss projection (hot-only
  projection fakes E_par -> fake plateau/gap), R4 one total-field Boris
  push (two deposits same worldline OK, two pushes NOT), R5 spectral
  cold fluid needs its own gate (test_cold_fluid_oblique covers the
  real-space path only) + cross-check vs BOTH Stix and real-space
  cold_full, R6 factor-2 ledger (W = W0 + 2W1, P = J0.E0 + 2Re(J1*.E1)).
  Plus semantics: m=-1 = reality conjugate NOT counter-prop; fixed
  k_perp != fixed WNA under chirp; commensurate-only multi-mode; no y
  box (line-density weights, phase period 2pi); jfilter on Re/Im J1 AND
  rho1; boundary A/B scan required for long runs; checkpoint must carry
  m=1 state + theta; Level 1 renamed Li-TYPE benchmark (kappa_v loader
  gap).
- Validation ladder now V0-V6 (V0 = case-2 statistical regression both
  deposit paths, EXECUTED+PASS today; V6 = notch forensics: never claim
  a gap from a single fixed-phase lineout — m0/m1 interference fakes
  notches; y-averaged modal spectra only).

## 2026-07-24 (evening): RSM first code block LANDED — V1 phase-load gate PASS

Green-lit ("go continue"). First implementation block written and gated:

- **theta-storage decision (better than both planned options)**: p.y at
  ny = 1 is ALREADY a pure phase coordinate — the legacy pusher advances
  it by v_y dt/dy and wraps mod 1, and the m = 0 gather/deposit are
  invariant to it. With the deck contract **Ly = 2 pi / k1** (enforced
  in [rsm] finalize + RsmState::init), cell-unit y IS theta/2pi and the
  legacy advance IS dtheta/dt = k_perp u_y/gamma EXACTLY. No new
  particle array, no sort/migrate/checkpoint plumbing, no pusher change
  for the phase. Constitution R1 amended (RESOLVED block).
- **Code**: config.hpp RunParams {rsm, rsm_k1, rsm_seed} (off by
  default); deck.hpp [rsm] section (enable, k1, seed) + finalize
  validation (ny = 1 required, k1 derived from Ly or checked against
  it); include/pic/rsm_oblique.hpp = RsmViews/RsmState (complex float2
  m=1 arrays E1/B1/J1/rho1 on x-nodes), rsm_theta_init (R1 remedy: y
  uniform [0,1) on RNG stream 11, loaders use 0-5), rsm_deposit_moments
  (CIC-in-x rho1/J1 moment snapshot; dynamical worldline-Esirkepov J1
  comes with the field update). Nothing launches/allocates unless
  rp.rsm = 1.
- **Gate V1 PASS** (tests/test_rsm_phaseload.cu, ctest rsm_phaseload):
  pinned y = 0.5 load -> |S|/N = 1.000000 (R1 trap demonstrated live);
  after rsm_theta_init: E|S_rho|^2/N = 1.06 / 0.87 and
  E|S_jz|^2/(N<uz^2>) = 0.92 / 0.73 at N = 2^18 / 2^20 (32 seeds,
  Exp(1) statistics, bounds [0.5, 1.6]) — shot noise + N^{-1/2} law.
- **Non-regression gate re-EXECUTED both paths** (RunParams layout
  changed -> full recompile): flat ratios WE 1.89 / WB 0.39 / bline
  0.98 / probe 1.03; tiled 0.94 / 0.98 / 1.01 / 1.03 vs 3x margin —
  PASS, PASS (build/rsm_regress_20260724/{rsm_cand_v1,tiled_cand_v1}).
- NEXT (V2): spectral m = 1 cold-fluid + Maxwell update (Dx, ik1) in
  rsm_oblique.hpp, uniform-B periodic-x oblique dispersion/polarization
  /E_par gate vs Stix AND real-space cold_full (R5), div-free eigenmode
  seed (R2). Then V3 conservation ledger, V4 Li-type plateau (long run
  — ask user first).

## 2026-07-24 (late evening): RSM V2 LANDED — spectral m=1 cold solver Stix-exact

- **Solver** (rsm_oblique.hpp): m=1 Maxwell with staggered Dx + EXACT ik1
  (k_rsm_faraday/k_rsm_ampere) + complex twin of cold_full
  (k_rsm_cold_fluid/k_rsm_cold_current: symmetric staggered gather/scatter,
  half-kick/exact-rotation/half-kick; Re/Im rotate identically since the
  rotation is a real operator). Staggering: x-staggered like Yee (E1x, B1y,
  B1z at i+1/2) but COLLOCATED in y — inheriting the Yee y half-shift would
  be e^{i k1 dy/2} = e^{i pi} = -1 at ny = 1, a sign catastrophe. Time
  layout identical to legacy step_at (faraday-half -> [hot J1 slot] -> cold
  -> faraday-half -> ampere) via rsm_cold_step().
- **Gate V2 PASS** (tests/test_rsm_cold_dispersion.cu, ctest
  rsm_cold_dispersion): k1 = 0.4, kx modes theta = 22-64 deg, div-free
  B1z-only white seed (k.B1 = 0 exactly, R2 satisfied w/o projection).
  Refined level: omega vs Stix <= 0.80% all modes (worst = the omega =
  0.07 wce mx=1 mode, measurement-floor-limited); interior modes 0.03-0.4%
  with clean 2nd-order convergence; energy drift +0.0000 EXACTLY (zero
  numerical damping, better than the real-space gate's margin);
  polarization |Ex/Ey| (= E_par response, B0 || x) and |Ez/Ey| match the
  Stix eigenvector to <= 1.1% relative (gate 5%). R5 cross-check satisfied
  transitively: same whistler_w_exact solver as test_cold_fluid_oblique
  which gates the real-space cold_full path.
- No shared-file changes this block (rsm_oblique.hpp + new test +
  CMakeLists only) -> legacy binary unchanged, regression gate not
  re-triggered.
- NEXT (V3, the big one): particle coupling — one-push total-field gather
  (E0 + 2Re[E1 e^{i theta}], R4), charge-conserving modal J1 deposit
  (worldline Esirkepov, R3), MaxwellSimulation::step_at integration under
  rp.rsm, complex continuity/Gauss(cold-inclusive)/energy ledger gates;
  then V4 Li-type plateau.

## 2026-07-24 (night): RSM V3 LANDED — particle coupling + conservation gates PASS

- **k_rsm_push_esirkepov** (rsm_oblique.hpp): ONE Boris push with the total
  field (m0 staggered gather + 1D complex m1 gather, force = 2Re[F1 e^{i
  theta}], m1 dB rides in the wave dB so b0_prof mirror branches work) +
  m0 esirkepov_scatter + charge-conserving MODAL m1 deposit from the same
  worldline. The modal deposit uses the exact factorization S1e1 - S0e0 =
  ebar(S1-S0) + Sbar(e1-e0): x-flow -> 1D Esirkepov prefix sum (J1x links,
  phase ebar), phase-flow -> J1y = i qw Sbar (e1-e0)/(k1 dt dV) = the
  analytic worldline integral. NO Gauss projection anywhere -> the
  cold-inclusive-projection trap (R3) cannot fire by design. Constitution
  R3 + R4 amended with RESOLVED blocks.
- **MaxwellSimulation integration** (shared file touched): RsmState member
  + step_at branches under rp.rsm (m1 faraday halves / fused rsm push /
  1D binomial filter_j1 / m1 cold twin / m1 ampere), all zero-cost when
  rsm = 0. Scope guards at construction: rsm rejects deltaf (full-f only),
  tile_sort (flat path only), bnd_x (periodic until V5), pump.
- **Gate V3 PASS** (tests/test_rsm_conservation.cu, ctest rsm_conservation):
  A) deterministic kick = qm dt 2Re[E1 e^{i theta}] to 1e-6 (factor-2 +
  phase, kernel-level); B) modal continuity residual 4.9e-5 of largest
  term (float atomics roundoff); C) divB1/(k1|B1|) = 3.1e-6 after 200
  steps; D) energy ledger W = KE + Wm0 + 2Wm1 conserved to 3.6% of
  transferred energy through a T_perp/T_par = 12 anisotropy instability
  where the m1 harmonic grew from 1e-8 seed to 2W1 = 2.2 (ledger
  exercised at saturation amplitude).
- **Non-regression gate re-EXECUTED both paths** (simulation_maxwell.hpp
  changed): flat ratios WE 2.43 / WB 0.63 / bline 1.00 / probe 1.00;
  tiled 0.56 / 1.18 / 1.03 / 1.02 vs 3x margin — PASS, PASS
  (rsm_cand_v3 / tiled_cand_v3). Full project builds clean; all 10
  unit-label ctests pass.
- NEXT: V4 = uniform-B Li-TYPE kinetic benchmark: hot kappa-tail load +
  cold_nc, m0 parallel chorus band + m1 oblique response, Landau plateau
  formation in f(v_par) around Vp(0.5) — LONG run (~1000 tau_g class),
  ASK USER before launching. Then V5 mirror + chirp coexistence
  (needs bnd_x m1 damping + chirp2d runner wiring: theta init + rsm
  deck + m1 diagnostics dumps).

## 2026-07-24 (night, cont.): V4 full run LAUNCHED + V5 infrastructure landed

- **V4 pilot PASS (wiring)**: rsm_band runner (liband_yee RSM twin: B0
  strictly along x, obliquity = spectral k1 = 0.16; theta init wired; m1
  complex line dumps; 2W1 in energy.csv) + decks/rsm_band_li.ini (Li G2.1
  plasma, geometry inverted, tile_sort = 0) + scripts/plot_rsm_band.py.
  Pilot 30k steps: m0 anisotropy band growing (WB x6), m1 at shot-noise
  floor with whistler-branch structure already visible in the SEPARATE P1
  spectrum, f(vpar) unchanged (expected at t = 120/We). 145 steps/s
  (flat-path atomics on the 1024-cell grid = the cost of the rsm
  tile_sort guard). FULL RUN launched: 500k steps = t 2000/We = 318
  tau_g, ~58 min (build/rsm_band_full). Level-1 criteria: m0 band vs
  G2.1; m1 oblique band on Stix; Landau plateau at +-Vp ~ 0.1c formed by
  the m1 channel ONLY (m0 is strictly E_par = 0 now — the clean
  two-channel separation).
- **V5 infrastructure landed (compiled, GPU-gated after V4)**:
  k_rsm_damp_x (complex twin of the Umeda masks, node/half sites + m1
  cold); bnd_x guard removed; specular wall + hybrid transverse damping
  copied into k_rsm_push_esirkepov (fold BEFORE deposits; phase y
  untouched); checkpoint m1 schema (dtype 4 = float2, e1*/b1*/vc1*
  conditional manifest entries — theta rides in "py"); chirp2d wiring
  (rsm_theta_init on fresh start only — resume restores theta; m1line
  dumps). PENDING GPU: checkpoint round-trip smoke + regression gate
  re-run (chirp2d.cu, simulation_maxwell.hpp, checkpoint_io.hpp touched).

## 2026-07-24 (late night): V4 VERDICT (honest) + V5 infra GPU-gated PASS

- **V4 full run COMPLETE** (rsm_band_full, 500k steps = 318 tau_g, 37 min):
  both channels grew from shot noise and saturated (WB = 1.6e-2,
  2W1 = 2.0e-2 — m1 comparable to m0); omega-k emission falls EXACTLY on
  the cold whistler branch (kinetic self-consistency at saturation);
  resonance-region f(vpar) gain 1.35-1.9x in v = 0.08-0.25.
- **V6 forensics BLOCKED a false gap claim**: crude window stats showed
  "gap/LB = 1.6e-5 at 0.45-0.55" — quarter-by-quarter analysis proved it
  is the INTER-LINE VALLEY of the discrete mode comb (dk = 0.123 ->
  whistler-mode spacing ~0.03-0.05 wce; valley floor ~constant across
  quarters = NOT progressive carving). P0 envelope is continuous 0.2-0.7;
  NO two-band, NO gap at 318 tau_g in the 51.2 c/wpe box. Verdict:
  V4 = engine + channels PASS; band/gap study needs the finer-k longer
  setup (Li repro needed 900+ tau_g) + an rsm-OFF control for clean
  Landau attribution (tail heating is two-sided -> cyclotron scattering
  contributes; the control subtracts it).
- **V5 infra GPU checks ALL PASS**: legacy checkpoint format test OK;
  RSM save/resume round-trip OK on decks/rsm_chirp_case2.ini (the
  HYPOTHESIS deck runs end-to-end: dipole mirror + cold + hybrid bnd +
  rel + rsm; m1 state restored, fields finite/continuous across resume);
  case-2 regression gate re-PASS both paths (flat <= 1.90x, tiled
  <= 0.98x; rsm_cand_v5/tiled_cand_v5).
- Level-1 completion options (user decision): (A) bigger-box longer V4
  (nx 4096, Lx 204.8 -> dk/4; ~2000+ tau_g; hours) for real band/gap
  statistics + rsm-off control run; (B) proceed to V5 chirp+m1 pilot
  (rsm_chirp_case2, 333k steps ~ 2-3 h at ppc 800) and study the
  processing question directly in the mirror geometry.

## 2026-07-25: rsm-OFF CONTROL BASELINE COMPLETE (band_ctrl_big, 1000 tau_g)

- **Parallel null hypothesis QUANTIFIED**: strictly parallel two-population
  physics alone produces a SHALLOW 0.5 depression — smoothed min(0.45-0.55)
  /LB per quarter = 0.22 / 0.11 / 0.10 / 0.082 (deepens only 2.7x over
  1000 tau_g). Mechanism: cold-kappa cyclotron damping strongest where
  |v_R| = (We-w)/k_par is minimal (= 0.5). This is the no-Landau baseline
  the RSM gap must beat: Li/G2.1 gap class is 3e-4.
- **f(vpar) heating PERFECTLY symmetric** (+1.337 vs -1.335 at [0.08,0.12];
  all bands match to 3 digits): pure pitch-angle scattering, zero E_par by
  geometry. Any +/- asymmetry or +Vp-localized excess in the RSM run is
  the m1 Landau channel, cleanly.
- Retrospective: small-box RSM resonance gain (1.35) ~= this parallel
  baseline (1.34) -> confirms the V4-small "attribution unclean" verdict;
  m1 Landau net effect was not yet visible there.
- **Quantitative criteria for rsm_band_big**: gap min/LB << 8.2e-2 AND
  faster-than-baseline deepening; Landau = excess + asymmetry at
  +[0.08,0.16]. Spectrum quasi-continuous at dk = 0.031 (comb resolved);
  WB saturates 0.28 (dB/B ~ 4%); omega-k textbook whistler branch.
- Big RSM run NOT yet launched (user directive) — launch command in
  memory/state; ~6-8 h flat path.

## 2026-07-25 CRITERIA CORRECTION (user caught it): NO +/- asymmetry expected

- The "Landau = +/- asymmetry" criterion was WRONG: the periodic box grows
  +-k_par waves symmetrically (also within m1; k_perp does not break
  x -> -x), so Landau plateaus form at BOTH +-Vp (as in Li 2019 itself).
- CORRECTED discriminant = LOCATION in velocity space: cyclotron acts at
  |v_R| = (We-w)/k_par ~ 0.15-0.3c (tails, smooth monotonic gain profile —
  measured in the control: 1.11/1.34/1.51/1.81 rising with |v|); Landau
  acts at |Vp| ~ 0.08-0.12c INSIDE it. Signature = localized excess
  flattening at |v| in [0.06,0.14] deviating from the smooth control
  profile, both signs, in the RSM-minus-control difference.
- Why the big box can show it at all: (1) resonance OVERLAP — dk/4 turns
  isolated-mode trapping oscillations (no net diffusion) into quasilinear
  diffusion once mode spacing < trapping width; (2) cumulative time
  (900+ tau_g for spectral back-reaction in G2.1); (3) statistics
  (32.8M markers resolve the localized deformation).
