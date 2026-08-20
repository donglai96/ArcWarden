# δf continuity gate — frozen criteria (2026-08-20)

Six-arm matrix (decks/cont_*.ini, 201 steps, --contcheck=100, quiet smoke):

| arm | R/||drho/dt|| | note |
|---|---|---|
| full-f fixed-weight | 5.40e-5 | Esirkepov float floor (identity verified) |
| δf FROZEN weights | 1.310e-3 | = δf deposit float floor (defines the floor) |
| δf LIVE weights | 1.310e-3 | == frozen -> variable-weight residual BELOW floor |
| jfilter=0 raw | 4.89e-4 | filter-free floor |
| dt/2 | 1.215e-3 | unchanged -> floor is float noise, not O(dt) |
| ppc 200 | 1.062e-3 | weak ppc dependence |

FROZEN PASS CRITERIA (production δf runs, via --contcheck + contcheck.csv):
  C-A  R_live / R_frozen-floor <= 2.0   (smoke-calibrated; floor rerun
       whenever deck geometry/ppc/dt changes)
  C-B  absolute R/||divJ|| recorded every check; a x10 jump between
       consecutive checks = investigate before trusting E_par physics.
Caveat: at element amplitudes wd excursions grow — C-A is exactly what
--contcheck monitors in-run; if it fails, weight-source correction or
Gauss projection becomes mandatory (audit ruling).

## RETRACTION + CORRECTED VERDICT (2026-08-20, audit-3)
v1 frozen arm was BROKEN (wdfreeze=1 landed in [antenna] via sed; frozen==
live was trivial). TRUE frozen arm (cg_frozen2): R/||drho/dt|| = 3.77e-6
— live delta-f (1.31e-3) is 350x ABOVE the frozen floor. The variable-
weight residual R_w = (q_new-q_old)S_old/dt is REAL and measurable even in
quiet smoke. Criterion C-A FAILS as frozen -> per the audit ruling the
production delta-f path needs a weight-source current / longitudinal
current correction / Gauss cleaning (WarpX-style) BEFORE E_par physics
claims. P2-real remains pilot-only. Next: driven large-dwd live/frozen
comparison + production divE-rho monitor + ppc(25/40/50 x3 seeds)/jfilter
(1/2/3) scans per audit-3 plan.

## GAUSS CLOSURE LANDED + GATED (2026-08-20 day-2)
The mandated correction is implemented as production Marder-Langdon
cleaning (deck `[diag] gauss_clean_every`, <0 = monitor-only):
- cold-charge ledger rho_c integrated every step from the SAME face
  stencil as k_cold_current; mask-damped alongside E/Vc (an undamped
  rho_c in the absorber frame reads as fake residual); checkpointed
  ("RHOC" trailer, absence tolerated with warning).
- hot delta-f charge via k_rho_node, binomial-filtered with the SAME
  passes as filter_j before the residual — divE pairs with the FILTERED
  deposit (continuity_probe lesson); raw-rho pairing overstated the
  residual ~11x (shot-noise mismatch, not real charge error).
- one Marder pass at half the explicit-diffusion stability bound,
  D = 0.25/(1/dx^2+1/dz^2) (Nyquist error exactly annihilated). First
  cut used D = 0.25 dx^2/dt = 13x the bound at dt 0.15 and NaN'd in two
  energy rows (gz_clean v1) — kept as the cautionary number.
- residual scalar = interior-only rms(divE - rho_c - rho_h), logged as
  `gauss_res` in energy.csv; guard: refuses full-f species (equilibrium
  charge untracked).

GATE (scripts/gauss_gate.py, decks/gauss_{mon,clean}.ini, 8000 steps
driven delta-f at L0=1086 with antenna ON to t=1200, wdrms -> 8e-2):
  G1 stability PASS; G2 end-residual ratio clean/mon = 0.47 PASS;
  G3 W_EM + all 14 probe |B| envelopes ratio 1.000 PASS;
  G4 wdrms ratio 1.000 PASS.
Monitor arm: residual 2.8e-6 at t=1200, tracks wave amplitude (not
secular) — the accumulated Gauss error of the live-weight deposit is
bounded and now actively cleaned. Production delta-f decks run with
gauss_clean_every = 20.
