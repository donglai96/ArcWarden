# PLAN_2D_REBORN — the 2D instrument for the 0.5 f_ce gap and the upper band
(branch `2d-reborn`, drafted 2026-08-17; supersedes PLAN_2D.md v1–v4 as the
2D design document; PLAN_2D's Phase-Q numerical verdicts and the H1/H2 gap
hypothesis space carry over unchanged)

## 0. Charter — three questions, one instrument

1. **Why does chorus have a power gap at ~0.5 f_ce?** (Gao 2019: 2/3 of
   events, centered 0.49 f_ce, width ~0.07 f_ce)
2. **What drives the upper-band whistler wave?** (UB confined to |λ|≲15°,
   Meredith 2009; discrete elements inside UB, Li 2019 Fig 1c)
3. **What forms the gap** — i.e. which causal chain, out of the hypothesis
   space below, actually operates?

Hypothesis space (pre-registered, carried from PLAN_2D v4 + PLAN_UB_1D):

- **H1 (no memory)**: the gap is a *separation*, not a carving — corridor
  stop at ~0.5 (Omura budget exhausted, WHY_NO_060), v_g/v_ph degeneracy,
  ambient-f₀ Landau damping resident at V_p(0.5), or two independent
  sources for two bands. Predicts: gap appears with the first element,
  survives τ_D > 0, no Δf precursor.
- **H2 (memory / plateau carving)**: LB-generated oblique E∥ writes a
  Landau plateau spanning [V_p(ω_LB), V_p(0.5)]; the V_p(0.5) = |V_R(0.5)|
  degeneracy (verified: both = 0.0995c at ω_pe/Ω_e = 5) pins the cyclotron
  image of the plateau top at 0.5; the plateau then cyclotron-damps
  counter-propagating 0.5–0.55 waves. Predicts the causal order
  **E∥ work → Δf(0.08–0.10c) → plateau → 0.5 power drop**, cumulative over
  elements, erased if τ_D > 0, direction-sensitive (one-way-street test).
- **UB mechanisms** (PLAN_UB_1D §1): A plateau-split two-component,
  B lower-band cascade (2ω, phase-locked — the only 1D-demonstrated one),
  C multi-source propagation/ducting, D injected low-energy anisotropy.

The instrument must therefore have, *simultaneously*: self-generated
obliquity (E∥) evolving with latitude, kinetic electrons in the 0.1–3 keV
plateau range, a runway long enough for convective UB gain, multi-element
duration, a noise floor far below element amplitude, and diagnostics that
can book every joule (Landau vs cyclotron, per species, per region). No
existing code — ours included — has all six. That is the redesign.

---

## 1. Verdicts on the existing instruments (why redesign)

### 1.1 Our mirror2d (paraxial slab ribbon) — four structural faults

The engine underneath (Yee/Esirkepov, tile-sort, cold_model=full, (E,μ)
loader, checkpoint) is validated and is kept. The *geometry and species
architecture* are not adequate for the charter:

1. **No propagation-generated obliquity.** Pre-registered ray tracing in
   the exact prof=3 slab (scripts/ray_mirror2d.py, G2.2 prep) showed rays
   twist ≤ 0.6° across the ribbon — the paraxial slab (Ly ≪ Lx) has no
   usable field-line curvature. All oblique power must come from the
   instability's own θ-spectrum. The H2 loop ("E∥ written at λ ~ 10–25°")
   and the observed WNA(λ) growth (Lu 2019 Fig 5–6) are structurally
   untestable. This is the single biggest wrongness.
2. **y-periodic wrap artifact.** By ∝ ỹ jumps by 2aB₀x̃Ly across the wrap;
   the straddling gyro-orbit layer is elastically pitch-scattered
   (measured: dT∥ −10.5%/dT⊥ +5.6% per 2T_b in the layer). Survivable, but
   it is a non-physical scattering agent sitting exactly on the quantity
   (pitch-angle structure) the gap physics lives in.
3. **Single-δf-species collapse.** The v4 δf pivot was correct (full-f
   noise floor 1.3e-3 ≥ signal; δf 2.1e-6), but the implementation guard
   "exactly one kinetic δf species + cold fluid" makes the UB question
   unposable: engine (keV, A~1) and shelf (0.1–2.6 keV, A₂) electrons
   cannot be represented, ledgered, or ablated separately. The 1D
   history's failure ("bimax A is energy-blind → γ_UB < 0 by
   construction") gets rebuilt into 2D.
4. **Metric/normalization hygiene bolted on late.** Local-f_ce band
   normalization, absorbers-outside-runway, ρ⊥/dy ≥ 2, probe-in-absorber —
   all discovered as run-design defects (v3 review) and enforced by
   analysis-script discipline instead of by the code. The reborn code
   makes them deck-finalize *hard gates*.

### 1.2 Lu-group 2D gcPIC (Lu 2019 JGR; Ke 2017…2025) — kept vs suspect

Verified from the PDFs (docs/lu2019.pdf p.4158–4160; GAP_PLAN Lu-audit):
cold electrons are a **T = 0 fluid** (their Eqs. 1–3), hot electrons
relativistic full-f PIC, modified orthogonal dipole coordinates (p,q),
grid **64 × 4000**, Δ⊥ = 3.13ρ_e0 at the equator (ρ_e0 = 0.705 c/ω_pe),
Δ∥ = 0.26–0.68ρ_e0, Ω_e0Δt = 0.02, ppc ≈ 1000 (hot only, 2.56e8),
absorbing waves + reflecting particles (Hu & Denton 2009), n_h/n_c = 1%,
ω_pe/Ω_e = 5, T⊥/T∥ = 6, β∥ = 0.01, L = 0.6 R_E topology (≈ ×12
compression, l_re ≈ 1131 c/ω_pe).

**Keep (they set the field standard):** electron-hybrid closure (fluid
cold + kinetic hot); Chan-1994 force-balance loading (= our (E,μ)
mapping, verified algebraically identical); absorbing-wave boundaries;
the near-square domain with real field-line curvature (their WNA(λ) and
poleward S∥ are the target phenomenology).

**Suspect details (each with the number that condemns it):**

1. **Fluid cold kills the plateau channel by construction.** The gap's
   kinetic engine (Li 2019, reproduced in-house: plateau at |v∥| ∈
   [0.08, 0.10]c → gap/LB → 3e-4) lives in electrons at V_p(0.5) ≈ 0.1c ≈
   0.1–3 keV. In every Lu-group run those electrons are inside the T = 0
   fluid: no Landau damping, no plateau, no gap — and indeed **no run in
   the series shows a gap or a two-band structure** (Ke 2025 concedes UB
   needs finer grids). A gap/UB study cannot inherit this closure.
2. **Transverse resolution contradicts their own WNA claim.** Δ⊥ =
   3.13ρ_e0 = 2.2 c/ω_pe → transverse Nyquist k⊥ = 1.43 ω_pe/c. Their
   reported WNA reaching 50° at λ = 30° (Fig 5j) with k∥ up to
   ~1.4 ω_pe/c implies k⊥ ≈ 1.7 ω_pe/c — **beyond the transverse Nyquist**;
   at 40° WNA it is ~2.5 cells per perpendicular wavelength. Yee-class
   grid dispersion at 2–3 cells/λ is O(10–40%) in phase speed: the high-λ
   oblique field they report is numerically mangled, and any would-be UB
   obliquity (k_tot ≈ 2.3 ω_pe/c at 0.65Ω_e, 60°) is unrepresentable.
   The series-wide UB absence is thus partly *numerical*, on top of (1).
3. **2D metric ambiguity for a 3D dipole.** The 3D dipole is solenoidal
   only with the azimuthal metric h_φ = r·cosλ. A planar orthogonal (p,q)
   solver (their "orthogonal coordinate system" statement) either carries
   h_φ (axisymmetric ring waves — not documented, and wrong for chorus
   packets) or drops it (then the background's meridional flux-tube
   narrowing, visible in their Fig 1b grid, is inconsistent with the wave
   metric: WKB focusing toward λ = 30° is overdone by ~1/√(cos³λ) ≈ 25%).
   Amplitude-dependent results (chirp rate ∝ B_w, their Fig 7e–f Omura
   Eq. 50 check) inherit this systematic. Undocumented in the paper or
   the gcPIC method papers available to us — flagged, not convicted.
4. **Reflecting particle wall with no loss cone.** At λ_w = ±34°
   (R_w = 4.3, equatorial loss cone 29°), every particle is returned.
   For Lu 2019's t ≤ 3000 (≪ T_b ≈ 2e4) it hardly matters; for the
   repetitive-emission claims (Lu 2021, longer/1D + injection) the
   boundary/refresh treatment is the pacemaker — our REFRESH_DESIGN
   already logged the circularity critique of the injection rate.
5. **Full-f at ppc 1000 in the ×12-compressed dipole.** B_th ∝ l_re⁻⁴
   (validated in-house ×5–×20): their threshold sits ~2e4× above the real
   magnetosphere's, and their δB/B = 3–4e-2 elements live in a
   trapping-width regime the real belt never visits; meanwhile the
   shot-noise seed and the threshold are degenerate knobs (our lre-scan
   lesson). Their headline phenomenology (risers, WNA(λ), poleward S∥)
   survives this; any *quantitative* transfer (chirp rates, amplitudes,
   onset times) does not.
6. **Charge conservation not documented.** Esirkepov-class deposit in
   curvilinear (p,q) is nontrivial; the paper states ∇·E = ρ/ε₀ but not
   how it is maintained. Unverifiable — listed for completeness.

### 1.3 What survives from our own stack (reused as-is)

- Yee + Esirkepov (charge-conserving to roundoff, test-enforced), tiled
  deposit + amortized sort (14.9 Gdep/s), fused gather-Boris-move-scatter
  (1.55e10 p-steps/s), binomial jfilter.
- cold_model=full linearized 3-component fluid: symmetric staggering,
  local-b̂ Rodrigues rotation, oblique dispersion gated to 0.003–0.46% vs
  Stix, zero numerical damping.
- (E,μ) mirror equilibrium loader (multi-species, isotropic-degrades-to-
  uniform), bi-Max / bi-kappa / loss-cone-subtracted f₀ machinery.
- δf two-weight scheme (Tao PPCF 2017 Eq. 19 lineage) + wdnoise ignition
  control; checkpoint M0 schema (streamed, atomic rename); Umeda masked
  damping; run_meta provenance; the analysis stack (STFT, k–t, ω–k,
  ledgers, ridge tracking).
- The numerical design rules, now promoted to deck-finalize gates:
  ρ⊥/dy ≥ 2 per species; ≥ 8 cells per shortest target wavelength;
  absorbers strictly outside the physical runway; probes never inside
  absorbers; CFL with margin; β⊥-load warning.

---

## 2. The redesigned model

### 2.1 D1 — Geometry: exact 2D flux-function dipole on a Cartesian grid

Background field from a **flux function** ψ(x,z), B₀ = ∇ψ × ŷ — exactly
solenoidal in 2D by construction, curl-free (vacuum) by choice of ψ,
*analytic in the pusher* (never gridded; Faraday touches wave fields
only). Family (deck `[background] profile = ...`):

| profile | ψ / form | use |
|---|---|---|
| `linedipole` | ψ = M·x/r², r² = x²+z² | **production**: 2D dipole analogue |
| `kemirror` | Az = B₀eq(z̃)(1+a x̃²) slab (≡ legacy prof=3) | Ke17 anchor + regression vs mirror2d |
| `uniform` / `tilted` | B₀ = const, any θ | Li-2019 arm, dispersion gates |

The line dipole is the important one. Its properties (derived, to be
gate-tested in V0):

- B_x = −2Mxz/r⁴, B_z = M(x²−z²)/r⁴, **|B| = M/r²** exactly.
- Field lines are circles through the origin: r = L·cosλ (λ = latitude,
  L = equatorial crossing distance = circle diameter). Arc length along a
  line is **s = L·λ exactly** (inscribed-angle theorem).
- Mirror profile along a line: **B(s)/B_eq = sec²(s/L)** =
  1 + (s/L)² + (2/3)(s/L)⁴ + … → the equatorial parabolic scale is
  **l_re = L, exactly the 1D code's parameter**. A 2D run at L = 1330.5
  is the ×10 geometry; L = 3326 is ×4; L = 13305 is ×1. 1D↔2D paired
  runs at identical l_re become clean (the "35× convective suppression"
  question finally gets a controlled experiment).
- L(x,z) = r²/x and λ(x,z) = atan2(z,x) are analytic → per-marker L, s,
  b̂, ∇L are closed-form (needed by the δf weight equation and by
  region-tagged diagnostics).
- Honest analogue disclosure: mirror ratio sec²λ is weaker than the 3D
  dipole's √(1+3sin²λ)/cos⁶λ at equal λ (1.33 vs 2.13 at 30°); the box
  therefore extends to λ_w ≈ 45–55° to buy wall mirror ratios 2–3.
  Equatorial-gradient physics (thresholds, chirp onset) matches the 1D
  dipole family at O(s²) by construction; wall-cone geometry is the same
  compromise every compressed simulation makes, stated not hidden.

Domain: Cartesian (x,z) box excluding the origin — derived by deck
finalize from (L, ΔL, λ_w, margins), e.g. x ∈ [0.34, 1.10]L, z ∈ ±0.58L
for the λ_w = 50° shell (a field line's maximum z is L/2, at λ = 45°). Waves see the whole
box (unducted propagation, refraction, cross-L leakage are *represented*,
not amputated); Umeda masks damp waves in a frame at the box edges,
placed beyond the runway. **Real curvature is back**: WNA evolves along
propagation because the field line actually bends through the Cartesian
grid (the mechanism the ribbon amputated and Lu's metric handles
ambiguously — here it is exact, trivially, because the grid is Cartesian
and B₀ is analytic).

OSIRIS cross-reference (2026-08-17, source-verified in
`osiris_stable/osiris-1.0.0/source/emf/os-emf-gridval.f03`; the user ran a
parabolic-field case with it): OSIRIS `ext_fld = static` evaluates the
external B (uniform / math-func / dipole) ONCE onto grid arrays at the
exact per-component Yee stagger offsets, keeps it out of the Maxwell
advance entirely, and particles interpolate a pre-summed `b_part = b +
ext_b`. Same architectural separation as ours (waves-only Maxwell +
additive background); the difference is gridded-B₀ (O(dx²) representation,
pointwise components so grid-∇·B₀ ≠ 0 at O(dx²)) vs our analytic-at-the-
particle B₀ (exact solenoidality; V0 μ secular 2e-5; ~10 flops/marker, no
memory traffic). Our variant is the strictly stronger form of the same
design; the field standard validates the architecture.

Out-of-plane invariance (∂/∂y = 0) is retained — meridional 2D, k_φ = 0,
same as Ke/Lu. Azimuthal drifts move particles in v_y only, never off
their meridional position: **flux shells cannot leak in-plane** (in-plane
gyrocenter drift is identically zero for this geometry), which is what
makes the next section affordable.

### 2.2 D2 — Electron model: cold fluid + N kinetic δf species (N ≥ 2)

    cold T=0 fluid (bulk, whole box)
      + δf "engine"  species: keV, anisotropic, shell-localized in L
      + δf "shelf"   species: 0.1–2.6 keV plateau/UB-fuel range, own A₂
      [+ optional δf/full-f extra species: pool, second source at L₂, …]

- **Multi-δf is first-class** (the single-species guard dies). Every
  kinetic species carries its own f₀(E, μ; L) with *analytic* ∂lnf₀ —
  bimax, losscone, and (new, required) prodkappa/conecut derivatives —
  its own weight ledger, its own deposit tag, its own fv/J·E diagnostics.
  Attribution (engine vs shelf) is a per-species readout, not an
  inference.
- δf contract in curved geometry: dw/dt = −(1−w)·dlnf₀/dt with
  dlnf₀/dt = (∂lnf₀/∂E)Ė + (∂lnf₀/∂μ)μ̇ + (∂lnf₀/∂L)(v·∇L); all three
  factors closed-form (E, μ from local analytic B₀; ∇L analytic). τ_D
  (weight relaxation) exists as a deck knob but **defaults 0** and is the
  H1/H2 discriminator arm, per the v4 ruling.
- **Shell-compact loading**: hot f₀ carries a raised-cosine L-profile of
  width ΔL (~40 c/ω_pe); δf markers exist only where f₀ > 0. Marker count
  scales with *shell* area, not box area — this is what makes curved-2D
  δf affordable today (§4). Cold fluid fills the whole box (arrays are
  cheap) and is the wave-propagation medium everywhere; deck-driven
  n_c(x,z) profiles enable duct/multi-source arms (UB mechanism C).
- **Cold fluid nonlinearity switch** (new physics capability): the
  validated fluid is linearized — which silently *turns off UB mechanism
  B* (lower-band cascade needs the second-order density/velocity beat;
  Lu's fluid, Eqs. 1–2, keeps ∇·(n_cV_c) and (V_c·∇)V_c). Deck
  `[cold] nonlinear = 0|1` adds the quadratic terms (explicit, same
  stagger). Default 0 (clean linear-response platform); the cascade arm
  and any full-f cross-checks run with 1. A cascade test on a linearized
  cold fluid would have been a pre-registered false negative — this
  switch is the difference between "mechanism B absent" and "mechanism B
  was amputated".

### 2.3 D3 — Fields: Yee/Esirkepov, fp32, jfilter (unchanged physics core)

Darwin remains a cross-check engine only (O(dt) lesson; darwin_tc). The
gap metric spans 3 decades of power — reproduced in fp32 on the Li arm;
keep fp32 fields/particles, double reductions/diagnostics.

### 2.4 D5 — Boundaries

- **Waves**: Umeda masked damping in an edge frame, thickness in physical
  units, *outside* the runway (deck-finalize enforces runway ⊄ absorber).
- **Particles**: **field-aligned reflection** at walls — flip u∥ about
  the local analytic b̂, keep u⊥ phase — exactly (E,μ)-conserving at any
  wall inclination (the curved geometry makes geometric specular flips
  wrong; u∥-flip is both simpler and exact). δf-safe (w, wd untouched).
  Hybrid u⊥-stripping stays full-f-only, off in δf runs (v4 ruling).
- Optional arms: loss-cone absorber + refresh bath (full-f), partial
  ionospheric reflection coefficient (future, SCALES_REALISM gap #4).

### 2.5 D6 — Diagnostics (the ledger IS the experiment)

All cadenced in physical time, checkpoint-consistent, species-tagged:

1. **Band powers in local units**: every station reports ω/Ω_e(local)
   *and* ω/Ω_e,eq (the dual-normalization rule — a fixed global 0.46–0.56
   band aliases the local f_ce shift into a fake gap; 08-07 lesson,
   now in-code).
2. **J·E work ledger**: per species × region × (Landau: qwE∥v∥ |
   cyclotron: qwE⊥·v⊥) × v∥-bin. The H2 causal chain and the Li-vs-Omura
   division of labor are one figure.
3. **f(v∥,v⊥ | region, species, t)** with wd-weighted δf mode (Σw·wd) —
   plateau monitor at [V_p(ω_LB), V_p(0.5)] = [0.08, 0.10]c.
4. **WNA maps + k⊥ station spectra** (B(x_st, k⊥, t) at eq/±λ stations)
   + full 6-component f2d snapshots.
5. **Phase-coherent probes** for bicoherence (mechanism B: phase-locked
   2ω along the element ridge) and for cross-gap phase continuity
   (ke2022 "dots": amplitude filter vs process killer).
6. **Poynting direction** per station (source-region identification;
   pre-registered hemisphere protocol).
7. wd rms/max per species per dump (δf validity gate wd_max < 0.3);
   energy closure (wave + fluid + Σ_s hot + boundary flux).

### 2.6 D8 — Units, deck, provenance

ω_pe = 1, m_e = 1, |e| = 1, ε₀ = 1, c per deck (=5 at ω_pe/Ω_e = 5 ⇒
Ω_e,eq = 0.2 via B₀eq = M/L²) — continuous with the 2D stack. New deck
schema `[domain] [background] [cold] [species …]×N [boundary] [diag]
[run]`; deck-finalize prints the **memory pre-flight** (per-array,
per-species byte budget vs device) and evaluates every numerical gate
(§1.3 list) with PASS/WARN/FAIL before allocation; FAIL refuses launch.
Checkpoints embed the git hash (closing the known provenance gap).

---

## 3. What the instrument can now test (mechanism → experiment map)

| Question | Arm (deck family) | Pre-registered readout |
|---|---|---|
| H1 vs H2 | τ_D = 0 vs τ_D > 0 pair | gap survives → H1; erased → H2 |
| H2 causal chain | flagship + ledger | E∥ work → Δf(0.08–0.10c) → plateau → 0.5 drop, *in that order, cumulative* |
| H2 directionality | one-way street (k-mask one direction) | surviving direction's 0.5 band heals |
| Plateau sufficiency/necessity | plateau surgery (pre-seed / relax) | gap appears / heals |
| UB-A (plateau split) | shelf species ledger in flagship | shelf carved by E∥ before UB rises |
| UB-B (cascade) | `[cold] nonlinear=1` + bicoherence; control =0 | phase-locked 2ω on ridge, present only with nonlinearity |
| UB-C (multi-source/duct) | two shells at L₁,L₂ / n_c duct profile | UB at observer arrives along ray path from remote source |
| UB-D (injection) | shelf (n₂, A₂) ladder above net-threshold curve (PLAN_UB_1D Step 1 tool) | UB band + LB coexist; 0.5 separation without carving |
| runway hypothesis | L-scan at fixed Δx, f₀, absorber-in-physical-units | ln B_UB ∝ ∫γ_UB/v_g ds across L |
| 1D↔2D | paired chirp2d vs warden2d at same l_re = L | convective-gain ratio, corridor endpoints |

Disciplines carried forward unchanged: judgment tools frozen before
looking, pre-registered failure readings, seed triplets for any "stable
source" claim, statistical non-regression envelope for shared-file
changes, staged gates with report-then-continue.

## 4. Memory & scaling (32 GB today, 141–288 GB tomorrow, multi-GPU later)

Marker layout (lean, δf-native): x, z (2×f32), ux, uy, uz (3×f32),
w (f32), wd (f32), cell (u32) = **32 B/marker**; sort via chunked
double-buffer (chunk = max 1/8 of markers) → **36 B/marker effective**
(vs 60 today); optional fp16 w/wd (−4 B, ruled acceptable for weights
only) when a run is memory-bound. Fields ≈ 15 f32 arrays.

Sizing ladder (dx = dz = 0.25 — the value the res/UB hard gate itself
demands for 0.75 Ω_e(local) at 45° WNA; 0.35-class grids are exactly
where the Lu-group series lost its upper band — 2 δf species, shell
ΔL = 40 c/ω_pe, λ_w = 50°, shell-compact loading: markers ∝ shell area
= ΔL·L·(λ_w + sinλ_w cosλ_w), not box area; W10 row is *measured* by the
warden2d pre-flight):

| rung | L (l_re) | box (c/ω_pe) | cells | shell cells | ppc(shell) | markers | GB total | platform |
|---|---|---|---|---|---|---|---|---|
| W10 | 1330.5 (×10) | 1013×1531 | 2.5e7 | 1.2e6 | 100+50 | 1.74e8 | **7.8** | **5090 now** |
| W4 | 3326 (×4) | 2185×3526 | 1.2e8 | 2.9e6 | 100+50 | 4.4e8 | ~23 | 5090 tight / H200 |
| W1 | 13305 (×1) | 8043×13504 | 1.7e9 | 1.2e7 | 100+50 | 1.7e9 | ~170 | B200/GB300 |
| W1-wide | ×1, ΔL 160 | same | 1.7e9 | 4.7e7 | 100+50 | 7e9 | ~360 | GB300 + 2×GPU |

The reborn geometry runs *today* at ×10 with real curvature and better
resonant-shell statistics than the ribbon flagship (δf, markers-per-area
held across grid changes), in ~8 GB. Scaling is a deck edit, not a redesign: nothing in the code
assumes box ≈ shell, and the ×1 rung is the SCALES_REALISM R1-class run
in the curved geometry (first realistic-l_re, realistic-f₀ 2D chorus).

Multi-GPU (deferred, interfaces shaped now): z-slab domain decomposition
(field halo exchange width = max(stencil, jfilter); per-slab tile sort;
migration buffers already species-segmented). Single GPU = one slab; the
kernels take (view, extent) pairs from day one so the decomposition is
additive. Reductions and diagnostics funnel through per-slab partials.

## 5. Validation ladder (each a ctest or a gated run; frozen before data)

- **V0 orbit**: single particle in `linedipole`, rel-Boris, ≥ 5 bounces:
  |u| exact to fp; μ spread < 1%; turning point s_m = L·arccos(sin α_eq)
  to < 1%; bounce period vs ∫ds/v∥(sec²) quadrature to < 1%.
- **V1 quiet hold**: (E,μ)-loaded shell, δf: wd stays at wdnoise, no
  drift over ≥ 2 T_b; n(x,z) matches mapping; fluid-only box quiescent.
- **V2 waves**: cold oblique dispersion vs Stix on `uniform`/`tilted`
  (regression of the 0.003–0.46% gate on the new stack); whistler packet
  WKB amplitude along the converging flux tube vs B^{1/2} law (the test
  Lu's metric ambiguity fails); ray path vs scripts ray tracing in
  `linedipole`.
- **V3 kinetic engine**: Li 2019 tilted-uniform arm re-run on the reborn
  stack → gap/LB reaches 1e-3-class by ~900 τ_g (regression vs G2.1).
- **V4 chirping anchor**: W10 single-species engine deck → rising tones
  in the 0.3–0.5+ corridor (lurepro-class), WNA(λ) growth qualitatively
  Lu-like, poleward S∥.
- **V5 noise & threshold**: δf floor vs wdnoise; B_th vs Omura corridor
  at L = 1330.5 (omura_threshold.py cross-check; lre⁻⁴ point at ×10).
- **V6 forensics rules**: y-averaged/local-f_ce-normalized spectra only;
  no gap claims off a single fixed-phase lineout; connected-component
  ridge protocol with frozen thresholds (A1–A6 amendments carry over).

## 6. Build phases

- **P0 scaffold** (this session): branch `2d-reborn`; this document;
  `include/pic2d/` skeleton — background family (linedipole math + V0
  gate test), lean particle store layout, deck2d schema stub, runner
  stub with memory pre-flight; CMake wiring; V0 passing.
- **P1 fields+fluid — LANDED 2026-08-17** (`include/pic2d/fields2d.hpp`):
  (x,z)/∂y=0 Yee with Range-based kernels, cold_full twin (symmetric
  staggering, EXACT Rodrigues rotation about local b̂ — robust at the
  strong-B inner corner where Ω_eΔt ≈ 0.11), Umeda edge-frame masks.
  V2 gates EXECUTED: oblique dispersion 12 modes θ = 6.6–51°,
  k = 0.49–1.32 vs Appleton–Hartree — **worst 0.053%, best 0.003%**
  (better than the legacy 0.003–0.46% gate); energy conservation 3.2e-3
  over T = 6554 post-transient (the By-only white seed redistributes in
  the first frames — instrumented, understood). Absorber gate in the
  production linedipole geometry: R-whistler packet transits undamped
  (99.88% over t = 400–1200), absorbed to 1.9e-4, wall pile-up 4.3e-6;
  the 50/50 seed split (R-whistler + high-EM-branch) is derived in the
  test header — an isotropic blob is the WRONG absorber seed (25%
  quasi-perpendicular v_g→0 residual, first-run lesson kept on record).
  V2 WKB flux-conservation gate (the one Lu's metric fails) moved to P3
  with the antenna.
- **P2 kinetic — first landing 2026-08-17** (`include/pic2d/kinetic2d.hpp`):
  (E,μ) shell loader (equal-weight rejection ∝ n(x,z), markers only where
  f₀ > 0), fused gather→δf-weight→Boris→reflect→Esirkepov kernel
  (continuity unit-checked to 6.4e-7 vs scale 13.3), u∥-flip walls.
  **V1 quiet hold EXECUTED and green** (2 T_b = 18000/ω_pe, 6.6M markers,
  ~7 min on the 5090): zero escapes, wd_rms *decays* 1.00e-3 → 0.93e-3
  (the weights damp the noise waves), hot KE drift **2.8e-7** — the
  ρ⊥/dz = 2.8 δf grid-heating WARN answered empirically: none — and the
  field floor sits at W_EM ≈ 2.6e-9 (δB_rms ~ 3e-7 vs full-f 1.3e-3: the
  3+ decade δf dividend, measured). Three pre-registered-forensics
  findings, each now a design rule in the header:
  1. **Gyro-center ruling**: f₀'s shell profile must take the
     GUIDING-CENTER L (in-plane offset carries only u_y; r_gc = r −
     γ(u×b̂)/(|qm|B)) — with the particle L, edge markers ring at
     dlnprof·ρ·sec²λ ≈ 0.3–0.5 (measured wd_rms → 0.55, floor 6e-3,
     KE +3.3%). Shell edges are flat-top + Gaussian (bounded dln), NOT
     raised-cosine (tan divergence at the foot).
  2. **jfilter is load-bearing**: without binomial current smoothing the
     shot-noise currents pump grid-scale modes near the resonance cone
     (v_g → 0: energy cannot reach the masks) — seed level ×7 higher.
  3. **Walls exist only at the high-|λ| line ends** (x < wx0, |z| > wz):
     in-plane gc drift is identically zero so L cannot be crossed, and
     testing the outer-radial edge pinned shell-tail markers in an
     equatorial flip loop (z_gc jitter) — a coherent antenna at
     (L = 226, λ = 0) driving a γ ≈ 0.07 Ω_e spurious instability through
     the live-wd loop (bisected: wd-frozen arm clean, mode imaged, fix
     verified: live-wd floor now BELOW the frozen-wd floor, 1.8e-9 vs
     6.6e-9 at 25k steps).
  **V1b anisotropic hold green** (T⊥/T∥ = 2, inert n₀; ζ-mapped loader
  quantitatively confirmed by the 1/ζ shell-integral reduction 8940→7729).
  **V3 Li 2019 anchor EXECUTED and green 2026-08-17**
  (tests/test_li2019_reborn.cu, 1.6M steps, 67 min, 8.4M markers,
  bi-kappa loader): **gap/LB carved 1.82 (τ_g 60–120, one band) →
  2.5e-3 (τ_g 890–1010), run minimum 1.49e-4 at τ_g 845 — the legacy
  3e-4 class**; UB/gap 1.8 → 13 (distinct upper band); saturation
  δBy/B₀ = 1.18% (legacy δB 1.9%); warm plateau fills symmetrically
  +31%/+29% at ±[0.08,0.10]c. Two originally-frozen gate thresholds were
  estimator-miscalibrated (early k–ω window vs probe spectra; plateau
  "10×" was a Δf-map contrast, not a warm-κ fat-tail ratio) —
  recalibrated in-file with the run data archived
  (docs/figs/li2019_reborn_v3.png). Supporting fixes en route: RNG
  stream separation (id·64+s aliasing), replicated-J deposit (contention
  measured NOT dominant: 2.93→2.53 ms/step; tile-sort stays P3),
  periodic particle wrap for wall-less arms.
  **The kinetic engine is now literature-anchored on the reborn stack.**
  Still open in P2: analytic ∂lnf₀ for losscone/prodkappa, multi-species
  orchestration + per-species ledgers (V3 ran two species inline).
- **P3 production — COMPLETE 2026-08-19** (status refresh per external
  review: this section previously read as future work): checkpoint v2,
  diagnostics pack, deck gate battery (now incl. cold-nonlinear refusal,
  nonrel particle-CFL, dist>=2 refusal), sparse tile pool, V4R5 shakedown,
  V4R6-XVAL delta-f cross-validation. Production background = dipole2d
  (linedipole retained for legacy gates only). Post-review fixes landed:
  gyrotropic loader (u_y local width), time-centred delta-f weights,
  gc_pos momentum contract, wd_max monitoring + representation gate.
  **P3+ optimization detour (user-directed 2026-08-18, "算力没有集中在
  有效的区域")**: sparse tile-pool field storage implemented
  (PLAN_SPARSE_GRID §5) — pool + int32 tslot table, guarded+counted
  deposits, ckpt v2, sparse-aware preflight, BAND_MARGIN=130 fixing the
  under-covered absorbing ramp; gates S1 (bitwise dense-vs-pool) and S2
  (three-arm Sim2D envelope) in tests/test_sparse2d.cu. Compact box
  V4R4 (x_min 2500, dL=60 ≥ 5.5 λ∥). V4R5-A5 shakedown deck: A=5 short
  run whose deliverables are the user's validation trio — (1) initial
  density map vs designed shell (dens_init + plot_density2d.py),
  (2) fast whistler excitation with visible chirping (quicklook2d.py
  spectrograms every 10k steps), (3) wave propagation maps in-domain
  (plot_wavemap2d.py from f2d snapshots). Deck-driven cadences
  (probe_every/fv_every/dens_init) close review item D4.
- **P4 science**: §3 arm ladder, in the PLAN_2D staged-gate style
  (report at every stage; no parameter roulette; corridors pre-computed).

## 7. Non-goals (stated, not hidden)

No azimuthal drift physics or 3D ducting (meridional 2D); no k_φ ≠ 0
modes; line-dipole mirror ratio is sec²λ (analogue, documented mapping);
ions immobile; no ionospheric reflection in v1 (absorb + reflect only);
RSM stays on its own 1D arm (ArcWarden_chirp) — this branch is the full
2D instrument.
