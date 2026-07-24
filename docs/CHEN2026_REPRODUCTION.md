# Chen et al. PoP 33, 072105 (2026) Fig. 1 reproduction (chirp2d full-f + dipole B0(s))

Target: "Computational study of whistler-mode wave excitation in the Earth's
inner magnetosphere" — Xu Chen, Huayue Chen, Lin, Zhao, Shi, Wang;
doi:10.1063/5.0311180. Their Fig. 1: three whistler regimes at ωpe/Ωe = 5
distinguished ONLY by hot-electron density and anisotropy:
Case I incoherent whistler (nh/ne0 = 0.0316, A = 1.19), Case II
discrete-element chorus (0.0178, 1.52), Case III hiss-like chorus
(0.0133, 3.75). Waves grow spontaneously from noise — NO antenna.

## Their model (CORRECTED 2026-07-20 late): gcPIC-δf + cold fluid

The gcPIC package has THREE models (Chen 2023 GRL, 2023GL103160, verbatim):
gcPIC (full-f), gcPIC-δf, gcPIC-hybrid. First reading concluded full-f
(the method refs [43,45] cite the full-f papers Wang 2024 / Lu 2019, and
the text never says "δf") — WRONG. The paper is gcPIC-δf for the hot
electrons (user-confirmed with the authors' lineage): the giveaway was the
physics, not the citations — "initial noise level ~1e-5 B_e0" is IMPOSSIBLE
for a plain-loaded 800-ppc full-f (fluctuation-dissipation pins that at
~5e-4; our own measurements sit exactly there), but is the natural
statement of a SEEDED field noise in a δf run, and their late onsets
(t = 1150–7250) require the seed to be real at waveform level.
Configuration: hot = δf markers (loss-cone subtracted bi-Max f0), cold =
fluid, immobile ions, dipole B0(s), 1e-5 B_e0 seeded noise, NO injection
("as particle injections are not included"). Lu 2019's field-line
equilibrium load (their ζ factor, Eqs. 14–17) is ALGEBRAICALLY IDENTICAL
to our (E,μ) mirror mapping.

Lesson for the record: method citations in this lineage are loose — decide
full-f vs δf from the NOISE PHYSICS (can the quoted floor exist at the
quoted ppc?), not from the reference list.

## New machinery this reproduction added (all gated)

- `[background] profile = dipole  lre = <L·R_E in c/ωpe>` (b0_prof = 2):
  exact dipole B(λ(s)) behind bg::b0x/db0dx via an even-polynomial fit
  (background_b0.hpp fit_dipole; 8e-8 rel accuracy — float round-off).
  Gate test_dipole_profile: fit vs exact + bounce period vs independent RK4
  (0.16% nonrel / 0.12% rel), μ spread < 5e-4. This front-loads M5a.
- Loss-cone subtracted bi-Maxwellian load (their Eq. 1, ρ = 1, κ = 0.3):
  `[species] dist = losscone  rho  kappa`. Written as a signed mixture of
  two bi-Maxwellians (weights 1/(1−ρκ), −ρκ/(1−ρκ); T⊥eq, κT⊥eq), each an
  (E,μ) equilibrium ⇒ maps along the field line component-wise; the local
  perp pdf keeps amplitude ratio exactly ρ at every x (n_i/T⊥i is
  x-independent). Sampled by rejection off the Rayleigh(T1(x)) envelope,
  acceptance 1 − ρ exp(−u⊥²(1/T2−1/T1)/2) ≈ 1−κ.
  n(x)/neq = [T1(x) − ρT2(x)]/(T⊥eq(1−ρκ)).
- `[diagnostics] probes = <offsets from equator>` (chirp2d).

## Geometry calibration

Their scale is a ×10-compressed dipole ("reduced effective Earth radius").
Pinned by their stated λ = 5° ↔ h = 116.4 V_Ae0/Ω_e0 (= c/ωpe 1:1):
lre = 116.4 / s(5°) = 1330.504 c/ωpe. Domain 5000 × 0.26 = ±650 → λ_max =
26.6° ("up to ~30°" in the paper), wall mirror ratio 2.47. All other
parameters map 1:1 (V_Ae0/Ω_e0 = c/ωpe; Ω_e0 = 0.2 ωpe; dt 0.03/Ωe =
0.15/ωpe; T∥ = 20 keV → U∥ = 0.19784c).

## Result (docs/figs/chen2026_fig1.png; decks/chen2026_case{1,2,3}.ini)

Final runs: full-f, ppc = 3200, jfilter = 3 (build/chen2026_case{1,2,3}_q;
first-pass ppc = 800 no-filter runs kept in build/chen2026_case{1,2,3}).

| | paper | ArcWarden |
|---|---|---|
| Case I regime | incoherent whistler band, no chirping | ✓ same |
| Case II regime | ONE discrete rising-tone element | ✓ single element, ridge 0.205 → 0.532 Ωe over t ≈ 670–1730/Ωe, subpackets |
| Case III regime | dense packed risers (hiss-like) | ✓ same |
| Case II peak δB/B | 0.027 (T₂ = 7713) | 0.030 (t ≈ 1018) |
| Case III peak δB/B | 0.038 (T₃ = 1237) | 0.038 (t ≈ 440) |
| Case I peak δB/B | 0.005 (T₁ = 16876) | ~0.015–0.02 (t ≈ 1100) |
| onset times | 1150–7250/Ωe | ~5–7× earlier |

The regime taxonomy + Case II/III peak amplitudes reproduce. Cost: ~35 min
total on the 5090 for all three (their Fig. 4 scan of 323 such cases is an
overnight job for us).

## Known deviations, all traced to ONE cause: seed noise floor

Our full-f load noise: δB/B ≈ 4.5e-4 rms (ppc 3200 + jfilter; 6e-4 at their
ppc = 800) vs their stated 1e-5. Consequences: earlier onsets (no long quiet
linear phase), Case I overshoot (0.02 vs 0.005 — that case is
amplitude-limited by the linear phase they resolve and we skip), weak
repetitive re-excitation after the main element (they see none — no
injection; our residual noise re-triggers).

A fluctuation-dissipation estimate says 800-ppc plain-loaded full-f CANNOT
sit at 1e-5 rms (it is the thermal PIC level ~5e-4; our M4 self-trigger
floor rescales to exactly our measured value). So their 1e-5 is either
(a) a per-STFT-bin / filtered-band number (our per-bin background IS
~3e-5), (b) routine k-space filtering in gcPIC never described in the
papers, or (c) a quiet-start load that delays thermalization. None of the
three method papers (Lu 2019, Wang 2024, Chen 2023) states any smoothing.
Next lever if we want their onset times: gyrophase-paired quiet load
(φ, φ+π twins at identical x cancel initial J⊥ exactly), noted as the
mirror-loader "future refinement".

ctest 32/32 (new gate dipole_profile), compute-sanitizer clean on the
dipole + losscone chirp2d path.

## δf campaign (2026-07-20 late session): method-faithful reproduction + what their paper doesn't say

After user confirmation that the paper is gcPIC-δf, Case II was rerun in our
δf branch. Seven runs (build/chen2026_case2_df{,2..7}, all kept) bracketed
the system one ingredient at a time:

| run | config | outcome |
|---|---|---|
| df1 | δf + WHITE 1e-5 field seed, rel | dead (in-band fraction of white noise ~1e-2) |
| df2 | band-limited 1e-5 seed, rel | ONE amplified pass (×4.5) then absorbed — dead |
| gbox | uniform-B periodic box, nonrel/rel | **γ = 3.86e-3 / 3.0e-3 vs their sim 3.9e-3** → loss-cone ∂lnf0 drive VALIDATED at 2%; their push is (effectively) NONREL |
| df3/4 | nonrel dipole | bootstrap ignites but wd>1 numerical runaway → NaN |
| df5 | + wd<1 clamp + cone-drive cap | clean but SUB-critical (one-pass amplifier) |
| df6 | + persistent wd-noise source (floor 8e-5) | floor holds, still sub-critical |
| df7 | + REFLECTIVE boundary (nd=60 ν=0.6, R~10%) | **FULL TIMELINE: quiet → net rate 4.3e-4/Ωe → element at t≈7000 (their T2 = 7713) → peak δB/B = 0.0263 (their 0.027) → rising ridge 0.2→0.4+ with subpackets** |

Figure: docs/figs/chen2026_df_timeline.png.

CONCLUSION — the missing undocumented ingredient is the WAVE BOUNDARY:
a bounded δf system with truly absorbing boundaries (our Umeda R < 1%) is a
one-pass convective amplifier (measured gain ×4.5/pass) and CANNOT ignite
from a one-time 1e-5 seed with no injection. With R ~ 10% the round-trip
gain R·G² > 1 turns it into a whistler cavity slightly above lasing
threshold, whose net rate reproduces their "fixed-position growth rate" and
their entire time history. Their "absorbing boundary" is therefore
partially reflecting (undocumented, decisive).

New machinery landed (all reusable for M5b/KO2016): loss-cone ∂lnf0 drive
(df_dist/rho/kappa, cone cap −10/T1), wd ∈ (−8, 1−1e-4] clamp (upper bound
is an EXACT property of wd = δf/f), [plasma] bnoise band-limited seed,
[species] wdnoise persistent weight noise, [species] taud injection
relaxation, deck chen2022_hooked.ini (Chen 2022 GRL, ready to run).

Method-independence lesson (the physics headline): regime taxonomy and
saturation amplitudes are ROBUST (full-f and δf agree with each other and
the paper); onset times and the very existence of self-starting depend on
seed/boundary bookkeeping that papers in this lineage do not document.

## GIANT full-f runs (2026-07-21): GPU-limit ppc, honest physics

Recipe: ppc = 110000 (equatorial), tiled deposit (tile_sort = 25 —
mandatory, flat atomics collapse 52x at this ppc), rel push, honest
absorbing boundary (nd = 300, ν = 0.1), NO seed/antenna. Shot-noise floor
drops to δB/B ~ 5e-5 — low enough for a clean linear phase, still self-
igniting (the δf runs proved a 1e-5 one-shot seed cannot self-start).

Case II (build/chen2026_case2_giant, 425M markers, 3.5 h): textbook quiet
linear phase (rel γ = 2.6e-3), then REPETITIVE discrete rising tones
0.2 → 0.6 Ωe with subpackets, peak δB/B = 0.0264 at λ = −10° (paper 0.027,
saturation off-equator matching their λ = 10–15°). By-latitude peaks:
eq 0.005 → ±5° 0.006/0.015 → ±10° 0.015/0.026.

Case III (build/chen2026_case3_giant, 288M markers): NOT 425M — ppc is the
equatorial reference and the loss-cone mirror-equilibrium n(s) falls faster
at A = 3.75 (wall density 14% of equatorial vs 46% for Case II;
line-integral ratio 0.68). Run killed at 88% (t = 4380/5000 Ωe) by a system
suspend; all Case III action is at t < 2000 so nothing lost. Result: ONE
broadband burst (0.15–0.6 Ωe, overlapping short risers = hiss), onset ~400,
peak t ≈ 750, then monotonic decay — single flood, no repetition, the
opposite relaxation character to Case II. Peaks by latitude: eq 0.009 →
±5° 0.021/0.023 → ±10° 0.040/0.052 (paper 0.038 at T3 = 1237 ✓ off-equator
saturation; our onset ~500/Ωe early, same noise-floor cause). Case I giant
deck written (decks/chen2026_case1_giant.ini) but NOT run (user cancelled).

Element repetition vs bounce (Case II, measured): equatorial element
spacing 850–900/Ωe early, stretching to ~1500 late. Resonant electrons
(ω = 0.25 Ωe → |v∥| = 0.26c, v ≈ 0.35–0.4c) have T_b = 1900–2500/Ωe in
this dipole → element period ≈ T_b/2: the equatorial resonant population
is replenished by the half-bounce return flow from BOTH hemispheres, and
the regrow time ln(10)/γ ≈ 900/Ωe coincides — a two-clock relaxation
oscillator. Late-time slowdown = reservoir depletion: closed box, no drift
resupply (real magnetosphere refills a source region by gradient-curvature
drift in ~minutes = 1e7/Ωe — far slower than elements, far faster than a
storm; the [species] taud knob or a full-f boundary refresh is the 1D
surrogate, untested).

Figures (4-panel format, scripts/plot_chen2026_giant.py: waveform /
envelope / STFT / δB(h,t) with shared time axis, attached colorbars so no
panel shrinks): docs/figs/chen2026_giant_final.png (Case II),
docs/figs/chen2026_case3_giant_final.png (Case III).

## lre scan (2026-07-21/22): element period ∝ T_b — three-point gradient scan

Question: is the repetition period the half-bounce refill clock (∝ lre) or
the ignition clock ln(B_th/B_floor)/γ? Scan Case II at x20 / x10 / x5
dipole compression (lre = 665 / 1331 / 2661), same latitude coverage
±26.6°, same physics. Decks: chen2026_case2_l{5,20}.ini. Omura 2021
threshold theory (scripts/omura_threshold.py, Eq. 105/97; a = 4.5/lre²,
th ∝ lre⁻⁴ at fixed ω): B_th @ ω = 0.25 Ωe = 7.8e-5 (x5) / 1.2e-3 (x10) /
2.0e-2 (x20); B_opt ≈ 1.1e-2 for all.

x20 (212M markers, complete): NO self-start over 6000/Ωe — all 7 probes
pinned at the 5e-5 noise floor (max 1e-3 transients). Matches th > B_opt
for ω ≤ 0.3 = Omura's literal suppression criterion. Clean negative.

x5 (425M markers @ ppc 55000, complete, ~4.6 h): threshold sits AT the
noise floor → instant broadband ignition (flood t ≈ 1000–3500, like a
mini Case III), then settles into REPETITIVE element complexes. Peaks:
eq 0.016 → ±5° 0.026/0.028 → ±10° 0.045/0.038 → ±15° 0.061/0.060
(convective amplification, N/S symmetric in aggregate; per-element
hemisphere choice random — post-flood deep-gap elements went 3/3 south
at ±5°, ±10° complexes interleave N/S). Emission never dies (WB ~ 1e-3
at t_end 15000/Ωe): low threshold keeps the system marginally lit.

Period (ACF of smoothed equatorial envelope — objective, no peak-picking;
scripts/analyze_chen2026_periods.py): x10 894/Ωe; x5 2795/Ωe (eq),
2446/Ωe at −10° with clean harmonics. Ratio 2.7–3.1 vs 2.0 predicted;
late-vs-late spacing (x10 late ~1500) gives 1.7–1.9 ≈ 2 ✓. Both runs land
inside their T_b/2 band. TWO-SCALE HIERARCHY: the periodic unit at x5 is
the element COMPLEX (2400–2800/Ωe, ACF-coherent) containing 2–3
NON-periodic sub-risers 700–1000/Ωe apart — the dense-riser regime the
low threshold predicts; the bounce refill clock survives as the modulation
envelope. x10 = threshold-gated single shots; x5 = refill-gated bursts.

Sweep rate vs Omura Eq. 88 (dω/dt = 0.4 s0 ω Ωw / s1, equator, no lre
dependence): spectral-ridge rising segments, linear fits, r² > 0.5 kept.
x10: 3 segments, meas/theory 0.70/0.72/0.74. x5: 13 segments, median
0.73 (clean isolated risers 0.7–1.3; flood-phase segments 0.2–0.36 —
amplitude attribution ambiguous when waves overlap). Same coefficient at
both gradients = sweep is amplitude-controlled, lre-independent, exactly
as Eq. 88 says; effective S ≈ −0.29 (vs the assumed −0.4 optimum),
the same level of agreement Omura's and Tao GRL17's own tests report.

Gating evidence (inter-element gaps, equatorial envelope): x10 gap min =
0.8x B_th (rides the threshold, fires on re-cross, γ_climb = 5.8e-3) —
threshold-gated, consistent. x5 gaps: min 8x ABOVE B_th, 100% of gap
super-threshold, yet waits 2000+/Ωe with γ_climb only 6–7e-4 (10x
suppressed — fuel drained); recovery at that γ takes ~2300/Ωe = the
observed gap. Super-threshold waiting = the threshold is NOT the gate:
x5 refill-gating is PROVEN from data. x10 needs an interventional test —
three designed: (1) fuel-gauge diagnostic (equatorial velocity moments;
at ignition x10 aligns on B_w = B_th, x5 aligns on a fuel level),
(2) post-element seed injection (x10 triggers early, x5 refuses until
T_b/2), (3) noise-floor/ppc scan (zero code: x10 period shifts by
~ln2/γ, x5 unchanged).

x5 figures: docs/figs/chen2026_case2_l5_final.png (4-panel, --hp5=232.8
for the doubled latitude scale), build/chen2026_case2_l5_probes.png
(7-probe stack), build/l5_element_audit.png (complex-vs-subpacket audit).
x20 archive: docs/figs/chen2026_case2_l20_final.png.

Checkpoint status (asked 2026-07-22): format schema exists and is
unit-tested (include/pic/checkpoint.hpp, tests/test_checkpoint_format.cu)
but array streaming + chirp2d wiring is NOT implemented (planned M7).
~Half a day to land; do it BEFORE the M5b 24 h run.
