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
