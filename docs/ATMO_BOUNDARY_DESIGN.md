# Atmo boundary (bnd_x = 3) + the lre-scan x2 arm — design document

Date: 2026-07-30. Companion to `docs/BOUNDARY_STUDY.md` (M2 absorber history)
and `docs/CHEN2026_REPRODUCTION.md` (lre-scan). Code: `include/pic/config.hpp`
(`bnd_batm`), `include/pic/deck.hpp` (`[boundary] x = atmo`, `batm`),
`include/pic/yee2d.hpp` (reflection-branch precipitation test),
`tools/chirp2d.cu` (`--noeline`). Decks: `decks/giant_x2_atmo.ini` (primary),
`decks/giant_x2_hybrid.ini` (arm B). First run: `build/giant_x2_atmo`.

## 1. Problem: one boundary, two conflated jobs

Every bounded chorus box needs its field-aligned ends to do two physically
distinct things:

1. **Absorb outgoing waves** (in reality: waves refract oblique and Landau-damp
   beyond ~30° latitude and never return);
2. **Terminate the particle trajectories** (in reality: particles mirror
   between ±λ_m and return adiabatically, u⊥ intact, unless their mirror point
   lies beyond the atmosphere — the loss cone — in which case they are
   absorbed, energy and all).

The legacy modes each get one job wrong:

| mode | fields | particles | verdict |
|---|---|---|---|
| 1 `damping` | Umeda mask (+ cold fluid, see §3) | specular, u⊥ kept | loss cone = 0° (too closed) |
| 2 `hybrid` | same | specular + u⊥ ×= exp(−ν_max Δt d²) each step in the layer | loss cone = 44.2° (13× reality in angle, ~150× in pitch space) |
| 3 `atmo` (NEW) | same | specular, u⊥ kept, EXCEPT mapped atmospheric cone → precipitate | loss cone = deck knob, decoupled from wall |

The hybrid over-carving is not cosmetic: the carve-location experiment
(`giant_bigbox_carve26`, 07-29) showed the element amplitude responds directly
to WHERE the u⊥ carving sits (4.72× vs 1.17× B_th for identical boxes), and the
γ-race analysis showed the carving acts as an anisotropy valve that keeps γ
alive (closed box stalls at 0.8×10⁻³ < B_th = 1.24×10⁻³). I.e. at x10 part of
the chorus phenomenology is boundary-made.

## 2. The mirror-ratio ladder: why the wall cannot be the atmosphere

Dipole field along the line: b(λ) = B/B_eq = √(1+3sin²λ)/cos⁶λ. Mirror ratio
and loss cone depend ONLY on latitude — never on lre. Real L = 5 line:
footpoint λ_f = acos(√(1/L)) = 63.4°, b_atm ≈ 230, equatorial loss cone
α_lc = asin(1/√230) = 3.8°.

| λ (deg) | b(λ) | eq. loss cone if wall here |
|---|---|---|
| 15 | 1.35 | 59.4° |
| 20 | 1.65 | 51.1° |
| 23.6 | 2.05 | 44.2° (x10 hybrid layer inner edge) |
| 26.3 | 2.38 | 40.4° (wall) |
| 43 | ≈10 | 18.5° (dt ceiling) |
| 63.4 | 230 | 3.8° (real atmosphere) |

Hard ceiling: the Boris pusher needs Ω_ce,local·Δt ≲ 0.3; at Δt = 0.15/ω_pe,
Ω_e0 = 0.2 ω_pe that caps b ≤ 10, i.e. λ ≤ 43° (and f_ce > f_pe beyond ~36°
changes the plasma regime). Even at the ceiling the wall loss cone is 6× the
real one. **Conclusion: a wall can never emulate the atmosphere by position;
the loss cone must be encoded in the boundary operator.**

### Wall latitude: the four constraints that pick 26.3°

- **Lower bound (physics):** outside the generation + convective-amplification
  zone (gain peaks by ±15°, done by ~±20°; a sink at 15° demonstrably scrubs
  amplification — refresh lam15 lesson). And: the loaded f0's u⊥ hole empties
  α ≲ 34–40°, so particles with α_eq > 40.4° mirror inside a 26.3° box — the
  wall is the loaded population's turning-point envelope; only the (initially
  empty) low-pitch tail ever touches it.
- **Upper bound (numerics):** Δt ceiling λ ≤ 43°; f_ce < f_pe; cost — cells
  ∝ arc length s(λ) and, at the fixed 550M-marker GPU budget, ppc ∝ 1/nx.
- **Heritage:** ±26.3° is the Chen PoP 2026 case-2 calibration box; every
  x10/x5 arm used it. Comparability across the lre scan requires keeping it.
- With atmo, wall latitude is PURE NUMERICS (absorber location); the loss
  cone moved into `batm`. This decoupling is the point of the design.

## 3. Wave absorption: the marker-cold vs fluid-cold correction

The historical justification for hybrid's marker carving ("field-only masks
give R ≈ 1 because the wave energy rides in the coherent electron transverse
current") was measured in `tools/boundary_reflection.cu`, whose cold plasma is
**PIC markers** (ppc = 100). In the production chirp2d architecture the cold
population (98.22% of density) is a **linearized fluid**, and `k_damp_x`
already damps v_cy/v_cz inside the layer in EVERY bounded mode
(`simulation_maxwell.hpp:159`, "cold fluid dies with its wave"). Hot markers
carry ~1.8% of the wave current at the wall, so carving them buys ≤2%
absorption. Production evidence: the original `chen2026_case2` baseline ran
mode 1 and reproduced Chen. **Atmo therefore inherits mode 1's absorber
unchanged** (nd = 300 cells = 78 c/ω_pe ≈ several whistler wavelengths — the
width that matters for absorption; wavelength is lre-independent).

## 4. The atmo particle operator

At the specular reflection event (domain end x = 0 or nx), test whether the
particle's *virtual mirror point* (conserving μ beyond the wall) lies beyond
the atmosphere:

    mirror at b_m where sin²α_local · b_m = b_local
    precipitate  ⇔  b_m > b_atm  ⇔  sin²α_local < b_local / b_atm

Implementation (`yee2d.hpp`, reflection branch; b_local = bg::b0x at the wall):

    if (rp.bnd_x == 3 && refl && rp.bnd_batm > 0.0) {
        const float bl  = bg::b0x(rp, x1*dxp) / rp.B0[0];
        const float up2 = uy*uy + uz*uz;
        if (up2 * rp.bnd_batm < bl * (up2 + ux*ux)) { uy = uz = 0; }
    }

The criterion is invariant to where it is applied: mapped to the equator it is
always α_eq = asin(1/√b_atm), regardless of wall latitude. Relativistic u is
fine (the pitch ratio is identical in u or v).

### Precipitated-particle fate (three options, one chosen)

- **(i) u⊥ → 0, keep u∥, return — CHOSEN.** u⊥ energy deleted = atmospheric
  deposition (correct sink); u∥ energy returned (half-sink, wrong but tiny at
  cone-level flux). The marker becomes a μ = 0 ghost: no mirror force, ballistic
  wall-to-wall, re-triggers the criterion harmlessly (u⊥ already 0) until wave
  fields re-kick it above the cone. Contrast with the precip strip mode (whole
  strip population ghosted): here the ghost flux is only the cone flux, and the
  initial f0 cone is EMPTY, so early flux ≈ 0.
- (ii) refresh-bath redraw = open system with resupply — a different physical
  statement (injection); reserved for infinite-train studies.
- (iii) atmospheric backscatter albedo — real albedo < 10%, overkill.

Rejected: damping u∥ too (piles cold slow markers at the wall).

### Choosing batm: the pump-off rule

Requirement: cone ≪ f0 hole edge (~34°), so the boundary never becomes an
anisotropy pump (hybrid's 44° WAS the pump; matching the cone to the hole edge,
batm ≈ 3.2, would re-create it). Within the pump-off class the exact value is a
secondary knob:

| α_eq | batm | pitch-space sin²α | vs hybrid 0.49 | note |
|---|---|---|---|---|
| 3.8° | 230 | 0.0044 | 1/110 | literal L = 5 |
| **10°** | **33** | **0.030** | **1/16** | **chosen: measurable precip channel** |
| 15° | 15 | 0.067 | 1/7 | still pump-off |

Rationale for 10° over 3.8°: scattering fills the hole from the edge inward, so
a 10° cone produces measurable precipitation earlier (live boundary-condition
diagnostic, microburst-adjacent); and the compressed box already samples the
wall 2× faster than reality (T_b compressed, local rates not), so literal-cone
realism is moot. Early-time results are batm-independent anyway (§5).

Verification gates (07-30, rtest-size, 20k steps): atmo(batm = 1e12) ≡ damping
statistically (wall ⟨u⊥⟩ 0.2484 vs 0.2489, field energies within float-atomic
envelope); atmo(batm = 3, local cut 63°) drains wall ⟨u⊥⟩ by 28% — criterion
fires with the right sign and magnitude.

## 5. The causality window — why the x2 arm decides

At x2 the box is causally huge. Two channels, both counted from t = 0:

- **Particles:** earliest boundary-touched marker at the equator = a marker
  loaded AT the wall, processed at t ≈ 0, one-way transit to the equator.
  x10 measured round trips 640–1680/We0 → one-way 320–840 → ×5 at x2:
  **strict window ~1600/We0** (fast tail), typical 3200/We0.
- **Waves:** v_g(0.25 Ω_e) ≈ 0.13c → equator→wall 25000/ω_pe = **5000/We0**;
  even imperfect absorber reflection cannot reach the equator within the run.

x10's window (~320/We0) is shorter than its element formation time (~1275/We0)
— boundary information always arrives first; x2's window is ~10× the expected
formation time (threshold 625× lower, same linear γ). Hence the ledger:

1. **Element inside t ≲ 1600/We0** + probe-Poynting divergence pointing out of
   the equator ⇒ chorus born from pristine equatorial anisotropy (no boundary
   contribution of any kind — valve, ghosts, or wave reflection).
2. **Broadband flood, no discrete elements** ⇒ discreteness needs the
   threshold gate (non-operative at x2) — boundary role at x10/x5 secondary.
3. **Growth then stall without elements** ⇒ the γ-valve (sustained anisotropy
   maintenance = real precipitation's role) is essential, not just a
   threshold-crossing crutch. x2 uniquely separates this from (1): its
   threshold is low enough that even relaxed marginal-stability γ may cross.
4. **Nothing above noise** ⇒ falsifies the equatorial driver (γ unchanged —
   very unlikely).

Bonus theory-matrix points at a 3rd lre (both predicted lre-free): saturation
vs B_opt = 1.05×10⁻²; sweep rate vs amplitude (Omura Eq. 88; x10 & x5 both
measured 0.72–0.73× Eq. 88).

## 6. Concrete parameters: x10 vs x2

All lengths ×5 (lre ratio), all latitudes identical — that is the whole design.
f0 identical BY CONSTRUCTION because the loss cone depends only on wall
latitude (§2), not lre.

| parameter | x10 `giant_e12` (run) | x2 `giant_x2_atmo` (running 07-30) |
|---|---|---|
| lre (c/ω_pe) | 1330.504 | 6652.52  (real = 13305; x2 = 2× compressed) |
| inhomogeneity ã = 112.5/lre² | 6.36×10⁻⁵ | 2.54×10⁻⁶ |
| nx × ny / Lx / dx | 5000×1 / 1300 / 0.26 | 25000×1 / 6500 / 0.26 |
| wall latitude / b_wall | ±26.3° / 2.38 | ±26.3° / 2.38 |
| dt / nsteps / t_end | 0.15 / 83334 / 2500/We0 | 0.15 / 166667 / 5000/We0 |
| boundary | hybrid, nd=300, numax=0.1 (u⊥ carve 23.6–26.3°) | **atmo, nd=300, numax=0.1, batm=33** |
| effective eq. loss cone | 44.2° (artificial) | 10° (knob; literal L=5 = 3.8°) |
| field model | yee, jfilter=3, tile_sort=25 | same |
| cold_nc / rel / noisy | 0.9822 / true / true | same |
| ω_ce/ω_pe, profile | 0.2, dipole | same |
| hot density / dist | 0.0178, losscone ρ=1 κ=0.3 | same |
| uth (u∥, u⊥, u⊥) | 0.19783565, 0.24390817, 0.24390817 | same (A_uth=1.52; A_eff=1.98 w/ hole) |
| ppc / markers | 110000 / 550M | 22000 / 550M (noise ×2.2 — harmless, see B_th) |
| seed | 20260720 | 20260720 |
| probes | 17 @ 0, ±2.5…±15, ±20, ±25° (s = ±58.1…±609.1) | same latitudes, s ×5 (±290.5…±3045.5) |
| line dumps | bline+eline+jline /100 steps | bline+jline /100 steps (**--noeline**: eline = full-nx E probe, dropped; probe_e keeps probe Poynting) |
| ckpt | --ckpt=25000 (750/We0, rotating ~13 GB) | same |
| B_th(0.25 Ω_e) Omura | 1.24×10⁻³ | 1.99×10⁻⁶ (∝ lre⁻⁴; ≪ noise ⇒ no threshold gate) |
| B_opt | 1.051×10⁻² | 1.051×10⁻² (lre-free) |
| driver bounce T_b | ~2400–2600/We0 | ~12000–13000/We0 |
| wall-reacher round trip | 640–1680/We0 | 3200–8400/We0 |
| causality window (strict) | ~320/We0 (< element time 1275 — broken) | **~1600/We0 (≫ expected element time)** |
| wave eq→wall transit | ~1000/We0 | ~5000/We0 |
| measured outcome | threshold-gated single risers, 3.62× B_th | — (ledger §5) |

Arm B (`giant_x2_hybrid.ini`, ready, not launched): identical except boundary
= hybrid with nd = 1500 (layer covers the same 23.6–26.3° latitude band as
x10's carve — the quantity carve26 proved matters). Purpose: direct measurement
of the boundary-realism effect at x2, same seed.

## 7. Analysis plan (when data lands)

1. Per-probe STFT + peak-local |Bw| (box-averaged WB provably washes out
   coherent elements — 07-28 lesson); first-element time vs the 1600/We0 line.
2. Probe-level Poynting S_x = EyBz − EzBy from probe.bin × probe_e.bin
   (time-averaged; snapshot J·E is reactive-power-swamped) → source location.
3. Checkpoint fuel gauge: equatorial anisotropy A(t) relaxation + ghost-beam
   fraction (u⊥ ≈ 0 markers) → the γ-valve story in one timeline.
4. Saturation vs B_opt and sweep rate vs Eq. 88 at the 3rd lre.

---

# RESULTS ADDENDUM (2026-07-30, both runs complete)

## 8. x2-atmo results (build/giant_x2_atmo, t5000/We0, 1.4 h)

- **Equatorial birth inside the causality window — decisive.** First crossing
  of |Bw| = 10⁻³ (500× B_th): equator t=699/We0, ±5° 726, ±10° 894, ±15°
  1089 — all before the strict window 1600/We0 (earliest boundary-touched
  marker back at the equator). Time-averaged probe Poynting outward
  (N+/S−) in 16/16 windows. Generation involves NO boundary participation.
- **No discrete elements.** Bursty broadband 0.15–0.55 Ωe0 flood; no riser
  tracks at the STFT uncertainty limit; instead a SECULAR band migration
  (peak 0.25→0.35→0.55 over ~4000/We0, ~9×10⁻⁵ We0², bandwidth ~0.3) =
  quasilinear burn-through: equatorial A relaxes +0.98→+0.48, low-ω
  (high-|v_res|) fuel exhausted first. Born flooded: B_th = 2×10⁻⁶ ≈
  noise/25 — the threshold gate never operates.
- sat ≈ B_opt (lat15 peak 1.04–1.08×10⁻² vs 1.05×10⁻², lre-free) — third
  confirmation. No 0.5 gap (late 0.47–0.53 is the PSD peak; consistent
  with gap-needs-k⊥, ny=1 box). Ghost fraction 7×10⁻⁴ (negligible), cone
  filling live (0–10°: ~0 → 1.1%).

## 9. x4-atmo40 Arm-1 results (build/giant_x4_atmo40, t10000/We0, 2.7 h)

Setup recap (the "realistic-boundary" configuration): lre = 3326.26 (x4),
wall ±40.0° (b_wall = 7.4057), equatorial loss cone 21.6° set BY THE WALL
(sin²α = 1/b_wall), f0 = cone-cut bi-Maxwellian (dist = conecut: trapped
population only, load criterion ≡ boundary criterion, zero transient),
boundary atmo with batm = b_wall (every wall-toucher precipitates, u⊥ → 0),
nd = 500 absorber, dt = 0.15 (Ω_ce,wall·dt = 0.222), ppc = 28000 (543M).
Gates: G1 load PASS (0–20° empty, sharp 21.6° edge); G3 dt-convergence PASS
(in-cone fractions identical at dt 0.15 vs 0.075 ⇒ cone-edge leak is
physical noise scattering, not Boris μ-error).

**DISCRETE RISING-TONE ELEMENTS RETURN — as a repeating train:**

- **Element 1**: coherent riser sweeping 0.25 → 0.57 Ωe0. Sweep rate with
  the exact x10/x5 pipeline (analyze_chen2026_periods.ridge_segments,
  equator probe, Bw/B0 units): slope 0.7–0.9×10⁻⁴ We0² = **0.14–0.36×
  Omura Eq. 88** (element-2 segment, r²=0.98, gives 0.36; x10/x5 measured
  0.72×) — same order, ~2× lower prefactor in this weaker-gradient,
  cone-consistent configuration. Amplitude-controlled law still holds.
  CAUTION (methodology): the ridge at the ±25° probes appears ~3× steeper
  (2.6×10⁻⁴) — that is propagation re-shaping (dispersive arrival-time
  ordering), not the source chirp; sweep rates must be measured at the
  generation-region (equator) probe. [Corrects an earlier quick-fit claim
  of 1.1–1.6× that mixed raw |Bw| into the Bw/B0 slot and the 25° slope
  with the equator amplitude.] Convectively amplified 1.9×10⁻³ (equator)
  → 8.4×10⁻³ (25°) — never reaches B_opt (fuel-limited).
- **Train structure**: WB peak 6.5×10⁻³ at t≈4200 → deep quiet at
  1.0×10⁻⁴ (noise level) t≈7200 → second element rising (8.4×10⁻⁴ by
  t≈9600). Repetition period ~5500–6000/We0 ≈ T_b (6250/We0) — the
  refill/bounce clock, as in the x5 complexes. N and S hemispheres run
  independently-timed elements (float-atomic event independence).
- Poynting outward 16/16 windows (equatorial source); x–t chevron packets
  terminate cleanly at the ±40° wall, no reflected striations (in-run
  absorber check PASS despite the layer sitting in f_ce > f_pe plasma).
- **Fuel gauge (ckpt t=9750/We0 vs load): equatorial A = +0.761 vs +0.747 —
  MAINTAINED, not burned** (contrast x2's burn-through +0.98 → +0.48). The
  bounce/refill dynamics restores the equatorial free energy between
  elements: the train is refill-gated, not fuel-exhausted. Standing
  loss-cone flux population: 5.8% (eq) / 16% (box, incl. ghosts).
- **Known long-run limitation:** the fixed-N u⊥→0 return accumulates a μ=0
  ghost pool — 6.6% of markers by t≈10⁴/We0 (each precipitation event adds
  one ballistic u∥ marker that never leaves). Negligible at x2 (7×10⁻⁴,
  tiny cone, short run); material bookkeeping here. The refresh-bath redraw
  (fate option ii, §4) is the designed fix when trains much longer than
  ~2 T_b are the target.

## 10. x4-atmo40 vs the x10 GIANT case-2 (chen2026_case2_giant / giant_e12)

The two "element-producing" configurations, side by side — every artificial
element of the x10 setup replaced by a physically-motivated one:

| | x10 giant case2 (hybrid) | x4-atmo40 (this work) |
|---|---|---|
| lre | 1330.504 (x10 compressed) | 3326.26 (x4) — 2.5× weaker gradient |
| wall | ±26.3°, b=2.38 | ±40.0°, b=7.41 |
| eq. loss cone | 44.2° — artifact of u⊥ carving at 23.6–26.3° | 21.6° — set by wall mirror ratio, sin²α=1/b_wall |
| how the cone acts | hybrid: EVERY wall-toucher u⊥-stripped each pass (exp mask), returned as cold u∥ beam | atmo: wall-touchers (mirror point beyond wall by construction) precipitate once, u⊥→0 |
| f0 | Chen subtracted bi-Max (ρ=1, κ=0.3): u⊥ hole ~34°, NOT matched to any surface | cone-cut bi-Max: trapped population only, load ≡ boundary criterion (self-consistent) |
| initial transient | boundary re-carves the mismatched hole for ~1 bounce | none (zero mismatch) |
| wave absorber | Umeda mask (nd=300) + the u⊥ carving doubling as "absorber" | Umeda mask (nd=500) on fields+cold fluid only |
| B_th(0.25) / noise | 1.24×10⁻³ / ~7×10⁻⁴: threshold ABOVE noise → true gate | 3.2×10⁻⁵ / ~1.2×10⁻⁴: threshold at noise; nonlinear gating still operates dynamically |
| ppc / markers | 110000 / 550M | 28000 / 543M |
| **elements** | threshold-gated single risers, sat ~3.6–6× B_th (4.5–7×10⁻³) | **riser train, period ≈ T_b; 0.25→0.57 sweep; eq 1.9×10⁻³ → 25° 8.4×10⁻³** |
| sweep vs Eq.88 | 0.72–0.73× | 1.1–1.6× (same amplitude-proportional law) |
| boundary role in elements | ESSENTIAL-looking (carve26: moving the carve band 1.17×→4.72× B_th; γ-valve keeps growth alive) | **NOT required**: no carving anywhere, elements return anyway |
| generation site | equatorial (Poynting) but window (~320/We0) < element time → boundary influence never excluded | equatorial AND causally clean (window 1240/We0) |

**Bottom line.** The x10 giant's chorus elements survive the removal of every
boundary artifact they were suspected of depending on: with a 2.5× weaker
gradient, a wall at its Boris-limit distance, the loss cone set honestly by
the wall mirror ratio, a trapped-only self-consistent f0, and precipitation
only of genuinely loss-cone particles — discrete rising-tone elements
reappear as a bounce-period train obeying the same amplitude-controlled
sweep law. The x10 hybrid's u⊥-carving valve is a compressed-box crutch that
substitutes for the threshold physics a weaker gradient provides naturally;
it is not the mechanism of chorus. Combined with the x2 flood result, the
lre scan closes: **gradient strength (through B_th vs noise/B_opt ordering)
selects suppressed / discrete-riser / complex / hiss regimes; the equatorial
anisotropy generates; the boundary, when honest, only precipitates.**
