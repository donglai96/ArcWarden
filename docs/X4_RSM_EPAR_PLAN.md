# x4-atmo40 × RSM: adding E∥ (one k_perp channel) to the honest-boundary chorus run

**Goal.** The x4-atmo40 run (2026-07-30, commit `ee586a4..7b9eb01`) produces a
sustained discrete rising-tone element train with an honest wall/loss-cone.
Every wave in that box is **strictly parallel** (ny = 1, k_perp = 0) — so
E∥ ≡ 0, Landau resonance is impossible, and no v∥-plateau / 0.5 fce gap can
form *by construction*. This plan adds the one-k_perp oblique channel (RSM
m = ±1 spectral harmonic, `rsm_oblique.hpp`, V1–V5 all gated PASS) to the x4
deck, so the self-generated chirping elements can Landau-process the hot
distribution. Question: does a v∥ plateau form, and does a 0.5 fce feature
appear in the modal spectra?

This is exactly the flagship RSM hypothesis (docs/RSM_MODEL_DEFINITION.md)
run on our best chorus engine instead of the Chen case-2 base.

---

## 1. Physics: why this can work, quantitatively

Parameters: ωpe = 1, Ωe0 = 0.2 ωpe (fpe/fce = 5), hot n_h = 1.78 %,
uth∥ = 0.198 c, A ≈ 0.75–0.86 sustained (the honest wall keeps pumping
anisotropy — free energy is NOT exhausted, so the oblique channel has fuel
for the whole run).

**Resonance geometry (the Vp-max@0.5 anchor).** Parallel-whistler phase
velocity Vp(ω) ≈ c·√(ω(Ωe−ω))/ωpe peaks at ω = Ωe/2:

| ω/Ωe | k∥ (ωpe/c) | v_res = ω/k∥ | θ_WNA at k1 = 0.16 |
|------|-----------|--------------|--------------------|
| 0.25 | 0.577     | 0.087 c      | 15.5° |
| 0.40 | 0.816     | 0.098 c      | 11.1° |
| 0.50 | 1.000     | 0.100 c      | 9.1°  |

- Landau resonance window swept by a 0.25→0.5 chirp: **v∥ ∈ [0.087, 0.100] c**
  — a narrow band at 0.44–0.51 uth∥ where f(v∥) is large (≈ 0.88 of peak)
  with healthy negative slope. Plenty of resonant particles.
- v_res(ω) is *maximal* at 0.5 fce: any plateau carved by the chirping
  lower band terminates exactly at the resonance velocity of the 0.5 fce
  wave → the ω → 0.5 wave sits on the restored (steepened) edge of the
  plateau → strongest Landau damping at 0.5. This is the Li 2019 NatComm
  mechanism, which we already reproduced in-house on Yee with a 15° tilt
  (two-band + 0.5-gap, 4 stages) — the E∥ channel is *proven sufficient*
  in our numerics; what's new here is making it **self-consistent with a
  self-generated chirping source** in the honest-boundary box.
- Elements alternate N/S propagation (period T_b/2) → both signs of v∥ get
  Landau-processed → expect **± symmetric plateaus** (Level-1 corrected
  criterion: the box is ±symmetric; the discriminant is *localized
  flattening at |v∥| ≈ 0.06–0.14 c*, never ±asymmetry).

**Trapping-time budget (the honest uncertainty).** Exact Stix polarization
of the cold oblique eigenmode at k1 = 0.16 (computed 2026-07-31; the 3 %
first estimate was conservative):

| ω/Ωe | θ | E∥/E_total |
|------|-----|-----------|
| 0.25 | 15.6° | 6.5 % |
| 0.40 | 11.1° | 9.2 % |
| 0.50 | 9.0°  | 11.3 % |

E∥/E *rises* toward 0.5 fce — the Landau channel strengthens exactly where
the gap should form. (Verified dynamically: the x4_rsm_smoke m1 spectral
peak measured E∥/E1y = 0.088 at ω ≈ 0.3–0.4 Ωe, inside the Stix range;
V2 gate had matched the eigenmode ratio to ≤ 1.1 %.)
If the m = 1 channel reaches Bw1 ~ 5×10⁻³ B0 (half the m0 saturation
B_opt ≈ 1.05×10⁻²), with η∥ ≈ 8 %:

- eE∥/m ≈ Vp·Bw1·η∥ ≈ 0.09 × 1×10⁻³ × 0.08 ≈ 7.2×10⁻⁶ (c·ωpe units)
- ω_tr = √(k∥·eE∥/m) ≈ 2.4×10⁻³ ωpe → τ_tr ≈ 524 /Ωe0

One element lasts ~1000–2000 /Ωe0 → **1–3 trapping periods per element
passage**; the train delivers 5+ elements per 10⁴/Ωe0 → plateau formation
is *cumulative and borderline-feasible* at that amplitude. If m = 1 only
reaches noise-grown 10⁻⁴ B0, ω_tr drops ~2.5×… → no plateau in t10⁴.
**The m = 1 saturation amplitude is the single controlling unknown.**
γ_oblique(15°)/γ_par ≈ 0.8–0.9 at A ~ 0.8, and A stays high all run, so
the oblique mode should grow toward a comparable saturation — but it
competes with m0 for the same resonant particles. Mitigation: `rsm_seed`
eigenmode injection arm if the noise-grown arm stalls.

**One conceptual point to keep straight (prediction discipline).**
E∥ lives *only* in the m = 1 channel. Landau damping therefore acts on the
m = 1 spectrum directly; the parallel m0 elements have no E∥ and cannot be
Landau-damped. So the clean predictions are:

1. **P1 spectrum**: 0.5 fce gap (waves near 0.5 damped by the plateau edge).
2. **f(v∥)**: localized flattening at |v∥| ∈ [0.087, 0.100] c (equator
   window), absent in the control.
3. **P0 spectrum** (the m0 chirp): whether the plateau back-reacts on the
   parallel chirp (elements stopping at 0.5, step features) is the *open
   science question* — any coupling is particle-mediated. Judge P0 and P1
   **only from separate modal spectra** (V6 rule); a comb valley is not a
   gap (V6 forensics lesson).

**Why RSM and not the alternatives.**
- Explicitly forcing an E∥ term on the m0 fields: not self-consistent,
  breaks charge conservation — rejected.
- Tilting B0 (Li2019-style Yee tilt): incompatible with the dipole mirror
  axis along x — rejected.
- Full 2D ny ≫ 1: ~64× the cell count forces ppc ≪ 1000; shot noise would
  bury the narrow plateau window (07-28 lesson) — rejected for now.
- RSM m = 1: same marker count, +O(nx) complex field lines, exact ik1
  derivatives, cold-inclusive m1 Gauss correction already gated for E∥
  accuracy — the only affordable self-consistent path. This is what it
  was built for.

---

## 2. Code status audit (2026-07-31)

Already in place, no changes needed:

| Piece | Status |
|---|---|
| m = 1 spectral engine (fields, cold twin, Gauss, exact modal Esirkepov) | V1–V5 PASS |
| Dipole B0(x) + mirror mc-term in the RSM pusher | present (`rsm_oblique.hpp` ~L287) |
| Phase load θ uniform (any loader, incl. new conecut) | `rsm_theta_init` post-load, chirp2d.cu L86-89 |
| x-end damping of m1 fields + m1 cold | `k_rsm_damp_x`, generic masks, invoked for any bnd_x ≠ 0 (`simulation_maxwell.hpp` L158-166) |
| m1 complex probe lines (B1y,B1z,E1x=E∥,E1y) | chirp2d.cu L255 |
| RSM ckpt/resume, --ckptseq | V5 PASS |
| deck guards | `[rsm]` only requires ny=1 + Ly=2π/k1; no conflict with bnd_x=3 / dist=conecut |

**The one code gap:** the RSM pusher's boundary block
(`rsm_oblique.hpp` L309-323) handles specular walls and the bnd_x=2 hybrid
carve, but **not the bnd_x=3 atmo precipitation criterion** — a wall-touching
particle with mirror point beyond the atmosphere must return with u⊥ → 0.
Without this, the RSM run silently loses the loss cone (wall becomes a pure
specular mirror: no valve, no precipitation, wrong f0 maintenance).

---

## 3. Code changes

### 3.1 `include/pic/rsm_oblique.hpp` — atmo branch in the pusher (~10 lines)

Mirror the yee2d.hpp implementation exactly (refl flag + batm criterion):

```cpp
if (rp.bnd_x) {
    bool refl = false;
    if (x1 < 0.f)               { x1 = -x1;             p.ux[t] = ux = -ux; refl = true; }
    else if (x1 >= (float)v.nx) { x1 = 2.f * v.nx - x1; p.ux[t] = ux = -ux; refl = true; }
    if (x1 >= (float)v.nx) x1 = nextafterf((float)v.nx, 0.f);
    if (rp.bnd_x == 3 && refl && rp.bnd_batm > 0.0) {
        const float bl = rp.b0_prof
            ? bg::b0x(rp, x1 * (float)v.dxp) / (float)rp.B0[0] : 1.f;
        const float up2 = uy * uy + uz * uz;
        if (up2 * (float)rp.bnd_batm < bl * (up2 + ux * ux))
            { p.uy[t] = uy = 0.f; p.uz[t] = uz = 0.f; }
    }
    if (rp.bnd_x == 2) { /* existing carve unchanged */ }
}
```

Care: unlike yee2d, the RSM kernel *re-uses* uy/uz below for the deposits
(vz1 already computed from pre-boundary uz — check the yee2d ordering and
match it exactly; in yee2d the atmo zeroing happens after the push and the
deposit uses the reflected path — keep identical op order so the m0 deposit
stays bit-matched to the flat path in the rsm-off comparison). Also update
the header comment ("bnd_x rejected" → modes 1/2/3 supported).

### 3.2 `decks/giant_x4_atmo40_rsm.ini` — new deck

Copy of `giant_x4_atmo40.ini` with:

```ini
[grid]
Ly = 39.269908169872     # = 2*pi/k1 (RSM phase contract; Ly inert at ny=1
                         # for the flat path — weights carry dyp, physics
                         # unchanged, so the existing x4 run stays a valid
                         # control)
[rsm]
enable = true
k1 = 0.16                # 15 deg WNA at chirp-onset k_par ~ 0.58 (w=0.25 We0)
# seed =                 # arm 2 only (eigenmode injection if noise growth stalls)
```

Everything else identical (nx=19392, dt=0.15, atmo batm=7.4057,
conecut cone_b=7.4057, ppc=28000, seed 20260720, same 21 probes,
nsteps=333333 → t10⁴/Ωe0).

### 3.3 Optional gate deck `decks/x4_rsm_smoke.ini`

Shrunk x4 (nx=2000, Lx=520, ppc=2000, nsteps=20000) purely to exercise
rsm × atmo × conecut together on GPU before the big burn.

### 3.4 `scripts/x4_rsm_gap_analysis.py` — analysis

- **P0 vs P1 modal spectrograms** per probe (m0 real lines vs m1 complex
  lines; STFT with the Δω·Δt ≥ 2π limit in mind; peak-local, never
  box-averaged — 07-28 lesson).
- **P1 growth curve** (∫|B1|² dx vs t) + γ_oblique vs linear expectation.
- **f(v∥) equator window** from ckpts (adapt `ckpt_fuel_gauge.py` reader:
  parse_header → (step,time,man,data_start); arrays px/pux/…): |x−x_eq| <
  200 c/ωpe, u⊥ < 0.35 c band, overlay control vs RSM at matched times;
  discriminant = localized slope reduction in |v∥| ∈ [0.06, 0.14] c.
- Landau-vs-cyclotron J·E ledger split by channel (reuse G1.3 machinery
  where applicable).

---

## 4. Gates before the big run

| Gate | Test | Pass criterion |
|---|---|---|
| G-R1 build+smoke | x4_rsm_smoke on GPU | runs, no NaN, ckpt resume works |
| G-R2 atmo parity | smoke with rsm on vs off, early t | precipitation flux & cone-edge structure statistically identical (m1 still tiny) |
| G-R3 flat-path regression | `scripts/regress_case2.py` envelope method on an rsm-off rerun of the smoke | ≤ 3× statistical envelope (float atomics forbid bit-identity) |
| G-R4 m1 hygiene | smoke P1 spectrum | m1 grows from noise with whistler dispersion (not a grid mode); divB1 ~ 1e-6 as in V3 |

## 5. Run plan

1. **Arm R0 (main)**: `giant_x4_atmo40_rsm.ini`, t10⁴/Ωe0.
   Cost: markers unchanged (543 M, 21.2 GB); RSM adds per-particle m1
   gather+deposit and forces the GlobalJSink (non-tiled) m0 deposit →
   expect ~1.5–2× the x4 wall-clock (≈ 1.5–2 days). Disk ~similar (~25 GB
   + m1 probe lines).
2. **Control**: the existing `build/giant_x4_atmo40` run (same seed, same
   physics — Ly is inert for the flat path) = free control. Only rerun if
   G-R3 flags a discrepancy.
3. **Arm R1 (contingency)**: if P1 stalls < 10⁻³ B0 by t ~ 4000, restart
   with `rsm_seed` eigenmode injection (amplitude ~ 10⁻³ wce, the Li2019
   posture: oblique waves present, question becomes plateau + gap rather
   than oblique self-excitation).

## 6. Decision tree / expected outcomes

| Outcome | Reading |
|---|---|
| P1 saturates ~mB_opt, f(v∥) plateau in window, P1 gap at 0.5 | RSM hypothesis core confirmed on the honest-boundary engine → flagship result |
| Plateau forms, P1 gap forms, **P0 chirp also stalls/steps at 0.5** | strongest possible result: particle-mediated back-reaction of the Landau plateau on the parallel chirp — new physics claim |
| P1 grows but no plateau (ω_tr too slow) | amplitude-budget failure → Arm R1 seed injection |
| P1 never grows | oblique competition loss vs m0 → measure γ_oblique, revisit k1 (θ scan 10–20°) |
| Plateau but no P1 gap | falsifies the Vp-max@0.5 edge mechanism in this geometry — report as-is |

## 7. Risks

- **Narrow window vs shot noise**: [0.087, 0.100] c is ~7 % in v; full-f
  ppc=28000 with a wide equator window should resolve it (unlike the
  07-28 phase-hole case which needed δf) — but keep the window wide in x
  and integrate over the whole element train.
- **m0↔m1 deposit cost**: if wall-clock explodes (> 2.5× x4), drop ppc to
  20000 (≙ 388 M) — noise floor rises √1.4, still ≪ B_th(x4)=3.2e-5 regime
  change.
- **Ghost pool** (u⊥=0 returns, 8.9 % @ t2×10⁴ in x4): unchanged by RSM;
  irrelevant for t10⁴ but note the ghosts sit at u⊥=0, far from the
  Landau window at u⊥~thermal — no contamination of the plateau
  diagnostic if the f(v∥) window excludes u⊥ < 0.02 c.
- **dipole ω/Ωe drift along x**: the 0.5 gap is a 0.5 fce_eq feature;
  measure spectra at the equator probe (the sweep-rate lesson: high-lat
  probes distort — ae34e12).
