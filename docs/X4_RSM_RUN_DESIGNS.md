# The x4 / x4+k⊥ run pair — concrete designs

The two-arm experiment at the heart of the gap program: one honest-boundary
chorus engine, run **without** (x4) and **with** (x4-RSM) the single oblique
k⊥ channel. Everything else is identical, so any difference between the arms
is attributable to E∥ physics. This document is the run-level specification;
the machinery lives in three sibling docs:

- `ATMO_BOUNDARY_DESIGN.md` — wall/loss-cone/loader design (shared core)
- `RSM_KPERP_IMPLEMENTATION.md` — how the k⊥ channel is built (spectral m=±1)
- `X4_RSM_EPAR_PLAN.md` — the physics plan, trapping budget, predictions

Decks: `giant_x4_atmo40.ini` (control) / `giant_x4_atmo40_rsm.ini` (RSM);
Arm-U variants `giant_x4_u2band_{ctrl,rsm}.ini` (§5).

---

## 1. Common core (identical in both arms)

| Parameter | Value | Why this number |
|---|---|---|
| lre | 3326.26 | x4 compression (= 13305.04/4): B_th(0.25 Ωe) = 3.2e-5 ≈ noise floor → structured-riser regime (between x5's complexes and x2's flood) |
| nx × Lx | 19392 × 5041.92 c/ωpe | wall at s = ±2520.96 = **λ = 40.000°** |
| Wall latitude 40° | b_wall = 7.4057 | Boris comfort limit: Ω_ce,local·dt = 0.222 ≤ 0.3; cost of going realistic (63°) ∝ 1/α_lc² is dt/23 — unaffordable |
| Loss cone | α_LC,eq = 21.6° | set by the wall alone: sin²α = 1/b_wall (user directive: the wall IS the equatorial loss cone) |
| Boundary | x = atmo, batm = 7.4057, nd = 500, numax = 0.1 | precipitate iff mirror point beyond wall (sin²α_loc < b_loc/batm) → u⊥→0 return; batm = b_wall makes boundary ≡ loader (zero transient); u⊥ otherwise KEPT (no hybrid carving crutch) |
| Hot f0 | dist = conecut, cone_b = 7.4057 | trapped-population-only bi-Max: f = biMax·Θ(sin²α_eq − 1/b_wall); load criterion is the same operator as the boundary |
| uth (x4 arms) | 0.19783565 ∥ / 0.24390817 ⊥ | Chen-lineage hot electrons; T⊥/T∥ = 1.52 (see §5 for what this forbids) |
| n_h / cold | 0.0178 / cold_nc 0.9822 (fluid) | Chen case-2 lineage; cold is linearized fluid → no cold Landau (physically right: real cold electrons are far below the 0.1c resonance) |
| ppc | 28000 (377 M markers after conecut taper) | plateau window [0.087,0.100]c is ~7% wide in v — needs full-f statistics |
| dt / nsteps | 0.15 / 333333 | t_end = 10⁴/Ωe0; dt set by Ω_ce(wall)·dt ≤ 0.3 |
| jfilter / tile_sort | 3 / 25 | tile_sort=25 runs the tiled deposit in BOTH arms (RSM-tiled kernel landed 2026-07-31: 1.51e10 p-steps/s at full scale, 8× over the flat path) |
| seed | 20260720 | shared → same load, same noise realization (float atomics still decohere after t ~ 2000) |
| Probes | 21, at 0, ±2.5°…±35° arcs | sweep-rate & spectra ONLY at the equator probe (ae34e12 lesson: high-lat probes steepen apparent chirp ~3×) |

## 2. Design 1 — x4-atmo40 (parallel-only control)

Strictly ny = 1, k⊥ ≡ 0: every wave is a parallel whistler, **E∥ ≡ 0 by
construction** — Landau resonance impossible. This arm established
(2026-07-30): discrete rising-tone element trains recover under a fully
honest boundary (5+ elements, N/S alternating, steady period T_b/2,
A maintained +0.75→+0.86 by the wall valve, sat ≈ B_opt). It is the
free control for the RSM arm: Ly is inert at ny = 1, so no rerun was needed.

## 3. Design 2 — x4-atmo40-RSM (+one k⊥ channel)

Exactly three deck deltas on top of §1:

| Delta | Value | Why |
|---|---|---|
| Ly | 39.269908169872 | = 2π/k1, the RSM phase contract (particle y ≡ θ/2π) |
| [rsm] k1 | 0.16 ωpe/c | θ_WNA(ω) = atan(k1/k∥): 15.5° at chirp onset (ω=0.25, k∥=0.577) → 9.1° at 0.5 (k∥=1.0). Li 2019's 15° at onset; E∥/E rises 6.5%→11.3% toward 0.5 (Stix, verified in-run) |
| deposit path | rsm-tiled kernel (auto when tile_sort>0 ∧ rsm) | shared m0 tile + 1-D complex m1 tile |

Extra state: complex 1-D lines E1/B1/J1(x) (+cold twin vc1), O(nx) memory.
Extra outputs: `m1line_*.bin` = (B1y, B1z, E1x = E∥, E1y) complex lines
every 100 steps. No seed injection: the m1 channel must self-excite from
particle noise off the same anisotropy (it did: 2|B1|rms/B0 → 8.5e-3 ≈
0.8·B_opt by t ~ 5000).

Key structural fact for interpretation: the m0 and m1 systems have **no
direct field coupling** — they interact only through the shared particles.
E∥ lives only in m1. So: plateau and P1-spectrum effects are direct;
any P0 (parallel chirp) response is particle-mediated back-reaction.

## 4. Analysis protocol (pre-registered discriminants)

1. **P0 / P1 modal spectrograms judged separately**, equator probe,
   peak-local windows, never box-averaged (V6 rule; comb valley ≠ gap).
2. **f(v∥) plateau**: equator window |x−x_eq| < 200 c/ωpe, 0.02 < u⊥ < 0.35
   (excludes the u⊥=0 ghost line), discriminant = *localized* slope
   reduction in |v∥| ∈ [0.087, 0.100] c (= v_res swept by a 0.25→0.5 chirp;
   Vp max at 0.5), **± symmetric** (N/S alternating elements). Control must
   show a smooth monotonic profile (Level-1 corrected criteria).
3. **ω_stop statistics** (`scripts/x4_omega_stop.py`): per-element ridge
   termination frequency — the gap-as-statistics test, robust to
   element-to-element variance.
4. **The 0.5 coincidence** (mechanism bookkeeping): at ω = Ωe/2,
   v_Landau = ω/k∥ and |v_cyc| = (Ωe−ω)/k∥ both equal 0.100 c — the m1
   plateau can gate both channels (Landau-damps P1 near 0.5 AND starves
   P0's cyclotron drive approaching 0.5).

## 5. Arm U — the two-band-capable f0 pair (2026-07-31 pivot)

The x4 f0 caps cyclotron growth at ω_m = A/(1+A): measured A = 0.75–0.88 →
ω_m = 0.43–0.47, so the **upper band is forbidden by construction** (user's
catch). A Li2019-style two-band gap therefore cannot appear in §2/§3 — only
a truncation edge at ~0.5. Arm U removes that cap with ONE delta on §1:

| | x4 arms | Arm U pair |
|---|---|---|
| uth⊥ | 0.24390817 | **0.31283** |
| T⊥/T∥ | 1.52 | 2.5 |
| A (load) | 0.52 | 1.5 |
| marginal ω/Ωe | 0.43–0.47 (maintained) | **0.60** |
| linear spectrum | ≤ 0.47, single band | **0.2–0.6 continuous (gapless)** |

Resonant-population geometry that makes the test clean: upper-band
cyclotron electrons sit at v∥ ≈ 0.065 c, lower-band at 0.15–0.26 c, and the
m1 Landau plateau [0.087, 0.100] c falls **between them**.

Pre-registered predictions (also in the deck headers):
- **u2band_ctrl** (no k⊥): filled 0.2–0.6 spectrum, **no gap**. If the
  control gaps on its own, the attribution fails — this arm is as important
  as the other.
- **u2band_rsm**: 0.5 valley carved between two bands = the textbook gap
  demonstration (same f0, same boundary, k⊥ switch = gap switch).

## 6. Status snapshot (2026-07-31 night)

- x4 pair complete: plateau discriminant fired **only** in RSM (slope 1.10
  vs control 3.03 in the Landau window, both signs, ~2.5σ/side); P0 power
  terminates 0.44–0.46 in RSM vs control extending through 0.5 to ~0.58;
  m1 self-excited to 0.8·B_opt. Single pair — ensemble (4 reseeds) designed,
  preempted by Arm U, re-runnable.
- Arm U pair launched (u2band_rsm → u2band_ctrl chained, ~7 h total).

## 7. Run commands

```
./chirp2d decks/giant_x4_atmo40.ini      <out> --ckpt=25000  --noeline   # control
./chirp2d decks/giant_x4_atmo40_rsm.ini  <out> --ckpt=25000  --noeline   # +k_perp
./chirp2d decks/giant_x4_u2band_ctrl.ini <out> --ckpt=111111 --noeline   # Arm U ctrl
./chirp2d decks/giant_x4_u2band_rsm.ini  <out> --ckpt=111111 --noeline   # Arm U rsm
python3 scripts/x4_omega_stop.py <out> [<out> ...]                       # ω_stop table
```
