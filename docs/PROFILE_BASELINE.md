# Performance baseline (L-shell plan M0) — RTX 5090, CUDA 13.3, 2026-07-13

Reference point for every later optimization PR (plan discipline: each M9 item
needs before/after against this). Entry points: `deposit_bench` (256²grid,
ppc 64, 16.8M markers unless noted) and the 1D `chirp1d` end-to-end runs.

## deposit_bench (2D spectral path)

| kernel | time | throughput | note |
|---|---|---|---|
| deposit tiled (ES ρ) | 0.275 ms | 14.9 G dep/s | 6.41× vs global atomics |
| push global (ES, B0 rot) | 0.116 ms | 35.4 G push/s | bandwidth-bound |
| push shared-field | 0.141 ms | 29.0 G/s | 0.82× — NOT a win (known) |
| deposit ρ+J (EM, unsorted) | 0.831 ms | 4.9 G/s | **EM bottleneck** |
| deposit ρ+J (EM, sorted) | 0.853 ms | 4.8 G/s | sort not paying at ppc 64 |
| EM fused push | 0.130 ms | 31.5 G/s | 6-component gather |
| Darwin solve (E_L+B) | 0.036 ms | — | spectral solve is negligible |

Correctness cross-checks in the bench: all deposit/push variants agree to
float roundoff (≤3.3e-6 rel).

## chirp1d (1D electron-hybrid, end-to-end incl. diagnostics)

| run | markers | steps | rate |
|---|---|---|---|
| full-f Tao setup | 11.4 M | 500 k | 4.0–4.4×10⁹ particle-steps/s |
| δf + antenna | 11.4 M | 250 k | 3.2×10⁹ (weight update + wd traffic) |
| two concurrent runs | — | — | ~2.1×10⁹ each (fair SM sharing) |

## eaw_case7 head-to-head (2026-07-14): tiled Esirkepov (M9) vs OSIRIS-CUDA

Ma et al. PoP 2024 case 7: 625² cells, 156.25M particles (ppc 400), dt 0.0145,
206,897 steps, jfilter 3. OSIRIS 4.4.4 CUDA (prebuilt, deck-only) on the same
RTX 5090, 25×25 tiles / 512-particle chunks, quadratic + binomial-3 smoothing,
no diagnostics; ArcWarden linear CIC.

| code / path | p-steps/s | full-run wall |
|---|---|---|
| ArcWarden flat + full diagnostics | 3.9×10⁹ | 8,359 s (measured) |
| ArcWarden flat, no diag | 4.5×10⁹ | ~7,250 s |
| OSIRIS-CUDA, no diag (52% GPU util) | 5.9×10⁹ | ~5,470 s (measured to t=836, rate flat) |
| **ArcWarden tiled (`tile_sort=20`) + full diagnostics** | **1.46×10¹⁰** | **2,210 s (measured)** |
| ArcWarden tiled, no diag | 1.51×10¹⁰ | ~2,140 s |

The tiled path (yee2d.hpp `k_push_esirkepov_tiled`): physical 16×16-cell tile
sort every `tile_sort` steps (existing chunk-sort), fused gather(L2)/Boris/
Esirkepov-into-shared-apron; strays beyond the DRIFT=2 apron fall back to
global atomics (always correct; gated by the `yee_tiled` ctest). Sort cadence
10/20/40 → 1.45/1.51/1.54×10¹⁰ on this problem. Deposit was ~20 global
atomics/particle; now shared atomics + a ~2k-cell apron flush per tile,
ppc-free. Physics parity on the full case-7 run: WB identical to 4 sig figs
through the linear phase, WNA 46.5°/44.5° vs flat 46.4°/44.3°.
Next dials if needed: 32×16 tiles, warp aggregation, TMA bulk flushes (sm_120).

### 2026-07-15 confirmation: full runs, MATCHED diagnostics, same figure

Both codes ran case 7 end-to-end with equivalent diagnostics (6-field 2D dumps
every 50/ωpe + x1 lineouts every 1/ωpe), figures produced by the same 8-panel
pipeline (`plot_eaw_case7.py` / `plot_eaw_case7_osiris.py`, ZDF reader from the
OSIRIS source tree — that build has no HDF5):

| code | wall (measured) | p-steps/s |
|---|---|---|
| ArcWarden tiled + migrate fusion (4825499) | **2,079.7 s** | 1.55×10¹⁰ |
| OSIRIS-CUDA (quadratic + smooth) | 5,392.8 s | ~6.0×10⁹ |

2.59× faster; physics identical (whistler kx peak 2.79 both, WNA 44–47° both,
EAW band 4–7× noise both; saturation t≈840 vs ≈1300 — shot-noise seed level
only, the quieter quadratic load starts lower). Comparison figure:
`docs/figs/eaw_case7_compare_arcwarden_osiris.png`. OSIRIS deck/data:
`~/Donglai_Ma_reborn/osiris_case7_bench/`.

## Reading for M1/M9 planning

- The 2D EM ρ+J deposit at 4.9 G/s is the number to beat; Esirkepov (M1) will
  be heavier still — budget accordingly. The plan's M9 target of ≥3×10⁸
  particle-STEPS/s end-to-end on the full 2D Maxwell loop implies the whole
  step (gather+push+Esirkepov+field update) must stay under ~1.4 ms per 4.2M
  cells × 100 ppc block — deposit is the pacing item, exactly as §13 assumed.
- 1D chirp1d numbers show the fused-kernel + tiny-grid-in-L2 ceiling
  (~4×10⁹): an upper bound, not a 2D promise.
