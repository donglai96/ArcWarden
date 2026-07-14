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

## Reading for M1/M9 planning

- The 2D EM ρ+J deposit at 4.9 G/s is the number to beat; Esirkepov (M1) will
  be heavier still — budget accordingly. The plan's M9 target of ≥3×10⁸
  particle-STEPS/s end-to-end on the full 2D Maxwell loop implies the whole
  step (gather+push+Esirkepov+field update) must stay under ~1.4 ms per 4.2M
  cells × 100 ppc block — deposit is the pacing item, exactly as §13 assumed.
- 1D chirp1d numbers show the fused-kernel + tiny-grid-in-L2 ceiling
  (~4×10⁹): an upper bound, not a 2D promise.
