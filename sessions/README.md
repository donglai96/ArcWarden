# Session 2026-06-28 — Darwin EM build + whistler-chorus reproduction

Saved record of the working session (`b58515dc`).

## Files
- **`2026-06-28_darwin_whistler_dialog.md`** — readable dialog (52 user turns, 236
  assistant turns; user verbatim, assistant spoken text + compact tool list).
- **`2026-06-28_darwin_whistler_raw.jsonl`** — full raw transcript (tool inputs/outputs,
  timestamps; 11 MB).

## What we did (arc of the session)
1. **Diagnosed "2D GPU not working"** — the `SharedTileDeposit` (whole-grid privatization)
   silently falls back to global atomics in 2D (deposit = 63% of GPU time).
2. **Tiled+binned deposit** — fixed-size spatial tile + coarse per-tile counting sort
   (`depositor.hpp`, `particles.hpp`); 1.6–3.4× in the production 2D regime, validated.
3. **Tested OSIRIS-style shared-field gather in the push** — measured it a *net loss* in
   2-component ES (fields already L2-resident; bin indirection hurts coalescing).
4. **Built the full spectral Darwin EM model** (planned + implemented in 6 phases):
   J/dcu/amu deposits, B-from-J solve, transverse inductive **E_T** with the resummed UPIC
   `ffe` Green's function and the `ndc` iteration, magnetized fused push, chunk-pool sort.
   - GATE 1 (magnetostatic B-from-J) and GATE 2 (Weibel growth rate) both pass. CTest 12/12.
5. **Reproduced An et al. (2019) "Unified view of nonlinear wave structures…
   whistler-mode chorus" (arXiv:1901.00953), Simulation 1** — added the external pump-field
   driver + background B0 in the push (`pump.hpp`); drove the whistler to δB/B₀≈0.1 and saw
   the trapped resonant island at v∥≈3.24 v_th excite **Langmuir waves (modes 300–400)**.
   - Found & fixed a real bug: the transverse-E solve needs the resummed Green's function
     `green_et = 1/(ε₀c²k² + n₀)` (the naive μ₀/k² diverges → NaN).
   - Found & fixed a field-decomposition error: δE⊥ must be the pure transverse field (E_y),
     not a projection mixing in the noisy longitudinal Eₓ.
   - Diagnostics reproduce the paper's Fig 2: **k–t spectrum** (2a), **ω–k dispersion** (2b),
     **field + phase-space portrait** (2c–d), plus a **time-evolution video** to t=3000
     (stable, physical late-time quasilinear relaxation — no blow-up).

## Key code added/changed
`include/pic/{config,sources,fields,spectral,depositor,particles,pusher,simulation}.hpp`,
new `include/pic/{solver_darwin,pump}.hpp`; `tools/whistler_pump.cu`, `tools/deposit_bench.cu`;
`tests/test_darwin_magnetostatic.cu`, `tests/test_weibel.cu`;
`scripts/plot_whistler_{kt,dispersion,video,sim1}.py`.

## Status / open items
- Sims 2 & 3 (v_r/v_th = 2.1 unipolar, 1.0 bipolar) not yet run.
- Whistler drive needed amp ×6 over Table-I (off-resonant; tune ω₀/M to our dispersion).
- Late-time heating looks physical; a total-energy-conservation trace would confirm.
- All `whistler_*` artifacts + this raw jsonl are large; consider gitignoring before commit.
