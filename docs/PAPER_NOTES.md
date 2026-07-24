# Paper notes (side project): ArcWarden — a GPU-native PIC for new-architecture hardware

Working thesis: **a particle-in-cell code designed around what modern GPUs
actually provide — a cache hierarchy large enough to hold the entire field
grid, fast shared-memory atomics, and cheap wide kernels — outperforms the
classic GPU-PIC chunk-pipeline design by 2.6× on identical physics**, using
fully fused step kernels and amortized, stale-tolerant particle sorting in
place of per-step particle bookkeeping.

Benchmark platform: NVIDIA RTX 5090 (Blackwell, sm_120, ~100 MB L2, 32 GB),
CUDA 13.3. Reference code: OSIRIS 4.4.4 CUDA (tiles + 512-particle chunk pool).

## Results already in hand (2026-07-14/15)

1. **Head-to-head, identical physics, matched diagnostics** — Ma et al. PoP
   2024 case 7 (whistler anisotropy instability → oblique whistlers + EAW/TDS,
   625², 156.25M particles, 206,897 steps):
   - ArcWarden (tiled, fused): **2,079.7 s** (1.55×10¹⁰ particle-steps/s)
   - OSIRIS-CUDA (quadratic + smooth): **5,392.8 s** (~6.0×10⁹) → **2.59×**
   - Same 8-panel figure from both codes (`plot_eaw_case7.py` /
     `plot_eaw_case7_osiris.py`); physics agreement: whistler kx peak 2.79
     both, WNA 44–47° both, EAW band 4–7× noise both, beam-mode ridge on
     v_ph = 0.0546c both. Figure: `figs/eaw_case7_compare_arcwarden_osiris.png`.

2. **Ablation chain** (case-7 config, no diagnostics):
   | step | rate |
   |---|---|
   | flat global-atomic Esirkepov | 4.5×10⁹ |
   | + 16×16 tile sort (every 20) + shared-apron deposit | 1.51×10¹⁰ |
   | + migrate fused into write-back | **1.61×10¹⁰** |

3. **The reference code is overhead-bound, not compute-bound** (three
   independent measurements):
   - GPU utilization 52% during the OSIRIS run;
   - linear vs quadratic interpolation changes its rate by only 7%
     (6.3 vs 5.9×10⁹) despite ~2× deposit flops;
   - chunk_size 512 → 1024 is an exact null (5.92×10⁹ both) — the cost is the
     per-step find-movers/exchange/compact pipeline itself, which scales with
     particles and pipeline stages, not with chunk granularity.

4. **Design elements** (the paper's methods section):
   - one fused kernel per step: staggered gather + Boris + move + Esirkepov
     scatter + periodic wrap + cell recompute (4–5 wide kernels per step
     total, no gaps → no utilization loss);
   - gathers served from L2 (the whole 6-component 625² Yee grid ≈ 14 MB
     ≪ 100 MB L2) — no shared-memory field staging needed at all;
   - deposit into shared-memory tile aprons (DRIFT=2), replacing ~20 global
     atomics/particle with shared atomics + a ~2k-cell flush per tile
     (ppc-independent);
   - sorting is amortized and stale-tolerant: physical counting sort every
     N steps (12.4 ms → 0.6 ms/step at N=20; rate flat for N=10–40); strays
     that out-drift the apron take a global-atomic fallback, so correctness
     never depends on sort recency (gated: J identity 2×10⁻⁷, continuity
     4.6×10⁻⁶ with 4-step-stale sort);
   - correctness gates as first-class: charge-conservation ctest on the
     tiled path incl. stray fallback; energy-history parity with the flat
     path to 4×10⁻⁷.

## Darwin branch measurement (2026-07-16)

Case-7-scale on the Darwin spectral branch (`decks/darwin_bench.ini`: 625²,
156.25M particles, same anisotropic load, dt = 0.1 — no light CFL):
- **5.0×10⁹ p-steps/s** (31.4 ms/step, marginal 150→550 steps) with the
  pre-tiling flat deposit — matches flat-Yee 4.5×10⁹ + spectral solves ≈ free
  (36 µs).
- Time-to-solution for t=3000/ωpe: 30,000 steps ≈ **943 s** → 2.2× cheaper
  than tiled Yee (2,080 s), 5.7× than OSIRIS (5,393 s). Model beats kernels.
- GPU utilization 98–100% @ ~485 W once stepping (`profiling/darwin_util.txt`).
- Kernel budget (nsys, ms/step of 31.4): **per-step tile sort 11.6 (36%!)**,
  push 8.0, ρ+J 3.4, amu 3.3, dcu 3.0, migrate 2.1, all spectral solves 0.17.
  The Darwin branch still sorts EVERY step (pre-M9 design) — adopting the
  amortized stale-tolerant sort + fused migrate projects ~18 ms/step
  (8.7e9 p-steps/s, ~550 s for case-7 physics, ~10× OSIRIS end-to-end).
  This is the top implementation TODO for the paper.

## Paper draft status (2026-07-16)

Full LaTeX draft lives in `paper/main.tex` + `paper/refs.bib` (compile on
Overleaf/tectonic; no local TeX). Sections written: intro, models (Darwin/
Yee/hybrid), GPU design (4 rules), validation (case 7 + chirping), perf
(head-to-head, ablation, overhead-bound, Darwin TTS, profiling), discussion,
**LLM-agent development-methodology section**, conclusions. Open items are
`\tocheck{}` markers in the tex. Profiling artifacts in `profiling/`
(nsys-rep for GUI; ncu pending sudo).

## Still needed before writing

- [ ] Scaling curves: rate vs ppc (100…4096) and vs grid (256²…2048²), both codes
- [ ] 2–3 more cases from the 32-case EAW_ECH scan (different β∥, anisotropy)
- [ ] Long-run energy conservation figure (both codes)
- [ ] Roofline / ncu section (needs sudo for GPU perf counters)
- [ ] Optional: quadratic (S2) shapes in the tiled framework for exact shape
      parity (est. 20–30% cost) — strengthens the comparison
- [ ] Clean single-session ablation rerun (same binary, same night)
- [ ] Related-work scan: WarpX / PIConGPU / VPIC numbers on comparable problems
      (cite, don't necessarily benchmark)

Venue candidates: Computer Physics Communications, J. Comput. Phys.
Data locations: `build/eaw_case7*/`, `~/Donglai_Ma_reborn/osiris_case7_bench/`.
