# Deck-only reproduction status (branch `paper/cpc-cleanup`)

Audit of 2026-08-12: can every paper result be rerun on this branch with
**no code changes, decks only**? Answer: yes for everything ArcWarden-side;
verified by build + smoke runs on the RTX 5090 (driver 610.43.02).

## Verified on this branch (2026-08-12)

| Paper result | Command | Verification |
|---|---|---|
| §4 verification suite | `ctest` (build dir) | **32/32 passed** (207 s total) |
| §5.1 benchmark, tiled path | `./eaw2d_yee decks/benchmarks/eaw_case7_tiled.ini out` | smoke `--tend=15`: **1.69e10** particle-steps/s (paper no-diag: 1.61e10) |
| §5.2 ablation, flat path | `./eaw2d_yee decks/benchmarks/eaw_case7.ini out` | smoke `--tend=15`: **5.21e9** → tiled/flat = 3.2x (paper: 3.4x) |
| ablation is deck-only | `diff decks/benchmarks/eaw_case7{,_tiled}.ini` | one line: `tile_sort = 20` |
| path equivalence | field energies at t=14.9 | WE = 5.910e-06 both paths |
| §2 two-stream smoke | `./arcsim decks/verification/two_stream.ini out` | runs, phase frames + energy out |

Not rerun (long): full case-7 (35 min), An et al. production runs, chirping
production runs — all deck-driven, launch when needed. Every runner writes
`run_meta` (full command line) into the output dir, so provenance is automatic.

## Deck/CLI knob map (already runtime, no recompile)

- `[field] model = es | darwin | yee` — field solver branch
- `[field] tile_sort = 0` (flat global-atomic) `| N` (tiled + sort every N) — the ablation dial
- `[field] jfilter = N` — binomial current-filter passes (OSIRIS `smooth` parity)
- `[pump] ex0/ey0/ez0, mode, w0` — wave driver; `--amp=X` CLI multiplier
- per-species `ppc`, `uth`, `ufl`, `density`; `[plasma] noisy/seed`
- CLI overrides: `--ppc --amp --nsteps --tend --tsnap --tline` (runner-dependent)
- microbenchmarks: `./deposit_bench nx ny ppc` (CLI, no deck)

## Full single-session sweep (2026-08-12 night, one binary, tend=3000)

`scripts/repro/perf_sweep.sh build build/perf_full --full` — full-length
(206,897-step) runs, matched diagnostics, every point a one-line deck edit:

| point | tile_sort | fused | wall (s) | p-steps/s |
|---|---|---|---|---|
| flat | 0 | — | 8560.4 | 3.78e9 |
| tiled, separate migrate (INVALID — knob ignored, see below) | 20 | 0* | 1899.0 | 1.70e10 |
| tiled, separate migrate (CORRECTED rerun) | 20 | 0 | 2947.4 | 1.10e10 |
| tiled, fused migrate | 20 | 1 | 1898.7 | 1.70e10 |
| cadence 10 | 10 | 1 | 1994.7 | 1.62e10 |
| cadence 40 | 40 | 1 | 1858.9 | 1.74e10 |

Findings: tiled/flat = **4.5×** (paper previously 3.4× from mixed-session
short runs). **CORRECTION 2026-08-12 (external review):** the sweep script
appended `tile_migrate_fused = 0` to the END of the deck, landing it in
`[species electrons]` where the section-scoped parser silently ignored it —
the "nofuse" row above actually ran WITH fusion, so the "fusion is null"
finding was an artifact. Fixed: script now inserts the knob inside `[field]`
and self-checks; the deck parser hard-errors on unknown keys in known
sections; the corrected full-length rerun gives **2947.4 s / 1.10e10** —
migration fusion is a real **+55%** effect (2.9x -> 4.5x in the chain), larger
than the short-smoke +20% because the penalty grows once the plasma is hot.
Energy histories fused vs nofuse agree to 6 significant figures through the
linear phase — physics identical, performance only.
Field energies bit-identical across paths at early times; ~4% spread at
t≈3000 (chaotic saturated phase, same amplitude). ArcWarden side of the
head-to-head remeasures at 1898.7 s (would be 2.8× vs OSIRIS's July
5392.8 s); paper keeps the conservative July matched pair (2.59×) until
OSIRIS is rerun on the current driver.

## Gaps + plan

1. **Middle ablation row** ("+tiled deposit" without fused migration) is not
   reachable from the current binary — the tiled kernel always fuses migration.
   - Option A (no code): two-point ablation (`tile_sort=0` vs `20`) + footnote
     the historical middle measurement.
   - Option B (recommended, ~20 lines): `[field] tile_migrate_fused = 0|1` —
     key in `deck.hpp` (next to `tile_sort`), field in `RunParams`, branch in
     `simulation_maxwell.hpp` step so write-back skips wrap/recompute and
     `parts_.migrate()` runs separately. Gate behind existing J-identity +
     continuity ctests. Enables the paper's "clean single-session ablation".
2. **Paper decks**: bake the canonical CLI overrides (e.g. An et al.
   `--amp=5 --ppc=262144`) into `*_paper.ini` variants so bare
   `./arcsim <deck> out` reproduces each figure; or add a 3-line
   `[pump] scale = X` key to keep Table I values verbatim while documenting
   the tuning.
3. **External / tooling** (not code): OSIRIS side of the head-to-head lives in
   `~/Donglai_Ma_reborn/osiris_case7_bench/` (reuse or rerun there); kernel
   budget table = `nsys profile` wrapper around the same decks; ncu roofline
   needs sudo perf counters.
4. **Sweep script**: `scripts/repro/` driver that regenerates every §5 number
   from decks alone (sort-cadence scan, ppc scan, grid scan).

## Definitive sweep in progress (2026-08-12, second review round)

The mixed-provenance issue (nofuse rerun on a rebuilt binary two commits after
the other four points; tracked CSV still holding the invalid nofuse row) is
being resolved by a full 5-point rerun with ONE binary at HEAD 54ab8bd
(md5 334bbb9f...), `scripts/repro/perf_sweep.sh build build/perf_v2 --full`.
The script now records git commit / binary md5 / driver in the CSV header and
emits `energy_consistency.csv` (relative total-field-energy difference vs the
fused reference at t = 100/300/600/900/1500/2999). On completion the tracked
CSV, logs, deck snapshots, and energy summary will be committed together and
the paper table finalized. Energy-agreement claims are now stated as
"physically consistent" with the defined metric and measured tolerances
(~1e-5 early linear, ~1e-3 by t≈600, few×1e-3 near saturation onset) — not
"bit-identical", not "six significant figures".
