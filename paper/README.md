# ArcWarden methods paper

**ArcWarden: a GPU-native particle-in-cell framework for high-throughput
kinetic plasma simulations.** Target venue: Computer Physics Communications
(alt: J. Comput. Phys.).

Story arc (one narrative, three headline results):

1. **Performance** — 2.6× OSIRIS-CUDA on identical physics, same GPU, with an
   ablation chain accounting for the gain (§5).
2. **Extensibility** — ES / Darwin / full-EM solvers, inhomogeneous B0,
   loss-cone & δf species, hybrid kinetic–fluid, absorbing boundaries — one
   tested particle engine, everything runtime deck-selected (§6, the module
   table).
3. **Physics capability** — three-level validation ladder: textbook
   verification (§4) → An et al. 2019 nonlinear reproduction (§7.1) →
   self-consistent whistler chirping in an inhomogeneous plasma (§7.2).

Deliberate choices: chirping is the closing demonstration, not the paper's
subject — no chorus-mechanism claims (no Omura-vs-Tao, no band gap); the title
does not contain "chirping".

## Files

- `main.tex` — the draft. Red `\tocheck{}` markers = numbers/figures still
  needed; grep for them.
- `refs.bib` — bibliography drafted from memory; **verify every entry against
  the publisher before submission.**
- `figs/` — figure copies (masters live in `../docs/figs/` and run outputs).
- Compile: `tectonic main.tex` or any pdflatex (Overleaf works). Switch to
  `elsarticle.cls` for CPC submission.

## Evidence base (keep in sync)

- `../docs/PROFILE_BASELINE.md` — all measured rates and wall times.
- `../docs/PAPER_NOTES.md` — thesis, evidence list, outstanding measurements.
- `../docs/TAO2017_REPRODUCTION.md`, `../docs/CHEN2026_REPRODUCTION.md`,
  `../docs/EAW_CASE7_REPRODUCTION.md` — per-result reproduction reports.
- `../decks/README.md` — canonical deck per figure.
- Run data: `build/eaw_case7*/`, `~/Donglai_Ma_reborn/osiris_case7_bench/` —
  never delete.

## Outstanding before submission (from `\tocheck{}`)

- Architecture figure (deck → configuration → core → kernels + module column).
- Four-panel verification figure (dispersion / growth / energy / charge) from
  existing ctest outputs.
- Scaling curves: rate vs ppc and vs grid size, both codes.
- Clean single-session ablation rerun.
- ncu roofline numbers (needs perf-counter permission).
- Bibliography verification; ma2024 final title.
