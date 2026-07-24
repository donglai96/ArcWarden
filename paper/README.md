# ArcWarden paper (side project)

Draft of the methods paper: **ArcWarden — a GPU-native PIC code for
cache-rich GPU architectures** (fused kernels + L2-resident fields beat the
classic chunk-pipeline design by 2.6× vs OSIRIS-CUDA on identical physics).

- `main.tex` — full draft (structure + written intro/methods/benchmark/
  methodology sections). Red `\tocheck{}` markers = numbers/figures still
  needed; grep for them.
- `refs.bib` — bibliography drafted from memory; **verify every entry
  against the publisher before submission.**
- Compile: no TeX on this box — use Overleaf, or `tectonic main.tex`.
  Figures reference `../docs/figs/` (copy in or symlink when compiling).

Evidence base (keep in sync):
- `../docs/PROFILE_BASELINE.md` — all measured rates and wall times.
- `../docs/PAPER_NOTES.md` — thesis, evidence list, TODOs.
- `../profiling/` — nsys/ncu artifacts (open in the GUIs: `nsys-ui`,
  `ncu-ui`); `*_util.txt` are nvidia-smi utilization samples.
- Run data: `../build/eaw_case7*/`, `../build/darwin_bench/`,
  `~/Donglai_Ma_reborn/osiris_case7_bench/` — never delete.

Venue candidates: Computer Physics Communications, J. Comput. Phys.
