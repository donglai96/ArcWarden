# Input decks

Every physics setup in ArcWarden is a plain-text `.ini` deck — no recompile per
setup. Each deck header states the runner, the reference (if it reproduces a
published result), and the run command. Decks are grouped by their role in the
methods paper (`paper/`).

## Running a deck

The unified runner is `arcsim` (2D branches; field model chosen by the deck):

```bash
./build/arcsim decks/verification/two_stream.ini out_dir
```

Specialized runners (1D electron-hybrid chirping, movies, benchmarks) are named
in each deck's header, e.g.:

```bash
./build/chirp1d  decks/chirping/chirping_1d_tao_trig.ini out   # Tao 2017 rising tone
./build/chirp2d  decks/chen2026/chen2026_case2.ini out --fullf # Chen 2026 Case II
./build/eaw2d_yee decks/benchmarks/eaw_case7.ini out           # OSIRIS head-to-head case
```

## Switching the field model

The 2D field model is a **runtime** deck setting, so one physics setup can be
run through any branch by editing one line:

```ini
[field]
model = es       # electrostatic  (spectral Poisson)
model = darwin   # spectral Darwin (magnetoinductive, no light-wave CFL)
model = yee      # full-Maxwell FDTD + Esirkepov current deposit
```

Common knobs (all runtime): `[grid] nx/ny/Lx/Ly`, `[time] dt/nsteps/dump_every`,
one `[species <name>]` block per population (`density`, `ppc`, `uth`, `ufl`),
`[plasma] noisy/seed` for noisy vs quiet start, `[field] B0` for the background
field. Compile-time template parameters (precision, deposit policy, shape order)
are fixed per binary; see `include/pic/config.hpp`.

## Directory map

| Directory | Contents | Paper section |
|---|---|---|
| `verification/` | two-stream, bump-on-tail, whistler anisotropy growth gate, linear-growth gamma box | §Verification |
| `benchmarks/` | Ma et al. 2024 case 7 (OSIRIS head-to-head), profiling deck | §GPU performance |
| `an2019/` | An et al. 2019 PRL Sims 1–3 (whistler-pump nonlinear trapping), Table I verbatim | §Nonlinear benchmark |
| `chirping/` | Tao et al. 2017 rising-tone chorus (1D electron-hybrid) + Chen 2022 hooked chorus | §Chirping demonstration |
| `chen2026/` | Chen et al. PoP 2026 Cases I–III (incoherent / discrete-element / hiss-like), full-f and δf variants | §Chirping demonstration |

### Canonical decks per figure

- **Two-stream / bump-on-tail phase space** — `verification/two_stream.ini`, `verification/bump_on_tail.ini`
- **Whistler anisotropy growth vs linear theory** — `verification/chirp1d_growth.ini` (checked against `scripts/whistler_kinetic_dispersion.py`)
- **OSIRIS comparison figure** — `benchmarks/eaw_case7.ini` (flat) and `benchmarks/eaw_case7_tiled.ini` (production tiled path)
- **An et al. reproduction** — `an2019/an2019_sim{1,2,3}.ini` (`v_r/v_th` = 3.2 / 2.1 / 1.0)
- **Rising-tone chirping element** — `chirping/chirping_1d_tao_trig.ini` (spectrogram) and `chirping/chirping_1d_tao_phase4.ini` (dense phase-space dumps, hole tracking)
- **Chen 2026 three-regime figure** — `chen2026/chen2026_case{1,2,3}.ini`

Suffix conventions in `chirping/` and `chen2026/`: `_df` = δf representation,
`_giant` = GPU-limit particle count, `_trig` = triggered (seeded) run, other
suffixes are documented parameter-study variants kept for reproducibility of
the study history (see `docs/TAO2017_REPRODUCTION.md`,
`docs/CHEN2026_REPRODUCTION.md`).
