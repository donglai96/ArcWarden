# M4 hot-representation comparison, round 1 (v4-gate-1d record)

Setting: Tao GRL17 rising-tone setup through the 2D Yee code path
(`decks/chirping_2dpath.ini`, runner `tools/chirp2d.cu`): parabolic B0(x)
(a = 3.448e-6 (wpe/c)², edge mirror ratio 3.32), linearized cold fluid
(nc = 0.994), hot bi-Max nh = 0.006 (uth 0.2/0.53c), Umeda hybrid layers
(nd = 300, numax = 0.1), antenna at the equator (amp 1e-4 → δB/B0 ≈ 1.2e-3,
w0 = 0.25 wce, off at t = 2500). Measurement: STFT of By+iBz at the ±100
c/wpe probes (the equator probe sits on the antenna node — noisy in BOTH
codes), ridge = max-power frequency per column, dw/dt from an LSQ line.
Fit windows: nonrel [1800,4800], rel [2500,6000] (the rel element starts
later — lower growth).

## Chirping-rate closure matrix (dω/dt in 1e-4 wce²)

| | chirp1d (1D reference) | 2D code path |
|---|---|---|
| Newtonian    | 5.2 (this study, `nonrel=true` deck key) | **5.3** (r = 0.98) |
| relativistic | 4.2 (tag v1d-chirping-tao2017)           | **4.0** (r = 0.98) |

Tao GRL17 printed value: 5.3 (their run is relativistic; DAWN and chirp1d
agree with each other, both ~20% below the paper — inherited offset, not a
2D-path artifact). The ~25% relativistic reduction reproduces in BOTH codes;
elements: nonrel 0.26→0.60 wce, rel 0.24→0.53 wce, both starting at the
0.25-wce trigger. Figures: `docs/figs/chirp2d_first_light.png` (nonrel),
`docs/figs/chirp2d_rel_closure.png` (nonrel vs rel + wd saturation).

## Marker convergence (rel, window [2500,6000], probe −100)

| ppc (equator) | markers | dω/dt (1e-4 wce²) | r |
|---|---|---|---|
| 1024 | 3.3M  | 4.02 | 0.97 |
| 2048 | 6.6M  | 3.47 | 0.98 |
| 4096 | 13.2M | 4.09 | 0.98 |

Spread ±9% with no ppc trend — the ridge-fit window choice contributes
comparable scatter (±8% across reasonable windows at fixed ppc). The
chirping rate is converged at ppc ≥ 1024 for this observable.

## Antenna-off self-consistent triggering (full-f, `--fullf --amp=0`)

W_B grows spontaneously from the shot-noise floor 1.5e-7 to 9.8e-4 at
t = 8800 (3.8 decades); the first spontaneous element rises 0.21→0.44 wce at
+2.2e-4 wce² (r = 0.85) — noisier and slower than the triggered delta-f
element, matching the chirp1d/PPCF17 full-f phenomenology. Delta-f with the
antenna off stays empty (no spontaneous emission — the marker noise is
wd-weighted; chirp1d found the same). Figure:
`docs/figs/chirp2d_selftrigger.png`.

## σ_wd evolution and representation health

- Nonrel triggered run: rms wd saturates at 1.9 (max ~ its distribution
  tail) — |δf| > f at many markers, the delta-f advantage is spent; the
  post-element state is broadband turbulence. Symptom of the OVER-DRIVEN
  Newtonian regime, not of the scheme.
- Rel triggered run: rms wd saturates at **0.82** and stays flat to
  t = 5e4 with no secular growth — healthy through element + turbulence.
- Full-f at the same ppc: pure shot noise floor ~1e-7 in W_B, i.e. the
  delta-f noise advantage at these amplitudes is ~3 decades in seed energy
  (consistent with the M3 growth-gate finding that the delta-f floor is set
  by the SEED, not ppc).

Round-1 recommendation (to be revisited in M8 with core/warm/hot splits):
delta-f for the hot population wherever δf/f stays ≲ 1 (triggered/controlled
studies); full-f when spontaneous re-seeding matters (repetitive elements,
long turbulent phases) or when trapping drives wd to O(1) anyway.

## Nonrel-vs-rel physics note

Relativity at uth_perp = 0.53c: lowers linear growth (heavier resonant
electrons), delays the element (t0 ≈ 2500 vs 1800), lowers dw/dt ~25%, halves
the wd saturation, and the post-element turbulence is weaker (W_B tail 8e-6
vs 4e-5). `[plasma] rel = true` is the production default for chorus runs.
