# M2 boundary study: absorbing x-ends for the 2D Yee branch

Goal (plan M2): absorbing boundaries in the field-aligned direction (x here;
"s" in the L-shell geometry) with whistler-band reflection R < 1%, plus a
recorded map of failure modes. No PML presupposed — candidates compared on the
plan's benchmark: R(ω) for quasi-parallel whistlers measured **in plasma**.

## Scheme implemented

Umeda-style multiplicative masking layers, ported from the validated chirp1d
1D implementation (`include/pic/hybrid1d.hpp`) to the 2D Yee branch:

- `[boundary] x = damping|hybrid`, `nd` (cells per side), `numax`.
- Mask m(x) = exp(−ν_max d² Δt), d = normalized depth into the nd-cell layer,
  multiplied onto ALL wave fields each step (`yee::k_damp_x`; staggered site
  masks m(i), m(i+½)). The grid holds only wave fields — the external B0 in
  RunParams is never touched.
- `hybrid` additionally damps each layer particle's transverse momentum
  (u_y, u_z) by the same mask (2D analogue of chirp1d damping the cold-fluid
  v_cy/v_cz next to the fields).
- Particles reflect specularly at the domain ends (x = 0, nx) — the layers
  may contain plasma; the x field indexing stays periodic and the layers
  suppress the wrap-around leakage.

## Benchmark

`tools/boundary_reflection.cu`: cold magnetized plasma (ω_pe = 1, ω_ce = 0.5,
c = 10, ppc 100 quiet start), antenna column at the domain center radiates an
R-mode whistler packet (trapezoid-enveloped, 6 drive periods); probe column
between antenna and layer measures the **narrowband complex envelope at the
drive frequency** (demodulation + one-pole low-pass, both helicities);
R = late-window max / outbound-window max. Windows from the cold Maxwell
whistler group velocity; the late window opens just 2 drive periods after
the estimated round trip (an early version opened 6 periods late and
clipped the head of the reflected packet, understating R by up to ~3× —
all numbers below use the honest window).

Sanity gates: vacuum EM pulse absorption R = 0.14% with a periodic control
returning the pulse at 99.95% (`tests/test_boundary_vacuum.cu`, ctest
`boundary_vacuum`); probe spectrum peaks at the drive frequency; incident
amplitude 100× the shot-noise floor.

## Results (2026-07-17, RTX 5090)

R = A_ref/A_inc at the drive frequency; λ = whistler wavelength in cells
(dx = 1, i.e. dx·ω_pe/c = 0.1):

| ω/ω_ce | λ (dx) | mode | nd | ν_max | R |
|---|---|---|---|---|---|
| 0.40 | 75  | hybrid | 64  | 0.3 | 1.1% |
| 0.40 | 75  | hybrid | 64  | 1.0 | ~2% |
| 0.40 | 75  | hybrid | 256 | 0.1 | **0.84%** |
| 0.40 | 75  | hybrid | 384 | 0.1 | 0.80% |
| 0.25 | 105 | hybrid | 64  | 1.0 | ~17% |
| 0.25 | 105 | hybrid | 256 | 0.1 | **0.87%** |
| 0.10 | 186 | hybrid | 64  | 1.0 | ~40% |
| 0.10 | 186 | hybrid | 256 | 0.1 | **0.87%** |
| 0.10 | 186 | hybrid | 384 | 0.15 | 0.62% |

(Rows measured before the window fix, re-scaled qualitatively, are marked ~;
the bold production rows are honest-window measurements. Field-only masks
track hybrid to within ~2× at the ν_max optimum and degrade the same way
off it.)

**Production recommendation: `x = hybrid`, `nd = 256`, `numax = 0.1` →
R = 0.84% / 0.87% / 0.87% at 0.4 / 0.25 / 0.1 ω_ce — below 1% across the
whole chorus band with one setting** (nd = 384 buys 0.6–0.8%). The ctest
gate `boundary_whistler` runs this config at 0.4 ω_ce (R = 0.74%).

### Conclusions so far

1. **The controlling variables are nd/λ(ω) and the damping adiabaticity.**
   R < 1% needs the layer roughly a wavelength deep at the lowest frequency
   to absorb, with ν_max small enough that the damping profile is adiabatic
   (ν_max ≈ 0.3 ≈ 1.5 ω at 0.4 ω_ce beat ν_max = 1 and 3 — over-damping
   reflects off the impedance step before absorbing).
2. **Hybrid beats field-only by ~2.5× at the ν_max optimum** (0.36% vs 0.89%
   at 0.4 ω_ce, ν_max = 0.3), though both are usable; at over-damped settings
   they converge (the gradient reflection dominates). The wave's
   magnetic-energy fraction (~60% here) lets field-only masks work at all.
3. **Low-frequency band is the cost driver**: at 0.1 ω_ce, λ = 186 cells
   forces nd ≥ 256 (per side) with ν_max ≲ 0.3. For chorus runs (0.1–0.5
   ω_ce band) budget the layers off the LOWEST band frequency.

### Oblique incidence (ny = 1 tilted-B0 proxy)

Tilting B0 in the x–y plane while the antenna keeps k ∥ x̂ gives a wave with
wave-normal angle θ hitting the layers (still y-uniform, so ny = 1 remains
valid). Production config (nd = 256, ν_max = 0.1):

| θ | R (0.4 ω_ce) | R (0.25 ω_ce) |
|---|---|---|
| 15° | 2.2% | 1.8% |
| 30° | 8.0% | 5.9% |
| 45° | 32% | 13% |

R at θ = 30° is INSENSITIVE to layer config (nd 256→384, ν_max 0.1→0.3 all
read ~13% with the honest window) — a config-independent floor that smells
like measurement contamination (with oblique B0 the antenna also couples to
slow quasi-electrostatic branches whose late arrival pollutes the window)
rather than pure layer reflection. Recorded caveat: the ny = 1 proxy is
trustworthy for quasi-parallel incidence; a true 2D oblique-packet benchmark
is future work. For the M10 use case — field-aligned chorus reaching the
s-ends quasi-parallel — the θ ≤ 15° numbers (≈ 2%) are the relevant ones.

### Density gradient (密度渐变专项)

Cosine taper of the plasma density from n0 to n_edge over W cells in front
of each layer (implemented via per-particle weights; a cold static profile
with E(0) = 0 is force-free). At 0.4 ω_ce, nd = 64, ν_max = 0.3
(uniform-density baseline R = 1.1%):

| taper W | n_edge | R |
|---|---|---|
| 128 | 0.25 | 3.6% |
| 256 | 0.25 | 3.6% |
| 128 | 0.0625 | 9.2% |

**A density falloff into the layers makes absorption WORSE, and the
mechanism is not gradient reflection** (R is independent of taper width):
lower density raises the whistler group velocity (2.8 → 5.1 at n = 0.25),
so the wave crosses the layer in half the time and picks up half the
integrated damping. Design rule for the dipole/L-shell runs (M5+), where
field-aligned density falls toward the boundary ends naturally: do NOT add
an artificial taper, and size nd/ν_max for the LOCAL (lower) density at the
layer — i.e. for the faster group velocity and longer wavelength there.

### Failure modes recorded

- ν_max too large → reflection off the damping gradient (R grows with ν_max
  past the optimum). ν_max too small → insufficient integrated damping.
- nd ≪ λ → the layer is transparent regardless of ν_max (42% at nd/λ = 0.34).
- **L-sense antenna drive couples to nothing below ω_ce** (no L-mode): the
  drive is evanescent, only broadband ramp transients radiate, and the slow
  near-ω_ce content contaminates any late-window measurement. Antenna must
  rotate in the electron-gyration sense: (J_y + iJ_z) ∝ e^{+iω0 t}.
- Hard antenna turn-off radiates a broadband burst with vg → 0 components
  near ω_ce that linger near the source — use the trapezoid envelope.
- Field-only masks with plasma in the layer leave the coherent transverse
  current undamped; measurable but subdominant here (hybrid exists for runs
  where it matters, e.g. hotter plasma with larger kinetic energy fraction).

## Status

- [x] R(ω) quasi-parallel: production config < 1% across 0.1–0.5 ω_ce.
- [x] Oblique R(ω, θ) via the ny = 1 tilted-B0 proxy (θ ≤ 15° ≈ 2%; caveat
      recorded for θ ≥ 30°; true 2D packet benchmark = future refinement).
- [x] Density-gradient study: taper hurts (vg speed-up in the layer);
      design rule recorded for M5+ dipole density profiles.
- [x] ctest `boundary_whistler`: production config at 0.4 ω_ce, R = 0.74%
      < 1% gate. ctest `boundary_vacuum`: EM pulse R = 0.14%.
- [ ] (deferred) vacuum-gap/PML candidate — only if a future run needs
      θ ≥ 30° absorption the masks can't provide.

## R-TEST IN THE CHEN CONFIG (2026-07-27) — cavity hypothesis KILLED,
## hybrid self-burst artifact found, element-sink hypothesis sharpened

Method: 0.25 We0 antenna packet (amp 8e-4, linear) from the equator in the
REAL case-2 medium (98% cold fluid + 2% hot, dipole), hot made ISOTROPIC
(A=1: no growth contaminating R), ppc=8000 (noise energy 10x down; the
coherent same-seed subtraction trick FAILS on reflection timescales —
atomics decorrelate runs in ~500/We0). Decks decks/rtest_*.ini, data
build/rtest_*. Three boundaries: damping / hybrid / damping+refresh+precip.

RESULTS:
1. DAMPING ABSORBS WHISTLER PACKETS FINE. The incident V dies at the
   walls; only a weak diffuse return (~3e-4 local max vs 8e-4 incident).
   The all-PIC boundary_reflection.cu R~1 verdict does NOT transfer to
   the cold-fluid-dominated chen config (there the layer damps the cold
   current = the wave's main carrier). => the damping-giant's no-element
   cavity turbulence CANNOT be blamed on wave reflection.
2. HYBRID + ISOTROPIC f = PATHOLOGY (R unmeasurable): the no-antenna
   control SELF-BURSTS (interior <|B|^2> x10 at t~900-1200/We0) at
   0.25-0.75 We0 on the whistler branch (probe peak 0.49 We0 = 0.2x the
   LOCAL We at the layer). Cause from the ckpt post-mortem: the layer
   u_perp-damping carves the isotropic f catastrophically — 60% of ALL
   hot markers perp-cold by t=9000/wpe, global A 1.00->0.83, boundary
   A->0, near-layer f(u_par) bumpy — a free-energy structure that goes
   unstable. In CHEN runs the loss-cone-subtracted f0 keeps the cone
   ~empty, so the carving flux is small and hybrid is benign — the
   pathology is specific to cone-filled distributions. (The "return
   packet" in the hybrid R panel was this burst, not reflection.)
3. PRECIP ~= DAMPING for packets (weak return, no burst): precip acts
   only at the wall on the true cone — no interface carving, no emitter.

SYNTHESIS — why hybrid grows elements and damping does not, given both
absorb waves: the difference is the SCATTERED-PARTICLE SINK. During
growth, waves pitch-scatter resonant electrons into the (empty) cone;
hybrid permanently silences them on their first boundary visit (energy
sink => the box can go quiet between bursts => threshold-gated discrete
elements); damping returns them at full energy and they keep exchanging
with the field => perpetual marginal turbulence at dB/B ~ 1e-3, no quiet
reloading, no elements, free energy never tapped (ckpt: A within 1% of
load after 9000/We0). PRECIP IS EXACTLY THAT SINK done cleanly (wall
strip, true cone only, energy counted, bath-recycled ghosts).

SHARPENED 3-ARM PREDICTION (giant, damping base): closed = marginal
turbulence (verified); +refresh = sustained turbulence (verified at
ppc800); +refresh+precip = DISCRETE ELEMENT TRAINS RETURN and are
endless with stationary period. Falsifiable at 3x ~3.2 h.

CORRECTION (2026-07-27, user caught the propagation direction): quadrant
k-w decomposition CALIBRATED against the known-southbound incident packet
(the By+iBz helicity puts propagation power in the OPPOSITE (w,k)-sign
quadrant — always calibrate direction analyses against a known signal).
Corrected reading of the self-burst: waves are generated at MID-LATITUDE
in each hemisphere and propagate EQUATORWARD (south half: northbound x6;
north half: southbound x5), converging/crossing at the equator — the
inward Lambda. Mechanism corrected accordingly: this is BEAM-driven
whistler growth (normal cyclotron resonance w - k v_par = We(x)/gamma,
wave counter-propagates to the driving streamer; mu=0 streamers have
CONSTANT v_par, so each w tunes to a mid-latitude shell where We(x)
matches — hence off-equator generation and the 0.25-0.75 We0 upshifted
band at the equator), NOT the trapped-anisotropy equatorial-amplifier
logic (which requires orbit-varying v_par and does not apply to mu=0
streamers).

## BURST MECHANISM NAILED (2026-07-27 burst_study: ppc=32k, probes +-150,
## 4 ckpt snapshots; figs build/burst_waves.png, build/burst_mechanism.png)

The hybrid-layer self-burst (cone-filled loads) is a COUNTER-STREAMING
CARVED-BEAM WHISTLER INSTABILITY, normal cyclotron resonance, generated
mid-latitude, propagating equatorward:
- Timeline: carving first (perp-cold fraction 0.39 by t=600/We0), burst
  second (WB x30 at t~900-1400), then carving jumps 0.55->0.60 (the burst
  scatters more markers into the cone — the snowball's second turn).
- Streamer f(u_par) at t=1200: counter-streaming beams at u_par ~ +-0.15
  -0.2 near the equator (transiting ghosts); a stalled u_par~0 population
  at high |h| (markers carved AT their mirror points — dead weight).
- RESONANCE CLOSURE: dominant burst waves (w = 0.45-0.53 We0, k =
  0.79-0.96 wpe/c) have v_res = (w - We(x)/gamma)/k lying INSIDE the
  southbound-beam bulk (-0.1..-0.35) across |h| ~ 50-550 — equatorward
  waves ride normal cyclotron resonance against the counter-streaming
  carved beam, gain integrated along the whole path (hence mid-latitude
  apparent origin + equator crossing, the X in the h-t map). The lower
  0.26 We0 component is beam-resonant only near the equator — the
  frequency-latitude mapping behind the upshifted 0.25-0.75 band.
- Probes +-150 show clean discrete RISING elements (0.4->0.55 We0,
  dw/dt ~ +3.6e-4 We0^2 — chorus-like sweep magnitude, amplitude-
  controlled scaling is mechanism-agnostic); weaker later events mixed/
  falling. A boundary artifact can fake rising-tone elements: any novel
  hybrid-base signal must be screened against this channel.
- Direction decomposition (calibrated): equatorward dominance 13x (S)
  and 17x (N) during the burst.
