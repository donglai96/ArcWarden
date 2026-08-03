# Arm K — single-population product-kappa f0 (plan, 2026-08-01)

Goal: a **single, realistic** hot distribution whose linear spectrum covers
the upper band without strong global anisotropy — no engineered two-band
mixture (user directive: 更接近现实的简单分布, 不要人工双带). Replaces the
rejected two-component "Arm W" design.

## 1. The physics: A is allowed to depend on velocity

Kennel–Petschek growth at ω needs A(v_res) > ω/(Ωe−ω) **at the resonant
velocity** v_res = (Ωe−ω)/k∥. For any *separable* f = F∥(v∥)·Max⊥(T⊥):

    A(V) = T⊥ · (−∂lnF∥/∂V)/V − 1

- bi-Maxwellian: A = T⊥/T∥ − 1, constant → upper band ⇔ global A ≥ 1.5
  ⇔ Arm U's burst. Dead end (verified numerically, A = 0.516 flat).
- standard bi-kappa (non-separable): A ≈ constant too (verified:
  0.49–0.52 across the whole resonance range). Also dead end.
- **kappa-parallel × Maxwellian-perp (product form)**: the parallel
  log-slope is Maxwellian-steep in the core (effective T = θ∥², small) and
  power-law-flat in the tail, so

      A(V) = (T⊥/θ∥²) / (1 + V²/(2κθ∥²)) − 1

  **decreases with energy automatically** — which is precisely the robust
  observed trend in the chorus source (Li et al. 2011: anisotropy is
  concentrated at low energy). Upper-band electrons (v_res ≈ 0.05–0.07 c)
  see the anisotropic core; lower-band electrons (0.15–0.26 c) live in the
  suprathermal tail with mild A.

Why this counts as *more* realistic than what we have been running, not
less: every measured hot-electron spectrum has a suprathermal power-law
tail (κ ≈ 2.5–6 is the standard empirical fit); the bi-Maxwellian is the
distribution with that feature amputated. One population, three parameters
(θ∥, κ, uth⊥) — no engineered second band.

## 2. Design point (KP scan, absolute calibration against known regimes)

Anchors in identical units: x4 f0 → γ_max = 2.9e-3 (proven element
regime); Arm U → 2.3e-2 (proven burst). n_h is a free regime dial — it
scales γ without touching A(v) or ω_m.

| θ∥ | κ | uth⊥ | n_h | T⊥/T∥ (moments) | ω_m | γ_max | γ_U/γ_L | verdict |
|---|---|---|---|---|---|---|---|---|
| **0.10** | **2.5** | **0.162** | **0.009** | **1.05** | **0.58** | **3.7e-3** | **1.4** | **PRIMARY** |
| 0.10 | 2.5 | 0.175 | 0.007 | 1.22 | 0.66 | 8.6e-3 | 1.6 | fallback: upper band too weak |
| 0.11 | 2.5 | 0.175 | 0.009 | 1.01 | 0.57 | 3.6e-3 | 1.1 | fallback: most isotropic |
| 0.09 | 2.5 | 0.150 | 0.008 | 1.11 | 0.60 | 4.0e-3 | 1.9 | fallback: colder core |

The primary point is remarkable on paper:
- **moment anisotropy T⊥/T∥ = 1.05 — essentially isotropic pressure**
  (x4: 1.52, Arm U: 2.5), yet ω_m = 0.58 covers the upper band, because
  the core anisotropy T⊥/θ∥² = 2.6 does the work where it matters.
- lower-band drive γ_L = 2.6e-3 ≈ the proven x4 engine's 2.9e-3 —
  the element train should survive by construction.
- γ_max = 3.7e-3 = 1.3× x4, far from Arm U's 8×. Element regime expected.
- wall valve raises A over time (x4: effective ω_m 0.34 → 0.45 observed),
  so load-time ω_m = 0.58 should stretch upward in operation.
- physical scales: core T∥ ≈ 2.6 keV, T⊥ ≈ 6.7 keV, tail index κ = 2.5,
  n_h = 0.9 % — a plain injection-time hot electron population.

Resonance geometry unchanged and clean: U-band 0.05–0.07 c <
**m1 Landau plateau 0.087–0.100 c** < L-band 0.15–0.26 c; the kappa core
populates the Landau window well (F∥(0.09)/F∥(0) = 0.69).

## 3. Loader: kappa = Gamma-mixture of Maxwellians (exact, reuses everything)

A 1D kappa is a scale mixture of Gaussians (Student-t identity, ν = 2κ−1):
draw per-marker T∥ = σ² from InvGamma(ν/2, ν s²/2), s² = 2κθ∥²/(2κ−1),
then the marker samples an ordinary **bi-Maxwellian (E,μ) equilibrium with
that T∥** — the existing mirror-loader machinery applies per marker, so the
mixture is an *exact* Vlasov equilibrium along the field line (each λ-slice
is one; equilibria superpose).

Implementation (`dist = prodkappa`, new):
- `species.hpp`: `kappa_par` param; validity checks (κ > 1.5 for finite
  T∥ moment; note ⟨v∥²⟩ = 2κθ∥²/(2κ−3) needs κ > 1.5, our κ = 2.5 fine).
- density taper nfac(x) and conecut kept-fraction K(x) become λ-averages:
  32-point Gauss–Laguerre quadrature over the Gamma weight, computed
  host-side per cell (same place T1/nfac are computed now).
- kernel: per-marker draw λ (new RNG stream — **salted**, θ-collision
  lesson), form T∥(λ), then the existing bi-Max/conecut sampling path
  verbatim. Conecut rejection loop already tolerant (96 trials).
- mirror loader currently *rejects* kappa_v (particles.hpp:677) — that
  guard stays for the old bi-kappa; prodkappa is the new supported path.

## 4. Gates (each blocks the next)

The simulation needs ONLY the loader change — no new field/pusher physics.
Regime tuning is done empirically on the GPU (G4 is ground truth and costs
the same as one theory iteration); an exact dispersion solver is NOT a
gate — it is deferred to the paper stage (linear-spectrum anchor figure),
via the Gamma-mixture trick (dispersion = quadrature of bi-Max Z-functions)
if ever needed.

- **G2 loader gate.** Load, then per-v∥-bin moments: measured A(v∥)
  profile vs the analytic formula (THE new failure surface — a wrong
  mixture reproduces global moments while missing the core anisotropy);
  n(x) taper vs quadrature prediction; conecut hole; ρ flatness;
  ⟨u∥e^{−iθ}⟩ < noise.
- **G3 smoke.** x4_rsm_smoke geometry, rsm-off/on: noise sanity, perf
  ≈ 1.5e10 unchanged.
- **G4 regime check (go/no-go, THE tuning loop).** `pk_ctrl` to
  t ≈ 2500–5000 (~1–2 h): element-like sustained emission, power above
  0.5, no self-gap, no one-shot burst. Burst → halve n_h and rerun G4;
  upper band absent → uth⊥ = 0.175 fallback row. Each iteration is one
  deck edit + 1.5 h — cheaper and more honest than theory pre-tuning.
- **G5 the pair.** `giant_x4_pk_{ctrl,rsm}.ini`, t = 10⁴, `--ckptseq`,
  ~3.5 h each. See §5 for the strict deck contract.

## 5. Run design: close the 2×2 (f0 × k⊥)

The new pair is defined AGAINST the existing x4 pair
(`giant_x4_atmo40{,_rsm}`, both complete): **every section of the deck is
byte-identical except the initial condition** — [grid], [time], [field]
(tile_sort 25), [background] (lre 3326.26), [boundary] (atmo, batm
7.4057), [rsm] (k1 0.16), [antenna], probes, seed 20260720 all unchanged.
The only edited blocks:

| block | x4 pair | pk pair | note |
|---|---|---|---|
| [species hot] dist | conecut (bi-Max) | **prodkappa + same cone cut** | f = F_κ(v∥)·Max⊥(v⊥)·Θ(sin²α_eq − 1/cone_b); cone_b = 7.4057 = batm unchanged — the loss cone stays the wall's operator, applied per Gamma-slice in the loader |
| uth | 0.19783565 0.24390817 0.24390817 | 0.10 0.162 0.162 | + kappa_par = 2.5 |
| density / cold_nc | 0.0178 / 0.9822 | 0.009 / 0.991 | γ_L-matched (see below); ppc 28000 unchanged |

This closes a 2×2 causal matrix, two cells already run:

| | k⊥ off (ctrl) | k⊥ on (rsm) |
|---|---|---|
| bi-Max f0 | giant_x4_atmo40 ✅ | giant_x4_atmo40_rsm ✅ |
| prodkappa+LC f0 | pk_ctrl (new) | pk_rsm (new) |

Rows isolate the f0 effect (does a realistic source add the upper band?),
columns isolate the k⊥ effect (does E∥ carve the 0.5 valley?) — each
comparison differs by exactly one ingredient.

On n_h = 0.009 ≠ 0.0178: the density change IS part of the initial
condition, chosen to match the **lower-band drive** γ_L to the proven x4
engine (2.6e-3 vs 2.9e-3) — dynamical similarity of the element train is
the meaningful invariant, not raw density (KP: γ ∝ n_h, so equal n_h with
the colder kappa core would over-drive, 2.5× x4). If a density-matched arm
is ever wanted, it is one deck line, gated through G4 like everything else.

## 6. Pre-registered predictions

- **ctrl**: element train persists (γ_L matched to x4); spectrum extends
  past 0.5 toward ~0.6; **no 0.5 gap** (attribution control).
- **rsm**: 0.5 valley between bands, now in the sustained element regime
  (the demonstration Arm U's burst could not give); f(v∥) plateau at
  [0.087, 0.100] c ± symmetric; per-band ω_stop statistics.
- Registered risks: (a) γ_U/γ_L = 1.4 is top-heavy — upper band may lead
  and reorganize the element train; that is a *finding* about
  weak-anisotropy sources, not a failure; (b) core anisotropy relaxes
  fast (small reservoir) → upper band intermittent; watch per-v∥-bin
  A(t) fuel gauge; (c) KP-heuristic γ is off by an O(1) factor → absorbed
  by the G4 n_h iteration loop.

## 7. Rejected alternatives (for the record)

- Two-component bi-Max mixture (Arm W): works on paper (γ_max = x4's,
  ω_m 0.64) but is an engineered two-band source — rejected by design
  intent 2026-08-01.
- Standard bi-kappa: A(v) ≈ constant — cannot decouple upper-band
  coverage from global anisotropy.
- Loss-cone subtracted bi-Max (dist=1): same parallel F∥ in both terms →
  A(v) constant as well.
- Tabulated observed f0 (THEMIS/RBSP fit): maximal realism, but the (E,μ)
  equilibrium mapping of an arbitrary table + conecut is a loader project
  of its own; revisit only if prodkappa fails its gates.

Cost: loader + G2/G3 ~2–3 h; G4 ~1.5 h GPU per iteration; G5 ~7 h GPU.
