# How k_perp enters ArcWarden: the RSM m = ±1 spectral channel — implementation design

Companion to `docs/RSM_MODEL_DEFINITION.md` (the model constitution: scope,
physics assumptions, success criteria). That document says *what* the reduced
model is; this one says *how* it is built into the code — the field
representation, the particle phase contract, the charge-conserving modal
deposit, and the numerical safeguards that make a ~1 % quantity (E∥)
trustworthy. Code: `include/pic/rsm_oblique.hpp`; deck section `[rsm]`;
gates V1–V5 in `tests/test_rsm_*`.

Design maxim: **y is demoted from a spatial dimension to a per-particle phase
label; k_perp is grafted on as one exact complex spectral line beside the
legacy engine; the two channels meet only inside the particles.** With the
flag off, none of the new code executes (separate kernels — non-invasive by
construction).

---

## 1. The ansatz: Fourier truncation in y, not a grid

The box keeps ny = 1 — there is no y grid at all. Every field is written as

    F(x, y, t) = F0(x, t)  +  [ F1(x, t) · e^{i k1 y} + c.c. ]

- **m = 0**: the legacy real Yee fields — the parallel chirping chorus
  engine. Its physics is untouched (same kernels' op sequence, same
  Esirkepov scatter).
- **m = ±1**: a conjugate pair (their sum keeps the total field real),
  represented by **complex 1-D arrays** E1x/E1y/E1z, B1x/B1y/B1z, J1x/J1y/J1z
  over x. k_perp = k1 is the single, global, fixed transverse wavenumber.
- Every ∂/∂y in Maxwell's equations is replaced by **exactly i·k1** — a
  spectral derivative, not a finite difference. Zero discretization error and
  zero phase error in y.

This is the literal meaning of "few-k_perp reduced model": instead of solving
the full (x, y) plane, retain exactly one oblique dispersion branch. The cost
is O(nx) complex arrays; the particle count does not change.

Lattice note: the m = 1 fields are staggered only in x (standard Dx); they
do **not** inherit the Yee y half-shift — y is spectral, and inheriting the
half-shift would multiply the mode by a constant spurious phase factor
(constitution §6.1).

## 2. The particle phase contract: y ≡ θ/2π, no new state

Particles couple to the m = 1 mode only through the phase θ = k1·y. Impose
the deck-enforced **phase contract**

    Ly = 2π / k1        (deck.hpp validates |k1·Ly/2π − 1| ≤ 1e-9)

and three things become true simultaneously:

1. The existing particle coordinate `p.y ∈ [0, 1)` (cell units, dyp = Ly)
   **is** θ/2π — no new particle state is added, no memory cost.
2. The legacy pusher already advances `y += vy·dt/dyp`, which under the
   contract is identically θ̇/2π = k1·vy/2π — **phase evolution is correct
   for free**, in both the RSM pusher and the flat path.
3. The periodic y-wrap is exactly phase-periodic.

**The R1 phase trap** (constitution §6.1, caught in design review): every
ny = 1 loader pins y = 0.5, i.e. all particles at identical phase — which is
equivalent to injecting a *coherent fake oblique seed*. Remedy: after any
loader (including the new conecut loader, since this acts post-load),
`rsm_theta_init` (chirp2d.cu L86-89) redraws θ uniform on [0, 2π) with its
own RNG stream (11; loaders use 0–5), independent of position and gyrophase.
Gate V1 (`tests/test_rsm_phaseload.cu`) demonstrates the trap and verifies
the remedy.

## 3. Gather: one push with the total field

The particle feels the reconstructed total field at its own phase
(`rsm_oblique.hpp` ~L260-299):

    Ex += 2·( Re E1x · cosθ − Im E1x · sinθ )      // = 2·Re[E1x e^{iθ}]
    ... (all six components; factor 2 = the m = +1 and m = −1 conjugates)

followed by **one** standard Boris kick–rotate–kick with the summed field
(identical op sequence to `yee_advance_particle`, including the dipole
B0(x) + mirror mc-term branch). This is the **one-push rule**: there is no
"push by m0, then push by m1" splitting. Landau trapping is an interference
effect between the wave the particle rides and its own phase-space motion;
split pushes would linearize exactly the physics this model exists to
capture.

## 4. Deposit: exact modal Esirkepov (charge conservation in mode space)

The particle projects its current onto the mode, J1 ∝ q·v·e^{−iθ}. A naive
scatter of that expression does **not** conserve charge. The modal continuity
equation is

    ∂t ρ1 + ∂x J1x + i·k1·J1y = 0

and the deposit (`rsm_oblique.hpp` L333-364) is built so this closes
per-particle, exactly:

- **J1x** goes on x-links via the Esirkepov shape-difference W = S1 − S0,
  carrying the *averaged* phase factor ē = ½(e^{−iθ0} + e^{−iθ1}) (before-
  and after-push phases).
- **J1y** does **not** use vy. It absorbs the *phase flow itself*:
  ∝ i·(e^{−iθ1} − e^{−iθ0}) / (k1·dt) on nodes. The particle's phase advance
  θ0 → θ1 *is* the transverse current — this is what makes ∂t ρ1 + i k1 J1y
  telescope against the phase change exactly.
- **J1z** on nodes with vz and the averaged phase factor.

Gate V3 numbers: per-particle kick consistency 1e-6, modal continuity
4.9e-5, div·B1 3e-6, energy ledger closure 3.6 % — with **no** post-hoc
current projection or smoothing.

The m = 0 deposit in the RSM pusher is the same Esirkepov scatter as the
flat path (GlobalJSink), so with the m1 amplitude at zero the two paths are
statistically indistinguishable (V3 regression re-PASS both paths; exact bit
identity is impossible under float atomics — the envelope method of
`scripts/regress_case2.py` is the standard).

## 5. Making E∥ believable: the three safeguards

E∥ (= E1x here, since x is along B0) is the target observable of the whole
program, but it is only ~1–3 % of the wave field. Three components exist
solely to keep it from being numerical garbage:

1. **m = 1 cold-fluid complex twin** (`k_rsm_cold_fluid/current`): the
   linearized response of the 98.2 % cold population to E1 (vc1 arrays).
   Without it, the m = 1 branch has the wrong dispersion entirely.
   Gate V2: ω vs Stix ≤ 0.8 %, **E∥/E⊥ ratio vs Stix ≤ 1.1 %**, secular
   drift +0.0000.
2. **Cold-inclusive m = 1 Gauss correction**: uncorrected electrostatic
   residue accumulates as fake E∥ → fake plateau → fake gap (named failure
   mode in the constitution risk register). The correction must include the
   cold-fluid charge, else it *creates* the artifact it is meant to remove.
3. **Zero y-phase error by construction** (§1 lattice note + spectral ik1).

## 6. Boundaries and damping for the m = 1 line

- **x-end absorbing layers**: `k_rsm_damp_x` is the complex twin of
  `k_damp_x` — the *same* masks (mn at integer-x sites: e1y/e1z/b1x + cold
  vc1; mh at half-integer sites: e1x/b1y/b1z), Re and Im damped identically.
  Host side (`simulation_maxwell.hpp` L158-166) invokes it for any
  bnd_x ≠ 0, so all boundary modes inherit field damping automatically.
- **Particle wall**: the RSM pusher folds the specular reflection before the
  deposits (Esirkepov sees the reflected path), and implements the bnd_x = 2
  hybrid u⊥ carve. **Known gap (2026-07-31, fix planned in
  `docs/X4_RSM_EPAR_PLAN.md` §3.1): the bnd_x = 3 atmo precipitation
  criterion is missing** — without it the wall degenerates to a pure
  specular mirror (no loss-cone valve). The fix mirrors `yee2d.hpp` L299-311
  (refl flag + batm mirror-point test → u⊥ → 0 return), preserving op order
  so the m0 deposit stays matched to the flat path.
- **y "boundary"**: none exists — the phase wrap is exact periodicity of the
  mode, not a boundary condition.

## 7. What k_perp means physically in this design

The box supports exactly two wave families:

- **k⊥ = 0** (m0): the parallel whistler branch — the chirping element
  engine. Strictly E∥ = 0; Landau resonance impossible by construction.
- **k⊥ = ±k1** (m = ±1): one oblique branch. Because k1 is fixed while k∥
  moves along the dispersion curve, the **wave-normal angle is a function of
  frequency**: θ(ω) = atan(k1/k∥). At k1 = 0.16 (fpe/fce = 5):
  15.5° at ω = 0.25 Ωe → 9.1° at 0.5 Ωe. E∥ arises from the oblique
  whistler polarization, and lands in a single named array (E1x) that probes
  write out directly — no decomposition step, no leakage from E⊥.

The two channels have **no direct field coupling** (linear Maxwell modes are
orthogonal; the cold fluid is linearized). They couple *only through the
particles*: the same full-f population is pushed by the total field and
deposits into both modes. Consequently, any back-reaction of the m1-carved
Landau plateau on the m0 chirp is a genuine particle-mediated result, not an
artifact of the construction — this is what makes the x4×RSM experiment a
real question rather than a tautology.

## 8. Alternatives that were rejected

| Alternative | Why rejected |
|---|---|
| Full 2D grid (ny ≫ 1) | ~64×+ cells forces ppc to O(100): shot noise buries the ~7 %-wide Landau plateau window (07-28 phase-hole lesson); uncontrolled k⊥ spectrum ruins attribution. Constitution §4. |
| Tilted B0 (Li 2019 repro style) | x is the dipole mirror axis; B0 direction varies with x — a global tilt is undefined in mirror geometry. |
| Forcing an explicit E∥ term on m0 | Not self-consistent; breaks charge conservation; assumes the answer. |
| Guiding-center / gyro-averaged particles | Landau + cyclotron resonances needed simultaneously on the same particles (constitution §7). |

## 9. Verification ladder status (all PASS)

| Gate | Content | Key numbers |
|---|---|---|
| V1 | phase-load trap + remedy | fake-seed demonstrated; θ uniform verified |
| V2 | m1 Maxwell + cold twin dispersion | ω vs Stix ≤ 0.8 %; E∥ ratio ≤ 1.1 %; drift +0.0000 |
| V3 | one-push + exact modal Esirkepov | kick 1e-6; continuity 4.9e-5; divB1 3e-6; ledger 3.6 %; flat-path regression re-PASS |
| V4 | engine integration | PASS (V6 forensics separately blocked a false gap: comb valley ≠ gap) |
| V5 | GPU infra: ckpt resume, hypothesis deck | rsm_chirp_case2 runs end-to-end |

Judging rule carried forward from V6: modal spectra P0 and P1 are read
**separately**, at the equator probe, peak-local (never box-averaged).

## 10. Code map

| Piece | Location |
|---|---|
| Complex field/current lines, RsmState/RsmViews | `include/pic/rsm_oblique.hpp` (top) |
| RSM pusher physics (gather + one-push + boundary) | `rsm_advance_particle`, `rsm_oblique.hpp` (shared by flat + tiled kernels) |
| m1 modal deposit (sink-templated) | `rsm_modal_scatter<J1Sink>`; GlobalJ1Sink (flat/strays), SharedJ1Sink (tiled) |
| Tiled RSM kernel (2026-07-31: 8× perf fix, 1.9e9 → 1.51e10 p-steps/s at 543 M) | `k_rsm_push_esirkepov_tiled<16,16,2>` — yee tile machinery + 1-D complex m1 shared tile (~0.6 KB) |
| m1 Faraday/Ampère (ik1 exact), cold twin, Gauss | `rsm_oblique.hpp` + host calls in `simulation_maxwell.hpp` L146-156 |
| m1 x-end damping | `k_rsm_damp_x`, `rsm_oblique.hpp` L367-388 |
| Deck: `[rsm] enable/k1/seed`, phase-contract validation | `include/pic/deck.hpp` L237-302 |
| Config flags | `include/pic/config.hpp` L239-248 |
| θ randomization after load | `rsm_theta_init`, called in `tools/chirp2d.cu` L86-89 |
| m1 probe output (B1y, B1z, E1x = E∥, E1y complex lines) | `tools/chirp2d.cu` L255+ |
| Gates | `tests/test_rsm_phaseload.cu` et al. |
| Decks | `decks/rsm_band_li.ini`, `rsm_band_li_big.ini`, `rsm_chirp_case2.ini`; planned `giant_x4_atmo40_rsm.ini` |
