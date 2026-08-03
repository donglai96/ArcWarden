// ArcWarden — host-side description of one particle species (OSIRIS/UPIC style).
//
// A species is a population loaded into the shared particle arrays with its own
// density, particle count, thermal spread (uth) and fluid drift (ufl). Classic
// kinetic setups are just a list of these:
//   two-stream    = two species, ufl = +v0 and -v0
//   bump-on-tail  = a warm bulk species + a small fast beam species
// so no special-case code path is needed in the loader (cf. the old two_stream /
// bump_on_tail flags). v1 keeps q/m global (all electrons); per-species q/m is a
// later step (ions) and needs per-particle q/m in the pusher.

#ifndef ARC_PIC_SPECIES_HPP
#define ARC_PIC_SPECIES_HPP

#include <string>
#include <vector>

namespace arc {

struct Species {
    std::string name    = "electrons";
    double      density = 1.0;          // number density n0 carried by this species
    int         ppc     = 0;            // macro-particles per cell for this species
    double      uth[3]  = {0.0, 0.0, 0.0};  // thermal velocity per dimension
    double      ufl[3]  = {0.0, 0.0, 0.0};  // fluid (drift) velocity per dimension
    bool        deltaf  = false;        // M3: delta-f representation ([species] rep = deltaf)

    // M5a/Chen-PoP-2026: loss-cone SUBTRACTED bi-Maxwellian (dist = losscone),
    //   f_perp ∝ exp(−u⊥²/2 U⊥²) − lc_rho · exp(−u⊥²/(2 lc_kappa U⊥²)),
    // U⊥ = uth[1] (Chen et al. Eq. 1; their Ut⊥). Only the mirror loader
    // (initialize_mirror) honors it; delta-f ∂lnf0 for it is a later step.
    int         dist     = 0;           // 0 = bi-Max, 1 = loss-cone subtracted,
                                        // 2 = cone-cut ([species] dist = conecut),
                                        // 3 = product-kappa-par + cone-cut
                                        //     ([species] dist = prodkappa)
    double      lc_rho   = 1.0;         // subtraction amplitude ρ ∈ [0,1]
    double      lc_kappa = 0.3;         // subtracted-component width κ ∈ (0,1)
    // dist = 2 (x4-atmo arm, 07-30): HARD pitch-cone cut on the bi-Maxwellian,
    //   f = biMax(uth) · Θ(sin²α_eq − 1/cone_b),  cone_b = B_wall/B_eq.
    // Trapped population only: markers whose mirror point lies beyond the wall
    // are never loaded (same criterion as the atmo boundary with batm = cone_b
    // — load and boundary are the SAME operator, zero initial transient).
    // Local form at mirror ratio b: reject sin²α_local < b/cone_b.
    double      cone_b   = 0.0;         // wall mirror ratio (>1 required)
    // dist = 3 (x4 Arm K, 08-01): PRODUCT form — kappa PARALLEL x Maxwellian
    // PERP, with the same hard cone cut as dist = 2:
    //   f = [1 + u∥²/(2κθ∥²)]^(−κ) · exp(−u⊥²/(2u⊥th²)) · Θ(sin²α_eq − 1/cone_b)
    // θ∥ = uth[0] (CORE width, not the moment: <u∥²> = 2κθ∥²/(2κ−3)),
    // u⊥th = uth[1]. KP anisotropy A(v∥) = (u⊥th²/θ∥²)/(1+v∥²/(2κθ∥²)) − 1
    // DECREASES with energy: upper-band resonant (low-v∥) electrons see the
    // anisotropic core, lower-band (tail) electrons a mild A — the B<->C
    // balance (docs/X4_PRODKAPPA_PLAN.md). NOT the bi-kappa kappa_v (whose
    // A(v) is constant). Loaded as an InvGamma (Student-t, ν = 2κ−1) scale
    // mixture of bi-Max (E,μ) equilibria — superposition of equilibria is an
    // equilibrium; see initialize_mirror for the tilted-mixture sampling.
    double      kappa_par = 0.0;        // dist=3 parallel kappa index (> 1.5)
    double      taud     = 0.0;         // delta-f drift-injection timescale
                                        // (physical time; RunParams::df_taud)
    double      kappa_v  = 0.0;         // G1.1: bi-kappa velocity index (0 =
                                        // Maxwellian). f ∝ [1 + (u∥²/θ∥² +
                                        // u⊥²/θ⊥²)/κ]^(−κ−1), θ = uth.
                                        // Sampled as Gaussian·√(κ/W),
                                        // W ~ χ²_ν shared across components,
                                        // ν = 2κ−1 (positive integer →
                                        // κ ∈ {1, 1.5, 2, 2.5, ...});
                                        // <u∥²> = κθ∥²/(2κ−3) needs κ > 1.5
                                        // for finite temperature. Noisy load
                                        // only (uniform loader).
    double      wdnoise  = 0.0;         // delta-f initial weight noise rms:
                                        // wd(0) = ±uniform, rms = wdnoise.
                                        // A PERSISTENT noise source (weights
                                        // are not absorbed at boundaries) —
                                        // the δf analog of full-f shot noise,
                                        // level ≈ wdnoise × full-f floor.
};

using SpeciesList = std::vector<Species>;

} // namespace arc

#endif // ARC_PIC_SPECIES_HPP
