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
    int         dist     = 0;           // 0 = bi-Max, 1 = loss-cone subtracted
    double      lc_rho   = 1.0;         // subtraction amplitude ρ ∈ [0,1]
    double      lc_kappa = 0.3;         // subtracted-component width κ ∈ (0,1)
    double      taud     = 0.0;         // delta-f drift-injection timescale
                                        // (physical time; RunParams::df_taud)
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
