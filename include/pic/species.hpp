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
};

using SpeciesList = std::vector<Species>;

} // namespace arc

#endif // ARC_PIC_SPECIES_HPP
