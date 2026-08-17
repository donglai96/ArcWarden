// pic2d — lean, δf-native, multi-species marker store (PLAN_2D_REBORN §2.2, §4).
//
// Design decisions vs the legacy Particles (pic/particles.hpp):
//   * one MarkerStore PER SPECIES (species-segmented, not sliced): every
//     kinetic species carries its own f₀, ∂lnf₀, weight ledger, deposit tag
//     and diagnostics — multi-δf is first-class (the mirror2d single-species
//     guard is gone by construction, not by relaxing a check);
//   * 32 B/marker resident: x, z, ux, uy, uz, w, wd (7×f32) + cell (u32).
//     wd exists for every species; full-f runs simply keep wd ≡ 1 (branchless
//     deposit uses w·wd in both representations). Sort scratch is a CHUNKED
//     double buffer (≤ 1/8 of markers at a time) → 36 B/marker effective,
//     vs 60 B/marker in the legacy layout (PLAN_2D_REBORN §4 table);
//   * positions are PHYSICAL (x,z), not cell units: the shell loader, the
//     δf weight equation and the analytic background all live in physical
//     coordinates; cell index is derived, never authoritative;
//   * markers exist only where f₀ > 0 (shell-compact loading): memory
//     scales with shell area, not box area — the property that makes the
//     curved geometry affordable (see the |∇L| = sec²λ shell-area formula
//     in background2d.hpp).
//
// P0 scope: layout, allocation, byte accounting (consumed by the deck
// pre-flight), views. Loader ((E,μ) shell mapping + analytic ∂lnf₀ for
// bimax/losscone/prodkappa), push/deposit and migration land in P2 —
// see PLAN_2D_REBORN §6.

#ifndef ARC_PIC2D_PARTICLES2D_HPP
#define ARC_PIC2D_PARTICLES2D_HPP

#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace arc2d {

// Distribution tags (analytic ∂lnf₀ REQUIRED for every member before a δf
// run may use it — the prodkappa/conecut gap in the legacy code is a P2
// deliverable, not an accepted restriction).
enum class Dist : int { bimax = 0, losscone = 1, prodkappa = 2, conecut = 3 };

// Host-side description of one kinetic species (deck [species <name>]).
struct SpeciesCfg {
    std::string name;
    bool   deltaf   = true;    // δf (production) vs full-f (cross-arm)
    int    dist     = int(Dist::bimax);
    double n0       = 0.01;    // peak density / n_c at the shell center, equator
    double uthpar   = 0.2;     // thermal momenta (units of c·[code c])
    double uthperp  = 0.3;
    double kappa    = 0.0;     // prodkappa
    double lc_rho   = 0.0;     // losscone depth parameter
    double taud     = 0.0;     // δf weight relaxation; DEFAULT 0 (H1/H2 arm knob)
    // shell-compact support in L (raised-cosine edges of width edge_dL):
    double shell_L0 = 1330.5;
    double shell_dL = 40.0;
    double edge_dL  = 8.0;
    int    ppc      = 300;     // markers per SHELL cell
    uint64_t nmax   = 0;       // filled by deck finalize (shell cells × ppc)
};

// POD device view — passed to kernels by value (repo Views pattern).
struct MarkerViews {
    float* x;  float* z;
    float* ux; float* uy; float* uz;
    float* w;  float* wd;
    uint32_t* cell;
    uint64_t n;
};

// Owning per-species store.
struct MarkerStore {
    arc::DeviceArray<float>    x, z, ux, uy, uz, w, wd;
    arc::DeviceArray<uint32_t> cell;
    uint64_t n = 0;         // live markers
    uint64_t capacity = 0;

    static constexpr double bytes_per_marker         = 32.0;
    static constexpr double sort_chunk_fraction      = 0.125;  // ≤ 1/8 in flight
    static constexpr double bytes_per_marker_effective =
        bytes_per_marker * (1.0 + sort_chunk_fraction);        // 36 B

    void allocate(uint64_t cap) {
        capacity = cap;
        x  = arc::DeviceArray<float>(cap);
        z  = arc::DeviceArray<float>(cap);
        ux = arc::DeviceArray<float>(cap);
        uy = arc::DeviceArray<float>(cap);
        uz = arc::DeviceArray<float>(cap);
        w  = arc::DeviceArray<float>(cap);
        wd = arc::DeviceArray<float>(cap);
        cell = arc::DeviceArray<uint32_t>(cap);
    }

    MarkerViews views() {
        return { x.data(), z.data(), ux.data(), uy.data(), uz.data(),
                 w.data(), wd.data(), cell.data(), n };
    }

    static double bytes_for(uint64_t nmarkers) {
        return double(nmarkers) * bytes_per_marker_effective;
    }
};

}  // namespace arc2d

#endif  // ARC_PIC2D_PARTICLES2D_HPP
