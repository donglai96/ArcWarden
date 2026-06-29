// ArcWarden — real-space source container (Step 7).
//
// Sources holds what the DEPOSIT step writes: the charge density rho on the
// real [ny][nx] grid. It is kept separate from Fields (what push READS) and
// from the SpectralWorkspace (k-space scratch) — the UPIC qe / fxye / qt split
// (design §6). Mixing them makes the Darwin extension explode; keeping them
// apart means later additions (jx,jy,jz, dcu, amu) are just new members here,
// with the main loop and responsibility boundaries unchanged.
//
//   SourceViews — POD handle pack passed BY VALUE into the deposit kernel.
//   Sources     — host-side owner (DeviceArray), allocated once, zeroed each step.

#ifndef ARC_PIC_SOURCES_HPP
#define ARC_PIC_SOURCES_HPP

#include "pic/device_array.hpp"
#include "pic/grid.hpp"

namespace arc {

// Kernel-facing handle bundle (trivially copyable; no ownership). The Darwin
// members are null DeviceViews in ES mode (never allocated, never read).
struct SourceViews {
    DeviceView<float> rho;
    // Darwin: current density J, acceleration density dcu, momentum flux amu.
    DeviceView<float> Jx, Jy, Jz;
    DeviceView<float> dcux, dcuy, dcuz;
    // amu = momentum-flux tensor, 4 independent comps for 2-1/2D (set in Phase E).
    DeviceView<float> amu0, amu1, amu2, amu3;
};

struct Sources {
    DeviceArray<float> rho;
    // Darwin (allocated only by allocate_em):
    DeviceArray<float> Jx, Jy, Jz;
    DeviceArray<float> dcux, dcuy, dcuz;
    DeviceArray<float> amu0, amu1, amu2, amu3;
    bool em = false;

    Sources() = default;
    explicit Sources(const Grid& g) : rho(g.real_size()) {}

    // Build-empty-then-size pattern (Step 3 §5): for members allocated at init.
    void allocate(const Grid& g) { rho = DeviceArray<float>(g.real_size()); }

    // Darwin extra sources. Call once (after allocate) when running FieldModel::Darwin.
    void allocate_em(const Grid& g) {
        const int n = g.real_size();
        Jx = DeviceArray<float>(n); Jy = DeviceArray<float>(n); Jz = DeviceArray<float>(n);
        dcux = DeviceArray<float>(n); dcuy = DeviceArray<float>(n); dcuz = DeviceArray<float>(n);
        amu0 = DeviceArray<float>(n); amu1 = DeviceArray<float>(n);
        amu2 = DeviceArray<float>(n); amu3 = DeviceArray<float>(n);
        em = true;
    }

    // Clear rho before each deposit (deposit accumulates with atomicAdd).
    void zero(cudaStream_t stream = nullptr) { rho.zero(stream); }

    // Clear charge + current before the ρ/J deposit (Darwin).
    void zero_rho_j(cudaStream_t stream = nullptr) {
        rho.zero(stream); Jx.zero(stream); Jy.zero(stream); Jz.zero(stream);
    }

    // Clear the Darwin transverse moments before each ndc deposit.
    void zero_dcu_amu(cudaStream_t stream = nullptr) {
        zero_dcu(stream);
        amu0.zero(stream); amu1.zero(stream); amu2.zero(stream); amu3.zero(stream);
    }

    // Clear only dcu (re-deposited each ndc iteration; amu is deposited once).
    void zero_dcu(cudaStream_t stream = nullptr) {
        dcux.zero(stream); dcuy.zero(stream); dcuz.zero(stream);
    }

    SourceViews views() {
        SourceViews v;
        v.rho = rho.view();
        if (em) {
            v.Jx = Jx.view(); v.Jy = Jy.view(); v.Jz = Jz.view();
            v.dcux = dcux.view(); v.dcuy = dcuy.view(); v.dcuz = dcuz.view();
            v.amu0 = amu0.view(); v.amu1 = amu1.view();
            v.amu2 = amu2.view(); v.amu3 = amu3.view();
        }
        return v;
    }
};

} // namespace arc

#endif // ARC_PIC_SOURCES_HPP
