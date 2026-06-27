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

// Kernel-facing handle bundle (trivially copyable; no ownership).
struct SourceViews {
    DeviceView<float> rho;
    // Darwin later: jx, jy, jz, dcu*, amu*
};

struct Sources {
    DeviceArray<float> rho;
    // Darwin later: jx, jy, jz, dcu*, amu*

    Sources() = default;
    explicit Sources(const Grid& g) : rho(g.real_size()) {}

    // Build-empty-then-size pattern (Step 3 §5): for members allocated at init.
    void allocate(const Grid& g) { rho = DeviceArray<float>(g.real_size()); }

    // Clear rho before each deposit (deposit accumulates with atomicAdd).
    void zero(cudaStream_t stream = nullptr) { rho.zero(stream); }

    SourceViews views() { return SourceViews{ rho.view() }; }
};

} // namespace arc

#endif // ARC_PIC_SOURCES_HPP
