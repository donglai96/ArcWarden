// ArcWarden — real-space field container (Step 7).
//
// Fields holds what the PUSH step reads: the electrostatic field components
// Ex, Ey on the real [ny][nx] grid (filled by the solver after rho -> E).
// Separate from Sources (deposit writes rho) and SpectralWorkspace (k-space),
// per the UPIC qe / fxye / qt split (design §6). Ez ≡ 0 in v1 electrostatic;
// the magnetic / Darwin fields (Ez, Bx, By, Bz) are reserved for later and are
// just new members here when the time comes.
//
//   FieldViews — POD handle pack passed BY VALUE into the push (gather) kernel.
//   Fields     — host-side owner (DeviceArray), allocated once.

#ifndef ARC_PIC_FIELDS_HPP
#define ARC_PIC_FIELDS_HPP

#include "pic/device_array.hpp"
#include "pic/grid.hpp"

namespace arc {

// Kernel-facing handle bundle (trivially copyable; no ownership).
struct FieldViews {
    DeviceView<float> Ex, Ey;
    // Darwin later: Ez, Bx, By, Bz
};

struct Fields {
    DeviceArray<float> Ex, Ey;
    // Darwin later: Ez, Bx, By, Bz

    Fields() = default;
    explicit Fields(const Grid& g) : Ex(g.real_size()), Ey(g.real_size()) {}

    void allocate(const Grid& g) {
        Ex = DeviceArray<float>(g.real_size());
        Ey = DeviceArray<float>(g.real_size());
    }

    // Optional: clear before a solve (the solver overwrites E, so not required
    // every step, but handy for init / tests).
    void zero(cudaStream_t stream = nullptr) {
        Ex.zero(stream);
        Ey.zero(stream);
    }

    FieldViews views() { return FieldViews{ Ex.view(), Ey.view() }; }
};

} // namespace arc

#endif // ARC_PIC_FIELDS_HPP
