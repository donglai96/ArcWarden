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

// Kernel-facing handle bundle (trivially copyable; no ownership). In Darwin mode
// (Ex,Ey,Ez) hold the TOTAL electric field E_total = E_L + E_T that the push
// gathers; B{x,y,z} is the magnetic field. ES mode uses only Ex,Ey.
struct FieldViews {
    DeviceView<float> Ex, Ey, Ez;     // ES: Ex,Ey. Darwin: E_total components.
    DeviceView<float> Bx, By, Bz;     // Darwin magnetic field
};

struct Fields {
    DeviceArray<float> Ex, Ey;
    // Darwin (allocated only by allocate_em):
    DeviceArray<float> Ez, Bx, By, Bz;
    DeviceArray<float> ELx, ELy;        // longitudinal E_L (kept to re-form E_total each ndc)
    DeviceArray<float> ETx, ETy, ETz;   // retained transverse E_T (carried across steps)
    bool em = false;

    Fields() = default;
    explicit Fields(const Grid& g) : Ex(g.real_size()), Ey(g.real_size()) {}

    void allocate(const Grid& g) {
        Ex = DeviceArray<float>(g.real_size());
        Ey = DeviceArray<float>(g.real_size());
    }

    // Darwin extra fields. E_T arrays are zeroed (the ndc iteration starts from
    // the retained E_T; at t=0 it is 0).
    void allocate_em(const Grid& g, cudaStream_t stream = nullptr) {
        const int n = g.real_size();
        Ez = DeviceArray<float>(n);
        Bx = DeviceArray<float>(n); By = DeviceArray<float>(n); Bz = DeviceArray<float>(n);
        ELx = DeviceArray<float>(n); ELy = DeviceArray<float>(n);
        ETx = DeviceArray<float>(n); ETy = DeviceArray<float>(n); ETz = DeviceArray<float>(n);
        ETx.zero(stream); ETy.zero(stream); ETz.zero(stream);
        em = true;
    }

    // Optional: clear before a solve (the solver overwrites E, so not required
    // every step, but handy for init / tests).
    void zero(cudaStream_t stream = nullptr) {
        Ex.zero(stream);
        Ey.zero(stream);
    }

    FieldViews views() {
        FieldViews v;
        v.Ex = Ex.view(); v.Ey = Ey.view();
        if (em) {
            v.Ez = Ez.view();
            v.Bx = Bx.view(); v.By = By.view(); v.Bz = Bz.view();
        }
        return v;
    }
};

} // namespace arc

#endif // ARC_PIC_FIELDS_HPP
