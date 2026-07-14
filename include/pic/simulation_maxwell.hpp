// ArcWarden — L-shell plan M1: MaxwellSimulation<Cfg> main loop (Yee branch).
//
// Step order (plan §12.1, reduced to the M1 periodic feature set):
//   1. B half step                    (Faraday, dt/2 → B at t^n)
//   2. fused gather + Boris push + Esirkepov J deposit  (E^n, B^n → u^{n+½},
//      x^{n+1}, J^{n+½})
//   3. particle migrate (periodic wrap + cell recompute)
//   4. B half step                    (→ B^{n+½})
//   5. E full step                    (Ampère with J^{n+½} → E^{n+1})
// Boundaries (M2), injection (M7), δf (M3 on this branch) plug in later.
//
// The spectral Simulation<Cfg> is untouched; Particles/Grid/RunParams shared.

#ifndef ARC_PIC_SIMULATION_MAXWELL_HPP
#define ARC_PIC_SIMULATION_MAXWELL_HPP

#include "pic/yee2d.hpp"

namespace arc {

class MaxwellSimulation {
public:
    MaxwellSimulation(const Grid& g, const RunParams& rp)
        : g_(g), rp_(rp), flds_(g, rp.c, rp.dt), diag_(2) {}

    Grid&       grid()      { return g_; }
    RunParams&  params()    { return rp_; }
    Particles&  particles() { return parts_; }
    YeeFields&  fields()    { return flds_; }
    CudaStream& stream()    { return s_; }

    void step() {
        const dim3 tb(16, 16);
        const dim3 nb((g_.nx + 15) / 16, (g_.ny + 15) / 16);
        YeeViews v = flds_.views();
        const float dt2 = 0.5f * (float)rp_.dt;

        yee::k_faraday<<<nb, tb, 0, s_>>>(v, dt2);
        if (parts_.n > 0) {
            flds_.zero_j(s_);
            const int threads = 256;
            const int blocks = ((int)parts_.n + threads - 1) / threads;
            yee::k_push_esirkepov<<<blocks, threads, 0, s_>>>(parts_.views(), v, rp_);
            parts_.migrate(g_, s_);
        }
        yee::k_faraday<<<nb, tb, 0, s_>>>(v, dt2);
        yee::k_ampere<<<nb, tb, 0, s_>>>(v);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    struct Energies { double we, wb; };
    Energies field_energy() {
        CUDA_CHECK(cudaMemsetAsync(diag_.data(), 0, diag_.bytes(), s_));
        const int n = g_.real_size();
        yee::k_energy<<<(n + 255) / 256, 256, 0, s_>>>(flds_.views(), diag_.data());
        double h[2];
        CUDA_CHECK(cudaMemcpy(h, diag_.data(), sizeof(h), cudaMemcpyDeviceToHost));
        return {h[0], h[1]};
    }

private:
    Grid       g_;
    RunParams  rp_;
    Particles  parts_;
    YeeFields  flds_;
    CudaStream s_;
    DeviceArray<double> diag_;
};

} // namespace arc

#endif // ARC_PIC_SIMULATION_MAXWELL_HPP
