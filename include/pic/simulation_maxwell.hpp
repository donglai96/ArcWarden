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
        : g_(g), rp_(rp), flds_(g, rp.c, rp.dt), diag_(2),
          jtmp_(rp.jfilter > 0 ? g.real_size() : 0) {}

    Grid&       grid()      { return g_; }
    RunParams&  params()    { return rp_; }
    Particles&  particles() { return parts_; }
    YeeFields&  fields()    { return flds_; }
    CudaStream& stream()    { return s_; }

    void step() { step_at(nstep_ * rp_.dt); ++nstep_; }

    void step_at(double tnow) {
        const dim3 tb(16, 16);
        const dim3 nb((g_.nx + 15) / 16, (g_.ny + 15) / 16);
        YeeViews v = flds_.views();
        const float dt2 = 0.5f * (float)rp_.dt;

        yee::k_faraday<<<nb, tb, 0, s_>>>(v, dt2);
        if (parts_.n > 0) {
            flds_.zero_j(s_);
            const int threads = 256;
            if (rp_.tile_sort > 0) {
                // tiled path: periodic physical sort + shared-memory deposit
                if (nstep_ >= next_sort_) {
                    parts_.sort_by_tile(g_, 16, 16, s_);
                    next_sort_ = nstep_ + rp_.tile_sort;
                }
                constexpr int bpt = 2;   // blocks per tile (fill the GPU)
                const BinViews b = parts_.bins();
                yee::k_push_esirkepov_tiled<16, 16, 2><<<b.ntiles * bpt, threads, 0, s_>>>(
                    parts_.views(), b, v, rp_, tnow, bpt);
                // migrate is fused into the tiled kernel (wrap + cell recompute)
            } else {
                const int blocks = ((int)parts_.n + threads - 1) / threads;
                yee::k_push_esirkepov<<<blocks, threads, 0, s_>>>(parts_.views(), v, rp_, tnow);
                parts_.migrate(g_, s_);
            }
            // binomial J smoothing (rp.jfilter passes per component; OSIRIS "smooth")
            if (rp_.jfilter > 0) {
                float* comps[3] = {flds_.jx_.data(), flds_.jy_.data(), flds_.jz_.data()};
                const size_t nb_j = (size_t)g_.real_size() * sizeof(float);
                for (float* jc : comps) {
                    float *src = jc, *dst = jtmp_.data();
                    for (int pass = 0; pass < rp_.jfilter; ++pass) {
                        yee::k_binomial3x3<<<nb, tb, 0, s_>>>(src, dst, v);
                        float* t2 = src; src = dst; dst = t2;
                    }
                    if (src != jc)   // odd pass count: result sits in the scratch
                        CUDA_CHECK(cudaMemcpyAsync(jc, src, nb_j,
                                                   cudaMemcpyDeviceToDevice, s_));
                }
            }
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

    // M1 running conservation diagnostics: rms divB and the DRIFT of the Gauss
    // residual r = divE - rho from its value at the first call (which absorbs
    // the neutralizing background and the skipped initial Poisson solve).
    // rho gets the same binomial passes as J so the comparison is against the
    // equally-filtered charge (see k_binomial3x3). Call between steps.
    struct Residuals { double divb_rms, gauss_drift_rms; };
    Residuals residuals() {
        const int n = g_.real_size();
        if (rho_.size() == 0) {
            rho_  = DeviceArray<float>(n);
            rres_ = DeviceArray<float>(n);
            r0_   = DeviceArray<float>(n);
        }
        YeeViews v = flds_.views();
        const dim3 tb(16, 16);
        const dim3 nb((g_.nx + 15) / 16, (g_.ny + 15) / 16);
        rho_.zero(s_);
        if (parts_.n > 0) {
            const int threads = 256;
            yee::k_rho_nodes<<<((int)parts_.n + threads - 1) / threads, threads,
                               0, s_>>>(parts_.views(), v, rho_.data(), rp_.qm);
        }
        if (rp_.jfilter > 0) {
            float *src = rho_.data(), *dst = jtmp_.data();
            for (int pass = 0; pass < rp_.jfilter; ++pass) {
                yee::k_binomial3x3<<<nb, tb, 0, s_>>>(src, dst, v);
                float* t2 = src; src = dst; dst = t2;
            }
            if (src != rho_.data())
                CUDA_CHECK(cudaMemcpyAsync(rho_.data(), src, (size_t)n * sizeof(float),
                                           cudaMemcpyDeviceToDevice, s_));
        }
        yee::k_gauss_residual<<<nb, tb, 0, s_>>>(v, rho_.data(), rres_.data());
        if (!have_ref_) {
            CUDA_CHECK(cudaMemcpyAsync(r0_.data(), rres_.data(), (size_t)n * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s_));
            have_ref_ = true;
        }
        CUDA_CHECK(cudaMemsetAsync(diag_.data(), 0, diag_.bytes(), s_));
        yee::k_div_stats<<<(n + 255) / 256, 256, 0, s_>>>(v, rres_.data(), r0_.data(),
                                                          diag_.data());
        double h[2];
        CUDA_CHECK(cudaMemcpy(h, diag_.data(), sizeof(h), cudaMemcpyDeviceToHost));
        return {std::sqrt(h[0] / n), std::sqrt(h[1] / n)};
    }

private:
    Grid       g_;
    RunParams  rp_;
    Particles  parts_;
    YeeFields  flds_;
    CudaStream s_;
    DeviceArray<double> diag_;
    DeviceArray<float>  jtmp_;   // binomial-filter scratch (empty if jfilter=0)
    DeviceArray<float>  rho_, rres_, r0_;   // residuals() scratch (lazy)
    bool have_ref_ = false;      // Gauss reference residual captured?
    long nstep_ = 0;
    long next_sort_ = 0;         // next tile re-sort step (tile_sort path)
};

} // namespace arc

#endif // ARC_PIC_SIMULATION_MAXWELL_HPP
