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

#include <cstring>
#include <stdexcept>
#include <vector>

namespace arc {

class MaxwellSimulation {
public:
    MaxwellSimulation(const Grid& g, const RunParams& rp)
        : g_(g), rp_(rp), flds_(g, rp.c, rp.dt), diag_(2),
          jtmp_(rp.jfilter > 0 ? g.real_size() : 0),
          mask_n_(rp.bnd_x ? g.nx : 0), mask_h_(rp.bnd_x ? g.nx : 0),
          vcy_(rp.cold_nc > 0.0 ? g.real_size() : 0),
          vcz_(rp.cold_nc > 0.0 ? g.real_size() : 0) {
        if (rp_.bnd_x) build_masks();
        if (rp_.cold_nc > 0.0) {
            if (g.ny != 1)
                throw std::runtime_error("cold_nc: the linearized cold fluid is "
                                         "ny = 1 only (Ey/Ez stagger degeneracy)");
            vcy_.zero(s_); vcz_.zero(s_);
        }
    }

    Grid&       grid()      { return g_; }
    RunParams&  params()    { return rp_; }
    Particles&  particles() { return parts_; }
    YeeFields&  fields()    { return flds_; }
    CudaStream& stream()    { return s_; }
    DeviceArray<float>& vcy() { return vcy_; }   // M4 cold fluid (tests/seeding)
    DeviceArray<float>& vcz() { return vcz_; }

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
        if (rp_.ant_amp != 0.0) {          // M2/M10 antenna current column
            if (parts_.n == 0) flds_.zero_j(s_);   // vacuum runs skip the deposit block
            yee::k_antenna<<<nb, tb, 0, s_>>>(v, rp_, tnow);
        }
        if (rp_.cold_nc > 0.0) {           // M4 linearized cold fluid (chirp1d port)
            if (parts_.n == 0 && rp_.ant_amp == 0.0) flds_.zero_j(s_);
            yee::k_cold_fluid<<<nb, tb, 0, s_>>>(v, rp_, vcy_.data(), vcz_.data());
        }
        yee::k_faraday<<<nb, tb, 0, s_>>>(v, dt2);
        yee::k_ampere<<<nb, tb, 0, s_>>>(v);
        if (rp_.bnd_x)     // M2 absorbing layers: damp wave fields at x ends
            yee::k_damp_x<<<nb, tb, 0, s_>>>(v, mask_n_.data(), mask_h_.data(),
                                             rp_.cold_nc > 0.0 ? vcy_.data() : nullptr,
                                             rp_.cold_nc > 0.0 ? vcz_.data() : nullptr);
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
                               0, s_>>>(parts_.views(), v, rho_.data(), rp_.qm,
                                        rp_.deltaf);
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

    // M3 weight diagnostics: Σwd (systematic drift), rms(wd), max|wd|
    // (FP64 reductions; plan M3 权重诊断).
    struct WdStats { double sum, rms, max; };
    WdStats wd_stats() {
        if (!rp_.deltaf || parts_.n == 0) return {0, 0, 0};
        if (wdiag_.size() == 0) { wdiag_ = DeviceArray<double>(2); wmax_ = DeviceArray<unsigned int>(1); }
        CUDA_CHECK(cudaMemsetAsync(wdiag_.data(), 0, wdiag_.bytes(), s_));
        CUDA_CHECK(cudaMemsetAsync(wmax_.data(), 0, wmax_.bytes(), s_));
        const int threads = 256;
        yee::k_wd_stats<<<((int)parts_.n + threads - 1) / threads, threads, 0, s_>>>(
            parts_.views(), wdiag_.data(), wmax_.data());
        double h[2]; unsigned int hm;
        CUDA_CHECK(cudaMemcpy(h, wdiag_.data(), sizeof(h), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hm, wmax_.data(), sizeof(hm), cudaMemcpyDeviceToHost));
        float fm; std::memcpy(&fm, &hm, sizeof(fm));
        return {h[0], std::sqrt(h[1] / (double)parts_.n), (double)fm};
    }

private:
    // M2 damping-layer masks: exp(-numax·d²·dt), d = depth into the bnd_nd-cell
    // layer normalized to [0,1] (chirp1d build_masks, ported). mask_n at
    // integer x sites, mask_h at half-integer sites.
    void build_masks() {
        std::vector<float> mn(g_.nx), mh(g_.nx);
        const int nd = rp_.bnd_nd;
        auto mval = [&](double xi) {
            double d = 0.0;
            if (xi < nd)                 d = (nd - xi) / (double)nd;
            else if (xi > g_.nx - nd)    d = (xi - (g_.nx - nd)) / (double)nd;
            if (d <= 0.0) return 1.0;
            return std::exp(-rp_.bnd_numax * d * d * rp_.dt);
        };
        for (int i = 0; i < g_.nx; ++i) {
            mn[i] = (float)mval(i);
            mh[i] = (float)mval(i + 0.5);
        }
        CUDA_CHECK(cudaMemcpy(mask_n_.data(), mn.data(), mn.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mask_h_.data(), mh.data(), mh.size() * 4, cudaMemcpyHostToDevice));
    }

    Grid       g_;
    RunParams  rp_;
    Particles  parts_;
    YeeFields  flds_;
    CudaStream s_;
    DeviceArray<double> diag_;
    DeviceArray<float>  jtmp_;   // binomial-filter scratch (empty if jfilter=0)
    DeviceArray<float>  rho_, rres_, r0_;   // residuals() scratch (lazy)
    DeviceArray<float>  mask_n_, mask_h_;   // M2 x-damping masks (empty if bnd_x=0)
    DeviceArray<float>  vcy_, vcz_;         // M4 cold-fluid velocity (empty if cold_nc=0)
    DeviceArray<double> wdiag_;             // M3 wd-stats scratch (lazy)
    DeviceArray<unsigned int> wmax_;
    bool have_ref_ = false;      // Gauss reference residual captured?
    long nstep_ = 0;
    long next_sort_ = 0;         // next tile re-sort step (tile_sort path)
};

} // namespace arc

#endif // ARC_PIC_SIMULATION_MAXWELL_HPP
