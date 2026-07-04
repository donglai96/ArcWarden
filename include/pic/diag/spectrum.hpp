// ArcWarden diagnostic — Spectrum: console line with whistler mode power + δB/B0.
// (Lifted from whistler_pump.cu `sample()`.) Prints |E_M|², |E_2M|², |E_3M|², the
// nonlinear-structure band power, and δB_rms/B0 every nsteps/40 steps.

#ifndef ARC_PIC_DIAG_SPECTRUM_HPP
#define ARC_PIC_DIAG_SPECTRUM_HPP

#include "pic/diag/module.hpp"
#include "pic/deck.hpp"

namespace arc { namespace diag {

class SpectrumDiag : public IDiag {
    int  nx_, M_, blo_, bhi_;
    long every_;
    std::vector<float> ex_, bx_, by_, bz_;
public:
    explicit SpectrumDiag(const Deck& d)
        : nx_(d.nx), M_(d.pump_M), blo_(d.band_lo), bhi_(d.band_hi),
          every_(d.rp.nsteps / 40 > 0 ? d.rp.nsteps / 40 : 1), ex_(d.nx) {}

    void sample(const DiagFrame& fr) override {
        if (fr.step % every_ != 0) return;
        cudaDeviceSynchronize();
        const int N = fr.g.real_size();
        cudaMemcpy(ex_.data(), fr.f.Ex.data(), nx_ * sizeof(float), cudaMemcpyDeviceToHost);
        const double pM  = mode_power(ex_, nx_, M_ > 0 ? M_ : 1);
        const double p2M = mode_power(ex_, nx_, M_ > 0 ? 2*M_ : 2);
        const double p3M = mode_power(ex_, nx_, M_ > 0 ? 3*M_ : 3);
        double pB = 0; for (int m = blo_; m <= bhi_; ++m) pB += mode_power(ex_, nx_, m);
        double dBB0 = 0.0;
        if (fr.f.Bx.size() > 0) {                       // magnetic field present (Darwin)
            bx_.resize(N); by_.resize(N); bz_.resize(N);
            cudaMemcpy(bx_.data(), fr.f.Bx.data(), fr.f.Bx.bytes(), cudaMemcpyDeviceToHost);
            cudaMemcpy(by_.data(), fr.f.By.data(), fr.f.By.bytes(), cudaMemcpyDeviceToHost);
            cudaMemcpy(bz_.data(), fr.f.Bz.data(), fr.f.Bz.bytes(), cudaMemcpyDeviceToHost);
            double db = 0;
            for (int i = 0; i < N; ++i) {
                const double dbx = bx_[i]-fr.rp.B0[0], dby = by_[i]-fr.rp.B0[1], dbz = bz_[i]-fr.rp.B0[2];
                db += dbx*dbx + dby*dby + dbz*dbz;
            }
            dBB0 = std::sqrt(db / N) / (fr.bmag > 0 ? fr.bmag : 1.0);
        }
        std::printf("t=%7.1f  |E_M|2=%.3e |E_2M|2=%.3e |E_3M|2=%.3e  band[%d-%d]=%.3e  dB/B0=%.4f\n",
                    fr.t, pM, p2M, p3M, blo_, bhi_, pB, dBB0);
    }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_SPECTRUM_HPP
