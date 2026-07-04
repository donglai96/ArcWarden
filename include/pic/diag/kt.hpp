// ArcWarden diagnostic — KTDump: δE_L(x,t) time series for the k-t and ω-k plots.
// (Lifted from whistler_pump.cu.) Appends the pure longitudinal field ELx every
// kt_stride steps to <pref>kt.bin, and writes <pref>kt.meta on finalize.

#ifndef ARC_PIC_DIAG_KT_HPP
#define ARC_PIC_DIAG_KT_HPP

#include "pic/diag/module.hpp"
#include "pic/deck.hpp"
#include <cstdio>
#include <fstream>

namespace arc { namespace diag {

class KTDump : public IDiag {
    int    nx_, stride_, M_, blo_, bhi_;
    double w0_, k0_, toff_, Lx_, dt_;
    std::string meta_;
    FILE*  fp_ = nullptr;
    int    nsamp_ = 0;
    std::vector<float> elx_;
public:
    KTDump(const Deck& d, const std::string& pref)
        : nx_(d.nx), stride_(d.kt_stride > 0 ? d.kt_stride : 5), M_(d.pump_M),
          blo_(d.band_lo), bhi_(d.band_hi), w0_(d.rp.pump_w0), k0_(d.rp.pump_k0),
          toff_(d.rp.pump_toff), Lx_(d.Lx), dt_(d.rp.dt),
          meta_(pref + "kt.meta"), elx_(d.nx) {
        fp_ = std::fopen((pref + "kt.bin").c_str(), "wb");
    }
    void sample(const DiagFrame& fr) override {
        if (fr.step % stride_ != 0 || !fp_ || fr.f.ELx.size() == 0) return;
        cudaDeviceSynchronize();
        cudaMemcpy(elx_.data(), fr.f.ELx.data(), nx_ * sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(elx_.data(), sizeof(float), nx_, fp_); ++nsamp_;
    }
    void finalize() override {
        if (fp_) { std::fclose(fp_); fp_ = nullptr; }
        std::ofstream o(meta_);
        o << "nx " << nx_ << "\nnsamp " << nsamp_ << "\ndt_sample " << stride_*dt_
          << "\ntoff " << toff_ << "\nLx " << Lx_ << "\nM " << M_ << "\nw0 " << w0_
          << "\nband_lo " << blo_ << "\nband_hi " << bhi_
          << "\nvphx " << (k0_ > 0 ? w0_/k0_ : 0.0) << "\n";
    }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_KT_HPP
