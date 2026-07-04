// ArcWarden diagnostic — PhaseVideo: per-frame f(x, v_∥/v_th) histogram + field rows
// (E∥, E⊥, δB_y) for the time-evolution movie. (Lifted from whistler_pump.cu.)
// Writes <pref>vid_phase.bin, <pref>vid_field.bin, <pref>vid.meta.

#ifndef ARC_PIC_DIAG_PHASE_VIDEO_HPP
#define ARC_PIC_DIAG_PHASE_VIDEO_HPP

#include "pic/diag/module.hpp"
#include "pic/deck.hpp"
#include <cstdio>
#include <fstream>

namespace arc { namespace diag {

class PhaseVideo : public IDiag {
    static constexpr int   NXB = 256, NVB = 200;
    static constexpr float VLO = -6.0f, VHI = 6.0f;   // v_∥/v_th
    int    nx_, stride_;
    double toff_, Lx_, dt_;
    std::string meta_;
    FILE  *vph_ = nullptr, *vfl_ = nullptr;
    int    nframe_ = 0;
    double vr_ = 0.0;
    DeviceArray<float> dhist_;
    std::vector<float> hh_, ex_, ey_, ez_, by_;
public:
    PhaseVideo(const Deck& d, const std::string& pref)
        : nx_(d.nx), stride_(d.rp.nsteps / (d.n_frames > 0 ? d.n_frames : 150) > 0
                             ? (int)(d.rp.nsteps / (d.n_frames > 0 ? d.n_frames : 150)) : 1),
          toff_(d.rp.pump_toff), Lx_(d.Lx), dt_(d.rp.dt), meta_(pref + "vid.meta"),
          dhist_((std::size_t)NXB * NVB), hh_((std::size_t)NXB * NVB),
          ex_(d.nx), ey_(d.nx), ez_(d.nx), by_(d.nx) {
        vph_ = std::fopen((pref + "vid_phase.bin").c_str(), "wb");
        vfl_ = std::fopen((pref + "vid_field.bin").c_str(), "wb");
    }
    void sample(const DiagFrame& fr) override {
        if (fr.step % stride_ != 0 || !vph_ || !vfl_) return;
        vr_ = fr.vr;
        dhist_.zero();
        const int thr = 256, blk = (int)((fr.p.n + thr - 1) / thr);
        phase_hist_kernel<<<blk, thr>>>(fr.p.x.data(), fr.p.ux.data(), fr.p.uz.data(),
                                        (long)fr.p.n, dhist_.data(), NXB, NVB, nx_,
                                        (float)fr.cost, (float)fr.sint, (float)fr.vth, VLO, VHI);
        cudaMemcpy(hh_.data(), dhist_.data(), dhist_.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(ex_.data(), fr.f.Ex.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ey_.data(), fr.f.Ey.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        const bool em = fr.f.Ez.size() > 0;
        if (em) cudaMemcpy(ez_.data(), fr.f.Ez.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        if (em) cudaMemcpy(by_.data(), fr.f.By.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(hh_.data(), sizeof(float), hh_.size(), vph_);
        std::vector<float> row(3*nx_);
        for (int i = 0; i < nx_; ++i) {
            const double ezi = em ? ez_[i] : 0.0, byi = em ? by_[i] : 0.0;
            row[i]       = (float)(ex_[i]*fr.cost + ezi*fr.sint);  // E_∥
            row[nx_+i]   = ey_[i];                                  // E_⊥
            row[2*nx_+i] = (float)(byi - fr.rp.B0[1]);              // δB_y
        }
        std::fwrite(row.data(), sizeof(float), row.size(), vfl_);
        ++nframe_;
    }
    void finalize() override {
        if (vph_) { std::fclose(vph_); vph_ = nullptr; }
        if (vfl_) { std::fclose(vfl_); vfl_ = nullptr; }
        std::ofstream o(meta_);
        o << "nframe " << nframe_ << "\nnxb " << NXB << "\nnvb " << NVB << "\nnx " << nx_
          << "\nLx " << Lx_ << "\nvlo " << VLO << "\nvhi " << VHI << "\ntoff " << toff_
          << "\ndt_frame " << stride_*dt_ << "\nvr " << vr_ << "\n";
    }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_PHASE_VIDEO_HPP
