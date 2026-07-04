// ArcWarden diagnostic — PhaseFrames: generic (x, vx) phase-space CSV frames + a
// manifest, the electrostatic-friendly output. (Lifted from run_deck.cu dump_phase.)
// Writes <outdir>/frame_%04d.csv and <outdir>/manifest.csv; caps at 100k points/frame
// via a fixed random sample. Cadence = [time] dump_every (default nsteps/100).

#ifndef ARC_PIC_DIAG_PHASE_FRAMES_HPP
#define ARC_PIC_DIAG_PHASE_FRAMES_HPP

#include "pic/diag/module.hpp"
#include "pic/deck.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>

namespace arc { namespace diag {

class PhaseFrames : public IDiag {
    static constexpr int CAP = 100000;
    long   stride_;
    double dt_;
    std::string dir_;
    std::vector<int> idx_;       // fixed sample of particle indices
    std::ofstream man_;
    int    frame_ = 0;
    bool   inited_ = false;
    std::vector<float> x_, vx_;
public:
    PhaseFrames(const Deck& d, const std::string& outdir)
        : stride_(d.dump_every > 0 ? d.dump_every
                                   : (d.rp.nsteps / 100 > 0 ? d.rp.nsteps / 100 : 1)),
          dt_(d.rp.dt), dir_(outdir.empty() ? "." : outdir) {
        std::filesystem::create_directories(dir_);
        man_.open(dir_ + "/manifest.csv");
        if (man_) man_ << "frame,step,time,file\n";
    }
    void sample(const DiagFrame& fr) override {
        if (fr.step % stride_ != 0) return;
        const int N = (int)fr.p.n;
        if (!inited_) {                                  // fixed random subsample (≤ CAP)
            idx_.resize(N); std::iota(idx_.begin(), idx_.end(), 0);
            if (N > CAP) { std::mt19937 rng(12345); std::shuffle(idx_.begin(), idx_.end(), rng);
                           idx_.resize(CAP); std::sort(idx_.begin(), idx_.end()); }
            inited_ = true;
        }
        cudaDeviceSynchronize();
        x_.resize(N); vx_.resize(N);
        cudaMemcpy(x_.data(),  fr.p.x.data(),  fr.p.x.bytes(),  cudaMemcpyDeviceToHost);
        cudaMemcpy(vx_.data(), fr.p.ux.data(), fr.p.ux.bytes(), cudaMemcpyDeviceToHost);
        char name[64]; std::snprintf(name, sizeof name, "frame_%04d.csv", frame_);
        { std::ofstream o(dir_ + "/" + name); o << "x,vx\n";
          for (int i : idx_) {
              float xd = x_[i];
              if (xd < 0.0f) xd += fr.g.nx; else if (xd >= fr.g.nx) xd -= fr.g.nx;
              o << xd * fr.g.dx << ',' << vx_[i] << '\n';
          } }
        if (man_) { man_ << frame_ << ',' << fr.step << ',' << fr.t << ',' << name << '\n'; man_.flush(); }
        ++frame_;
    }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_PHASE_FRAMES_HPP
