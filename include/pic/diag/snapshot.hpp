// ArcWarden diagnostic — Snapshot: at t_snap, dump the paper field+phase figure
// (<pref>fields.csv: x, Epar, Eperp, Ey, dBy, dBmag ; <pref>fhist.bin: f(x,v_∥/v_th)
// over ALL particles ; <pref>fhist.meta). (Lifted from whistler_pump.cu.)

#ifndef ARC_PIC_DIAG_SNAPSHOT_HPP
#define ARC_PIC_DIAG_SNAPSHOT_HPP

#include "pic/diag/module.hpp"
#include "pic/deck.hpp"
#include <cstdio>
#include <fstream>

namespace arc { namespace diag {

class Snapshot : public IDiag {
    static constexpr int NXB = 512, NVB = 480;
    static constexpr double VLO = -6.0, VHI = 6.0;   // v_∥/v_th
    int    nx_;
    long   snap_step_, last_step_;
    double dx_, Lx_;
    std::string pref_;
    bool   done_ = false;
public:
    Snapshot(const Deck& d, const std::string& pref)
        : nx_(d.nx), snap_step_((long)std::lround((d.tsnap > 0 ? d.tsnap : d.rp.nsteps*d.rp.dt) / d.rp.dt)),
          last_step_(d.rp.nsteps), dx_(d.Lx / d.nx), Lx_(d.Lx), pref_(pref) {}

    void sample(const DiagFrame& fr) override {
        // fire at t_snap, or at the final step if the run ends before t_snap (fallback).
        if (done_ || (fr.step < snap_step_ && fr.step < last_step_)) return;
        done_ = true;
        cudaDeviceSynchronize();
        const Fields& f = fr.f;
        const bool em = f.Ez.size() > 0;
        std::vector<float> fex(nx_), fey(nx_), fez(nx_, 0), fbx(nx_, 0), fby(nx_, 0), fbz(nx_, 0);
        cudaMemcpy(fex.data(), f.Ex.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fey.data(), f.Ey.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        if (em) {
            cudaMemcpy(fez.data(), f.Ez.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(fbx.data(), f.Bx.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(fby.data(), f.By.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(fbz.data(), f.Bz.data(), nx_*sizeof(float), cudaMemcpyDeviceToHost);
        }
        { std::ofstream o(pref_ + "fields.csv"); o << "x,Epar,Eperp,Ey,dBy,dBmag\n";
          for (int i = 0; i < nx_; ++i) {
              const double Epar  = fex[i]*fr.cost + fez[i]*fr.sint;
              const double Eperp = fey[i];
              const double dbx = fbx[i]-fr.rp.B0[0], dby = fby[i]-fr.rp.B0[1], dbz = fbz[i]-fr.rp.B0[2];
              o << i*dx_ << ',' << Epar << ',' << Eperp << ',' << fey[i] << ','
                << dby << ',' << std::sqrt(dbx*dbx+dby*dby+dbz*dbz) << '\n';
          } }
        // f(x, v_∥/v_th) over ALL particles (host histogram)
        const Particles& P = fr.p;
        const int Np = (int)P.n;
        std::vector<float> px(Np), pvx(Np), pvz(Np);
        cudaMemcpy(px.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost);
        cudaMemcpy(pvx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(pvz.data(), P.uz.data(), P.uz.bytes(), cudaMemcpyDeviceToHost);
        std::vector<float> hist((size_t)NXB*NVB, 0.0f);
        for (int i = 0; i < Np; ++i) {
            double xx = px[i]*dx_; if (xx < 0) xx += Lx_; else if (xx >= Lx_) xx -= Lx_;
            const double vp = (pvx[i]*fr.cost + pvz[i]*fr.sint) / fr.vth;
            int ix = (int)(xx / Lx_ * NXB), iv = (int)((vp - VLO) / (VHI - VLO) * NVB);
            if (ix>=0 && ix<NXB && iv>=0 && iv<NVB) hist[(size_t)iv*NXB + ix] += 1.0f;
        }
        std::ofstream hb(pref_ + "fhist.bin", std::ios::binary);
        hb.write((char*)hist.data(), hist.size()*sizeof(float));
        { std::ofstream m(pref_ + "fhist.meta");
          m << "nxb "<<NXB<<"\nnvb "<<NVB<<"\nLx "<<Lx_<<"\nvlo "<<VLO<<"\nvhi "<<VHI
            <<"\nvr "<<fr.vr<<"\ntsnap "<<fr.t<<"\n"; }
        std::printf("  [snapshot] wrote %sfields.csv + %sfhist.bin at t=%.1f (%dx%d over %d particles)\n",
                    pref_.c_str(), pref_.c_str(), fr.t, NXB, NVB, Np);
    }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_SNAPSHOT_HPP
