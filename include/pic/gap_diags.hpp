// G1.3 — flagship diagnostics pack (docs/GAP_PLAN.md). ADDITIVE ONLY: nothing
// here touches the simulation loop; runner tools call these explicitly, so
// legacy tools are bit-identical without them.
//
// 1. FvHist  f(vpar, vperp | x-region): the 2D plateau/scar monitor. The
//    (E,mu) scar-transport prediction (gap_scar_transport.py) lives in
//    (vpar, vperp), so the plateau must be resolved in BOTH — a 1D f(vpar)
//    cut cannot separate the Landau plateau from its cyclotron image.
//    vpar/vperp are taken w.r.t. the LOCAL background b-hat (mirror2d aware).
// 2. WorkLedger  resonance-tagged J.E: every acc step it gathers the wave E
//    at each marker (the pusher's own staggered gather) and accumulates
//      q w (E_par v_par)          -> Landau channel
//      q w (E.v)  -  (E_par v_par) -> cyclotron (E_perp) channel
//    into (x-region, vpar) bins. Landau work concentrates at vpar ~ +Vp;
//    cyclotron work at vpar ~ -|V_R|. One figure settles the
//    Li-vs-Chen-vs-Omura division of labor for the 0.5 fce gap.
// 3. Full-2D field snapshots are a plain memcpy+fwrite in the tool (offline
//    WNA maps + E_par(x,y) with the analytic background from meta) — no
//    device code needed here.
//
// File formats (all raw little-endian):
//   fv_XXXXXX.bin  float64 [nreg][npar][nperp]   instantaneous, w-weighted
//   wl_XXXXXX.bin  float64 [nreg][nwb][2]        accumulated since last dump
//                  ([...][0] = Landau, [...][1] = cyclotron), then reset

#pragma once

#include "pic/background_b0.hpp"
#include "pic/particles.hpp"
#include "pic/yee2d.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace arc {
namespace gapdiag {

// local background unit vector at a marker (b0_prof = 3 evaluates the 2D
// slab mirror; 1D profiles have B0 along x; uniform uses the B0 vector)
__device__ inline void bhat(const RunParams& rp, float xph, float yph,
                            float& bx, float& by, float& bz) {
    if (rp.b0_prof == 3) {
        const float Bx = bg::b0x(rp, xph), By = bg::b0y2d(rp, xph, yph);
        const float ib = rsqrtf(Bx * Bx + By * By);
        bx = Bx * ib; by = By * ib; bz = 0.f;
    } else if (rp.b0_prof) {
        bx = 1.f; by = 0.f; bz = 0.f;
    } else {
        const float ib = rsqrtf(rp.B0[0] * rp.B0[0] + rp.B0[1] * rp.B0[1]
                                + rp.B0[2] * rp.B0[2]);
        bx = rp.B0[0] * ib; by = rp.B0[1] * ib; bz = rp.B0[2] * ib;
    }
}

__global__ void k_fvhist(ParticleViews p, RunParams rp, float dxp, float dyp,
                         int nx, double* bins, int nreg, int npar, int nperp,
                         float vmax, long base, long cnt,
                         float y1, float y2) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= cnt) return;
    const long t = base + tid;
    const float x0 = p.x[t], y0 = p.y[t];
    if (y0 < y1 || y0 >= y2) return;      // Phase Q y-window (wrap-layer split)
    float bx, by, bz;
    bhat(rp, x0 * dxp, y0 * dyp, bx, by, bz);
    const float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float gi = rp.rel ? rsqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
    const float vx = ux * gi, vy = uy * gi, vz = uz * gi;
    const float vpar  = vx * bx + vy * by + vz * bz;
    const float vperp = sqrtf(fmaxf(vx * vx + vy * vy + vz * vz
                                    - vpar * vpar, 0.f));
    const int r  = min(nreg - 1, (int)(x0 * (float)nreg / (float)nx));
    const int ip = (int)((vpar + vmax) / (2.f * vmax) * (float)npar);
    const int jp = (int)(vperp / vmax * (float)nperp);
    if (ip < 0 || ip >= npar || jp >= nperp) return;
    atomicAdd(&bins[((long)r * npar + ip) * nperp + jp], (double)p.w[t]);
}

__global__ void k_workledger(ParticleViews p, YeeViews v, RunParams rp,
                             double* wl, int nreg, int nwb, float vmax,
                             float wdt, long base, long cnt) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= cnt) return;
    const long t = base + tid;
    const float x0 = p.x[t], y0 = p.y[t];
    const float Ex = yee::gather_stag(v.ex, v, x0, y0, 0.5f, 0.f);
    const float Ey = yee::gather_stag(v.ey, v, x0, y0, 0.f, 0.5f);
    const float Ez = yee::gather_stag(v.ez, v, x0, y0, 0.f, 0.f);
    float bx, by, bz;
    bhat(rp, x0 * v.dxp, y0 * v.dyp, bx, by, bz);
    const float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float gi = rp.rel ? rsqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
    const float vx = ux * gi, vy = uy * gi, vz = uz * gi;
    const float vpar = vx * bx + vy * by + vz * bz;
    const float epar = Ex * bx + Ey * by + Ez * bz;
    // charge per marker: m = 1 in code units so q = rp.qm; w carries the
    // macro density scale. wdt = dt * acc_every (the represented interval).
    const double qw   = (double)(float)rp.qm * (double)p.w[t] * (double)wdt;
    const double wpar = qw * (double)epar * (double)vpar;
    const double wtot = qw * ((double)Ex * vx + (double)Ey * vy + (double)Ez * vz);
    const int r  = min(nreg - 1, (int)(x0 * (float)nreg / (float)v.nx));
    const int ib = (int)((vpar + vmax) / (2.f * vmax) * (float)nwb);
    if (ib < 0 || ib >= nwb) return;
    const long o = ((long)r * nwb + ib) * 2;
    atomicAdd(&wl[o],     wpar);          // Landau (E_par) channel
    atomicAdd(&wl[o + 1], wtot - wpar);   // cyclotron (E_perp) channel
}

struct GapDiags {
    int   nreg, npar, nperp, nwb;
    float vmax;
    DeviceArray<double> fv;   // [nreg][npar][nperp]
    DeviceArray<double> wl;   // [nreg][nwb][2], accumulated

    GapDiags(int nreg_, int npar_, int nperp_, int nwb_, float vmax_)
        : nreg(nreg_), npar(npar_), nperp(nperp_), nwb(nwb_), vmax(vmax_),
          fv((std::size_t)nreg_ * npar_ * nperp_),
          wl((std::size_t)nreg_ * nwb_ * 2) {
        fv.zero(nullptr);
        wl.zero(nullptr);
    }

    // instantaneous f(vpar, vperp | region) snapshot -> file
    // base < 0: all markers; else the [base, base+cnt) species block
    // (Particles::sp_base/sp_cnt from the multi-species mirror loader)
    void fv_snapshot(Particles& p, const RunParams& rp, const Grid& g,
                     cudaStream_t s, const char* fn,
                     long base = -1, long cnt = -1,
                     float y1 = -1e30f, float y2 = 1e30f) {
        fv.zero(s);
        if (base < 0) { base = 0; cnt = (long)p.n; }
        constexpr int threads = 256;
        const int blocks = (int)((cnt + threads - 1) / threads);
        k_fvhist<<<blocks, threads, 0, s>>>(p.views(), rp, (float)g.dx,
                                            (float)g.dy, g.nx, fv.data(),
                                            nreg, npar, nperp, vmax,
                                            base, cnt, y1, y2);
        CUDA_CHECK(cudaPeekAtLastError());
        write_dev(fv, s, fn);
    }

    // accumulate one ledger sample representing wdt of physical time
    void wl_accum(Particles& p, YeeFields& f, const RunParams& rp, float wdt,
                  cudaStream_t s, long base = -1, long cnt = -1) {
        if (base < 0) { base = 0; cnt = (long)p.n; }
        constexpr int threads = 256;
        const int blocks = (int)((cnt + threads - 1) / threads);
        k_workledger<<<blocks, threads, 0, s>>>(p.views(), f.views(), rp,
                                                wl.data(), nreg, nwb, vmax,
                                                wdt, base, cnt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // dump the accumulated ledger and reset it
    void wl_write_reset(cudaStream_t s, const char* fn) {
        write_dev(wl, s, fn);
        wl.zero(s);
    }

private:
    static void write_dev(DeviceArray<double>& a, cudaStream_t s,
                          const char* fn) {
        std::vector<double> h(a.size());
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(h.data(), a.data(), h.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::FILE* fo = std::fopen(fn, "wb");
        if (!fo) throw std::runtime_error(std::string("gapdiag: cannot open ") + fn);
        std::fwrite(h.data(), sizeof(double), h.size(), fo);
        std::fclose(fo);
    }
};

// A0 fvline (PLAN_TWO_TRACK v2.1 §A0.2): equatorial-window velocity
// diagnostics for the Track-A mechanism chain P_L1,hot -> f(v_par) plateau
// -> P_C0 flip -> chirp stop. Window = |i - i_eq| < hw CELLS around the
// equator column (NOT physical x < hw — plan hard gate). One kernel serves
// both products: h1 = fine w-weighted f(v_par) [nv], h2 = coarse
// (v_par, v_perp) [npar][nperp]; pass nullptr for the one not wanted.
__global__ void k_fvwin(ParticleViews p, RunParams rp, float dxp, float dyp,
                        int ieq, int hw, float vmax,
                        double* h1, int nv, double* h2, int npar, int nperp) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= p.n) return;
    const float x0 = p.x[t];
    const int i = (int)x0;
    if (i < ieq - hw + 1 || i > ieq + hw - 1) return;   // |i - ieq| < hw
    float bx, by, bz;
    bhat(rp, x0 * dxp, p.y[t] * dyp, bx, by, bz);
    const float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float gi = rp.rel ? rsqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
    const float vx = ux * gi, vy = uy * gi, vz = uz * gi;
    const float vpar  = vx * bx + vy * by + vz * bz;
    const double w = (double)p.w[t];
    if (h1) {
        const int ip = (int)((vpar + vmax) / (2.f * vmax) * (float)nv);
        if (ip >= 0 && ip < nv) atomicAdd(&h1[ip], w);
    }
    if (h2) {
        const float vperp = sqrtf(fmaxf(vx * vx + vy * vy + vz * vz
                                        - vpar * vpar, 0.f));
        const int ip = (int)((vpar + vmax) / (2.f * vmax) * (float)npar);
        const int jp = (int)(vperp / vmax * (float)nperp);
        if (ip >= 0 && ip < npar && jp < nperp)
            atomicAdd(&h2[(long)ip * nperp + jp], w);
    }
}

struct EqFvDiag {
    int   ieq, hw, nv, npar, nperp;
    float vmax;
    DeviceArray<double> h1;   // [nv]           fine f(v_par)
    DeviceArray<double> h2;   // [npar][nperp]  coarse (v_par, v_perp)

    EqFvDiag(int ieq_, int hw_, int nv_, int npar_, int nperp_, float vmax_)
        : ieq(ieq_), hw(hw_), nv(nv_), npar(npar_), nperp(nperp_), vmax(vmax_),
          h1((std::size_t)nv_), h2((std::size_t)npar_ * nperp_) {
        h1.zero(nullptr);
        h2.zero(nullptr);
    }

    void line_snapshot(Particles& p, const RunParams& rp, const Grid& g,
                       cudaStream_t s, const char* fn) {
        h1.zero(s);
        launch(p, rp, g, s, h1.data(), nullptr);
        write_dev(h1, s, fn);
    }
    void f2d_snapshot(Particles& p, const RunParams& rp, const Grid& g,
                      cudaStream_t s, const char* fn) {
        h2.zero(s);
        launch(p, rp, g, s, nullptr, h2.data());
        write_dev(h2, s, fn);
    }

private:
    void launch(Particles& p, const RunParams& rp, const Grid& g,
                cudaStream_t s, double* d1, double* d2) {
        constexpr int threads = 256;
        const int blocks = (int)((p.n + threads - 1) / threads);
        k_fvwin<<<blocks, threads, 0, s>>>(p.views(), rp, (float)g.dx,
                                           (float)g.dy, ieq, hw, vmax,
                                           d1, nv, d2, npar, nperp);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    static void write_dev(DeviceArray<double>& a, cudaStream_t s,
                          const char* fn) {
        std::vector<double> h(a.size());
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(h.data(), a.data(), h.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::FILE* fo = std::fopen(fn, "wb");
        if (!fo) throw std::runtime_error(std::string("gapdiag: cannot open ") + fn);
        std::fwrite(h.data(), sizeof(double), h.size(), fo);
        std::fclose(fo);
    }
};

} // namespace gapdiag
} // namespace arc
