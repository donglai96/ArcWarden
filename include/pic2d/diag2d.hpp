// pic2d — P3b diagnostics pack (PLAN_2D_REBORN §2.5: the ledger IS the
// experiment). Three instruments, all species-tagged, all cadenced:
//
// 1. LINE PRODUCTS: fields sampled along the CENTRAL FIELD LINE (the L₀
//    circle), in the FIELD-ALIGNED frame — E∥ = E·b̂, E1 = E·ê1 (ŷ×b̂),
//    Ey, and the wave-B triplet likewise. This is the chorus spectrogram
//    source (legacy bline equivalent, but following the curved line), plus
//    high-cadence probes at fixed latitudes {0, ±5, ±10, ±20}° for STFT.
//    Dual-normalization contract: meta records Ω_e(local) per probe so the
//    analysis can report ω/Ω_e(local) AND ω/Ω_e,eq (the fixed-band aliasing
//    lesson, now structural).
// 2. J·E WORK LEDGER, per species × λ-region × v∥-bin × channel
//    (Landau: q w wd E∥ v∥ | cyclotron: q w wd E⊥·v⊥), accumulated inside
//    the push kernel from the already-gathered fields (zero-cost when off,
//    null-pointer branch — bit-identical trajectories). The H2 causal
//    chain (E∥ work → Δf → plateau → 0.5 drop) is this array plotted.
// 3. f(v∥, v⊥ | λ-region) histograms per species, BOTH Σw (marker) and
//    Σw·wd (δf) weighted — the plateau/scar monitor.
//
// λ-regions (fixed v1): signed bands {−35,−20,−10,−5,0,5,10,20,35}° → 8
// bins (hemisphere asymmetry is a science observable — one-way-street).

#ifndef ARC_PIC2D_DIAG2D_HPP
#define ARC_PIC2D_DIAG2D_HPP

#include "pic2d/fields2d.hpp"
#include "pic2d/kinetic2d.hpp"
#include "pic2d/particles2d.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace arc2d {

constexpr int DIAG_NREG = 8;

// POD passed to kernels; acc layout [reg][vbin][2]; null acc = ledger off
struct LedgerViews {
    double* acc = nullptr;
    int nvb = 64;
    float vmax = 0.35f;
};

namespace d2d {

__host__ __device__ inline int lam_region(float lam_deg) {
    const float e[DIAG_NREG + 1] = {-35, -20, -10, -5, 0, 5, 10, 20, 35};
    if (lam_deg < e[0]) return 0;
    for (int i = 0; i < DIAG_NREG; ++i)
        if (lam_deg < e[i + 1]) return i;
    return DIAG_NREG - 1;
}

#ifdef __CUDACC__

// fields at precomputed line points, rotated to the field-aligned frame:
// out[6][NS] = {E∥, E1, Ey, B∥w, B1, By}
static __global__ void k_sample_line(FieldViews2D v, const float* px,
                                     const float* pz, const float* pbx,
                                     const float* pbz, int ns, float x0,
                                     float z0, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const float gx = (px[i] - x0) / v.dx, gz = (pz[i] - z0) / v.dz;
    auto g = [&](const float* f, float ox, float oz) {
        const float ax = gx - ox, az = gz - oz;
        const int i0 = int(floorf(ax)), k0 = int(floorf(az));
        const float fx = ax - i0, fz = az - k0;
        return (1.f - fx) * (1.f - fz) * f[v.idx(i0, k0)] +
               fx * (1.f - fz) * f[v.idx(i0 + 1, k0)] +
               (1.f - fx) * fz * f[v.idx(i0, k0 + 1)] +
               fx * fz * f[v.idx(i0 + 1, k0 + 1)];
    };
    const float Ex = g(v.ex, 0.5f, 0.f), Ey = g(v.ey, 0.f, 0.f),
                Ez = g(v.ez, 0.f, 0.5f);
    const float Bx = g(v.bx, 0.f, 0.5f), By = g(v.by, 0.5f, 0.5f),
                Bz = g(v.bz, 0.5f, 0.f);
    const float bx = pbx[i], bz = pbz[i];        // b̂ (in-plane), ê1 = (bz,−bx)
    out[0 * ns + i] = Ex * bx + Ez * bz;
    out[1 * ns + i] = Ex * bz - Ez * bx;
    out[2 * ns + i] = Ey;
    out[3 * ns + i] = Bx * bx + Bz * bz;
    out[4 * ns + i] = Bx * bz - Bz * bx;
    out[5 * ns + i] = By;
}

// J·E ledger: STANDALONE kernel (the validated push kernel stays untouched
// — zero regression risk); called every ledger_every steps with
// dt_eff = dt·ledger_every (Riemann sampling of the work integral).
// Shared-memory per-block partials kill the small-array atomic contention.
// acc layout [reg][vbin][2]: ch0 = Landau (q w wd E∥ v∥), ch1 = cyclotron
// (q w wd E⊥·v⊥); wave fields only.
static __global__ void k_ledger(MarkerViews p, KineticCfg c, FieldViews2D v,
                                Background2D bg, float x0, float z0,
                                LedgerViews lv, float dt_eff, uint64_t n) {
    __shared__ double part[DIAG_NREG * 64 * 2];
    for (int i = threadIdx.x; i < DIAG_NREG * lv.nvb * 2; i += blockDim.x)
        part[i] = 0.0;
    __syncthreads();
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m < n) {
        const float x = p.x[m], z = p.z[m];
        const float gx = (x - x0) / v.dx, gz = (z - z0) / v.dz;
        auto g = [&](const float* f, float ox, float oz) {
            const float ax = gx - ox, az = gz - oz;
            const int i0 = int(floorf(ax)), k0 = int(floorf(az));
            const float fx = ax - i0, fz = az - k0;
            return (1.f - fx) * (1.f - fz) * f[v.idx(i0, k0)] +
                   fx * (1.f - fz) * f[v.idx(i0 + 1, k0)] +
                   (1.f - fx) * fz * f[v.idx(i0, k0 + 1)] +
                   fx * fz * f[v.idx(i0 + 1, k0 + 1)];
        };
        const float Ex = g(v.ex, 0.5f, 0.f), Ey = g(v.ey, 0.f, 0.f),
                    Ez = g(v.ez, 0.f, 0.5f);
        const Vec2<float> bh = b0_bhat<float>(bg, x, z);
        const float gam = c.rel
            ? sqrtf(1.f + p.ux[m]*p.ux[m] + p.uy[m]*p.uy[m] + p.uz[m]*p.uz[m])
            : 1.f;
        const float vx = p.ux[m] / gam, vy = p.uy[m] / gam, vz = p.uz[m] / gam;
        const float vpar = vx * bh.x + vz * bh.z;
        const float Epar = Ex * bh.x + Ez * bh.z;
        const float vdotE = vx * Ex + vy * Ey + vz * Ez;
        const float qwwd = c.qm * p.w[m] * p.wd[m];   // signed charge · weight
        const float wl = qwwd * Epar * vpar * dt_eff;
        const float wc = qwwd * (vdotE - Epar * vpar) * dt_eff;
        const int reg = lam_region(atan2f(z, x) * 57.29578f);
        int iv = int((vpar + lv.vmax) / (2 * lv.vmax) * lv.nvb);
        iv = max(0, min(lv.nvb - 1, iv));
        atomicAdd(&part[(reg * lv.nvb + iv) * 2 + 0], double(wl));
        atomicAdd(&part[(reg * lv.nvb + iv) * 2 + 1], double(wc));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < DIAG_NREG * lv.nvb * 2; i += blockDim.x)
        if (part[i] != 0.0) atomicAdd(&lv.acc[i], part[i]);
}

// f(v∥,v⊥ | region) histograms: hist layout [reg][npar][nperp][2] (Σw, Σw·wd)
static __global__ void k_fv_hist(MarkerViews p, Background2D bg, double* hist,
                                 int npar, int nperp, float upmax, float uqmax,
                                 uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const float x = p.x[m], z = p.z[m];
    const Vec2<float> bh = b0_bhat<float>(bg, x, z);
    const float up = p.ux[m] * bh.x + p.uz[m] * bh.z;
    const float u1 = p.ux[m] * bh.z - p.uz[m] * bh.x;
    const float uq = sqrtf(u1 * u1 + p.uy[m] * p.uy[m]);
    const int reg = lam_region(atan2f(z, x) * 57.29578f);
    const int ip = int((up + upmax) / (2 * upmax) * npar);
    const int iq = int(uq / uqmax * nperp);
    if (ip < 0 || ip >= npar || iq >= nperp) return;
    const size_t c = ((size_t(reg) * npar + ip) * nperp + iq) * 2;
    atomicAdd(&hist[c], double(p.w[m]));
    atomicAdd(&hist[c + 1], double(p.w[m]) * p.wd[m]);
}

#endif  // __CUDACC__

}  // namespace d2d

// host-side owner: line geometry, per-species ledgers/histograms, dump I/O
struct Diag2D {
    // line sampling along the L0 circle
    arc::DeviceArray<float> px, pz, pbx, pbz, line_out;
    std::vector<float> lam_line;           // latitude of each sample (deg)
    int ns = 0;
    std::vector<int> probe_idx;            // indices of the probe stations
    std::vector<float> probe_wce;          // local Ω_e per station
    // per-species products
    struct SpDiag {
        arc::DeviceArray<double> ledger;   // [DIAG_NREG][nvb][2]
        arc::DeviceArray<double> fv;       // [DIAG_NREG][npar][nperp][2]
    };
    std::vector<SpDiag> sp;
    int nvb = 64, npar = 96, nperp = 48;
    float vmax = 0.35f, uqmax = 0.5f;

    void build_line(const Background2D& bg, double L0, double lam_w_deg,
                    double ds, int nspecies) {
        std::vector<float> hx, hz, hbx, hbz;
        const double lw = lam_w_deg * M_PI / 180.0;
        const double dlam = ds / L0;
        for (double lam = -lw; lam <= lw; lam += dlam) {
            double x, z;
            line_point_of(bg, L0, lam, x, z);
            hx.push_back(float(x));
            hz.push_back(float(z));
            const Vec2<double> b = b0_bhat<double>(bg, x, z);
            hbx.push_back(float(b.x));
            hbz.push_back(float(b.z));
            lam_line.push_back(float(lam * 180 / M_PI));
        }
        ns = int(hx.size());
        auto up = [&](arc::DeviceArray<float>& a, std::vector<float>& h) {
            a = arc::DeviceArray<float>(h.size());
            CUDA_CHECK(cudaMemcpy(a.data(), h.data(), h.size() * 4,
                                  cudaMemcpyHostToDevice));
        };
        up(px, hx); up(pz, hz); up(pbx, hbx); up(pbz, hbz);
        line_out = arc::DeviceArray<float>(size_t(6) * ns);
        for (float lam : {0.f, 5.f, -5.f, 10.f, -10.f, 20.f, -20.f}) {
            int best = 0;
            for (int i = 1; i < ns; ++i)
                if (std::fabs(lam_line[i] - lam) < std::fabs(lam_line[best] - lam))
                    best = i;
            probe_idx.push_back(best);
            probe_wce.push_back(float(b0_abs<double>(bg, hx[best], hz[best])));
        }
        sp.resize(nspecies);
        for (auto& s : sp) {
            s.ledger = arc::DeviceArray<double>(size_t(DIAG_NREG) * nvb * 2);
            s.fv = arc::DeviceArray<double>(size_t(DIAG_NREG) * npar * nperp * 2);
            s.ledger.zero();
            s.fv.zero();
        }
    }

    LedgerViews ledger_views(int is) {
        return { sp[is].ledger.data(), nvb, vmax };
    }

#ifdef __CUDACC__
    void sample_line(FieldViews2D v, float x0, float z0) {
        d2d::k_sample_line<<<(ns + 127) / 128, 128>>>(
            v, px.data(), pz.data(), pbx.data(), pbz.data(), ns, x0, z0,
            line_out.data());
    }
    void fv_accumulate(int is, MarkerViews p, const Background2D& bg, uint64_t n) {
        d2d::k_fv_hist<<<int((n + 255) / 256), 256>>>(
            p, bg, sp[is].fv.data(), npar, nperp, vmax, uqmax, n);
    }
    void ledger_accumulate(int is, MarkerViews p, const KineticCfg& c,
                           FieldViews2D v, const Background2D& bg, float x0,
                           float z0, float dt_eff, uint64_t n) {
        d2d::k_ledger<<<int((n + 255) / 256), 256>>>(
            p, c, v, bg, x0, z0, ledger_views(is), dt_eff, n);
    }
#endif

    // dump helpers (raw doubles/floats; python quicklooks decode)
    void dump_line(const std::string& path) {
        std::vector<float> h(size_t(6) * ns);
        CUDA_CHECK(cudaMemcpy(h.data(), line_out.data(), h.size() * 4,
                              cudaMemcpyDeviceToHost));
        FILE* f = std::fopen(path.c_str(), "wb");
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }
    void dump_and_reset(int is, const std::string& led_path,
                        const std::string& fv_path) {
        std::vector<double> h(sp[is].ledger.size());
        CUDA_CHECK(cudaMemcpy(h.data(), sp[is].ledger.data(), h.size() * 8,
                              cudaMemcpyDeviceToHost));
        FILE* f = std::fopen(led_path.c_str(), "wb");
        std::fwrite(h.data(), 8, h.size(), f);
        std::fclose(f);
        sp[is].ledger.zero();
        std::vector<double> g(sp[is].fv.size());
        CUDA_CHECK(cudaMemcpy(g.data(), sp[is].fv.data(), g.size() * 8,
                              cudaMemcpyDeviceToHost));
        f = std::fopen(fv_path.c_str(), "wb");
        std::fwrite(g.data(), 8, g.size(), f);
        std::fclose(f);
        sp[is].fv.zero();
    }
};

}  // namespace arc2d

#endif  // ARC_PIC2D_DIAG2D_HPP
