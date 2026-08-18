// pic2d — P1 field engine: 2D Yee Maxwell in the (x,z) meridional plane
// (∂/∂y = 0, out-of-plane = y) + linearized T=0 cold electron fluid +
// Umeda masked damping frame. (PLAN_2D_REBORN §2.2–§2.4, phase P1.)
//
// Axis contract vs the legacy yee2d.hpp: legacy works in (x,y) with ∂z = 0;
// pic2d relabels (legacy y → z in-plane, legacy z → y out-of-plane) so that
// the linedipole background (B0 in the (x,z) plane, ∥ ẑ at the equator)
// composes with the same Maxwell scheme. Staggering (node (i,k) at
// (x0+i·dx, z0+k·dz)):
//     Ey(i,k)   Ex(i+½,k)   Ez(i,k+½)
//     By(i+½,k+½)   Bx(i,k+½)   Bz(i+½,k)
// Update (leapfrog, B at half steps, c² = 1/(μ0ε0), ε0 = 1):
//     ∂Bx/∂t = +∂Ey/∂z             ∂Ex/∂t = −c²∂By/∂z − Jx
//     ∂Bz/∂t = −∂Ey/∂x             ∂Ez/∂t = +c²∂By/∂x − Jz
//     ∂By/∂t = −∂Ex/∂z + ∂Ez/∂x    ∂Ey/∂t = c²(∂Bx/∂z − ∂Bz/∂x) − Jy
//
// Cold fluid (cold_model=full lineage, validated 0.003–0.46% vs Stix on the
// legacy stack): 3-component Vc at NODES, half-kick E / exact rotation about
// the LOCAL analytic b̂(x,z) of Background2D / half-kick E; symmetric
// staggered E-gather and Jc-scatter (real transfer cos(kd/2), zero phase
// error). Linearized: rotation about B0 only — wave-B×Vc and advection are
// the [cold] nonlinear=1 switch, NOT implemented here (P4 cascade arm).
//
// Kernels take views by value + an explicit index Range so a z-slab domain
// decomposition stays additive (PLAN_2D_REBORN §4): single GPU = one slab
// covering the full grid, periodic wrap in idx(); the production dipole box
// runs periodic + edge masks (fields damped to ~0 before every wrap).

#ifndef ARC_PIC2D_FIELDS2D_HPP
#define ARC_PIC2D_FIELDS2D_HPP

#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic2d/background2d.hpp"

#include <cmath>
#include <vector>

namespace arc2d {

struct Range { int i0, i1, k0, k1; };   // half-open [i0,i1)×[k0,k1)

struct FieldViews2D {
    float *ex, *ey, *ez, *bx, *by, *bz, *jx, *jy, *jz;
    float *vcx, *vcy, *vcz;             // cold fluid velocity at nodes
    float *maske, *maskb;               // Umeda damping profiles (nullptr = off)
    float *jxr, *jyr, *jzr;             // replicated J (contention relief);
    int nrep;                           //   nrep = 1 ⇒ aliases of jx,jy,jz
    int nx, nz;
    float dx, dz, dt, c2;
    float qm_c, nc;                     // cold q/m (−1) and density

    __host__ __device__ int idx(int i, int k) const {
        i += (i < 0) * nx - (i >= nx) * nx;   // periodic wrap (|off| ≤ nx)
        k += (k < 0) * nz - (k >= nz) * nz;
        return k * nx + i;
    }
};

namespace f2d {

constexpr int TX = 16, TZ = 16;

inline dim3 blocks_for(const Range& r) {
    return dim3((unsigned(r.i1 - r.i0) + TX - 1) / TX,
                (unsigned(r.k1 - r.k0) + TZ - 1) / TZ);
}

#ifdef __CUDACC__

__device__ inline bool in_range(const Range& r, int& i, int& k) {
    i = r.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    k = r.k0 + blockIdx.y * blockDim.y + threadIdx.y;
    return i < r.i1 && k < r.k1;
}

// ---- Maxwell ---------------------------------------------------------------

static __global__ void k_faraday(FieldViews2D v, Range r, float dt2) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const float ey_zp = v.ey[v.idx(i, k + 1)], ey_xp = v.ey[v.idx(i + 1, k)];
    const float ex_zp = v.ex[v.idx(i, k + 1)], ez_xp = v.ez[v.idx(i + 1, k)];
    v.bx[c] += dt2 * (ey_zp - v.ey[c]) / v.dz;
    v.bz[c] -= dt2 * (ey_xp - v.ey[c]) / v.dx;
    v.by[c] += dt2 * (-(ex_zp - v.ex[c]) / v.dz + (ez_xp - v.ez[c]) / v.dx);
}

static __global__ void k_ampere(FieldViews2D v, Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const float by_zm = v.by[v.idx(i, k - 1)], by_xm = v.by[v.idx(i - 1, k)];
    const float bx_zm = v.bx[v.idx(i, k - 1)], bz_xm = v.bz[v.idx(i - 1, k)];
    v.ex[c] += v.dt * (-v.c2 * (v.by[c] - by_zm) / v.dz - v.jx[c]);
    v.ez[c] += v.dt * (+v.c2 * (v.by[c] - by_xm) / v.dx - v.jz[c]);
    v.ey[c] += v.dt * (v.c2 * ((v.bx[c] - bx_zm) / v.dz - (v.bz[c] - bz_xm) / v.dx)
                       - v.jy[c]);
}

// ---- cold fluid ------------------------------------------------------------
// Vc(t−½) → Vc(t+½) with E(t): half-kick, exact Rodrigues rotation about the
// LOCAL analytic b̂ (angle from |B0|(x,z)), half-kick; deposit Jc(t+½) with
// the symmetric staggered scatter. One kernel = one node.

static __global__ void k_cold_step(FieldViews2D v, Range r, Background2D bg,
                                   float x0, float z0) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    // symmetric staggered E gather to the node
    const float Ex = 0.5f * (v.ex[c] + v.ex[v.idx(i - 1, k)]);
    const float Ez = 0.5f * (v.ez[c] + v.ez[v.idx(i, k - 1)]);
    const float Ey = v.ey[c];
    const float hk = 0.5f * v.qm_c * v.dt;
    float ux = v.vcx[c] + hk * Ex;
    float uy = v.vcy[c] + hk * Ey;
    float uz = v.vcz[c] + hk * Ez;
    // rotation: dv/dt = qm v×B0 ⇒ angular velocity ω⃗ = −qm B0⃗ (electrons
    // qm = −1 ⇒ right-handed about +b̂), angle φ = |qm| B dt exact
    const float xp = x0 + i * v.dx, zp = z0 + k * v.dz;
    const Vec2<float> b = b0_field<float>(bg, xp, zp);
    const float B = sqrtf(b.x * b.x + b.z * b.z);
    const float s = (v.qm_c < 0.f) ? 1.f : -1.f;      // n̂ = s·b̂
    const float nx_ = s * b.x / B, nz_ = s * b.z / B; // n̂_y = 0 (B0 in-plane)
    const float phi = fabsf(v.qm_c) * B * v.dt;
    const float cf = cosf(phi), sf = sinf(phi);
    // Rodrigues, axis (nx,0,nz): n̂×u = (−nz·uy, nz·ux − nx·uz, nx·uy)
    const float ndot = nx_ * ux + nz_ * uz;
    const float rx = ux * cf + (-nz_ * uy) * sf + nx_ * ndot * (1.f - cf);
    const float ry = uy * cf + (nz_ * ux - nx_ * uz) * sf;
    const float rz = uz * cf + (nx_ * uy) * sf + nz_ * ndot * (1.f - cf);
    ux = rx + hk * Ex;
    uy = ry + hk * Ey;
    uz = rz + hk * Ez;
    v.vcx[c] = ux; v.vcy[c] = uy; v.vcz[c] = uz;
}

// Jc scatter (separate pass: neighbours' Vc must be final). Jc = qe nc Vc,
// qe = −1 ⇒ Jc = −nc Vc; symmetric average onto the staggered J sites.
static __global__ void k_cold_current(FieldViews2D v, Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const float q = -v.nc;
    v.jx[c] += q * 0.5f * (v.vcx[c] + v.vcx[v.idx(i + 1, k)]);
    v.jz[c] += q * 0.5f * (v.vcz[c] + v.vcz[v.idx(i, k + 1)]);
    v.jy[c] += q * v.vcy[c];
}

// ---- Umeda masked damping (edge frame absorbers) ---------------------------

static __global__ void k_mask_e(FieldViews2D v, Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const float m = v.maske[c];
    v.ex[c] *= m; v.ey[c] *= m; v.ez[c] *= m;
    v.vcx[c] *= m; v.vcy[c] *= m; v.vcz[c] *= m;
}
static __global__ void k_mask_b(FieldViews2D v, Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const float m = v.maskb[c];
    v.bx[c] *= m; v.by[c] *= m; v.bz[c] *= m;
}

// ---- replicated-J reduction ------------------------------------------------
// High-ppc/small-grid runs (Li-class: 2000 markers/cell on 4096 cells)
// serialize on global atomics; depositing into nrep block-strided replicas
// and reducing removes the contention. nrep = 1 is the exact legacy path.

static __global__ void k_reduce_j(FieldViews2D v, Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const size_t n = size_t(v.nx) * v.nz;
    float sx = 0, sy = 0, sz = 0;
    for (int rep = 0; rep < v.nrep; ++rep) {
        sx += v.jxr[rep * n + c];
        sy += v.jyr[rep * n + c];
        sz += v.jzr[rep * n + c];
    }
    v.jx[c] = sx; v.jy[c] = sy; v.jz[c] = sz;
}

// ---- binomial current filter (legacy jfilter, production recipe) ----------
// 1-2-1 smoothing per axis; kills the k ~ Nyquist shot-noise currents that
// otherwise pump grid-scale modes near the whistler resonance cone (v_g → 0
// there: the energy cannot propagate to the masks and accumulates — the
// P2 V1 first-run finding: W_EM grew secularly without it).

static __global__ void k_filter_x(const float* src, float* dst, FieldViews2D v,
                                  Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    dst[v.idx(i, k)] = 0.25f * src[v.idx(i - 1, k)] + 0.5f * src[v.idx(i, k)] +
                       0.25f * src[v.idx(i + 1, k)];
}
static __global__ void k_filter_z(const float* src, float* dst, FieldViews2D v,
                                  Range r) {
    int i, k;
    if (!in_range(r, i, k)) return;
    dst[v.idx(i, k)] = 0.25f * src[v.idx(i, k - 1)] + 0.5f * src[v.idx(i, k)] +
                       0.25f * src[v.idx(i, k + 1)];
}

// fused binomial^3: three 1-2-1 passes per axis == one separable 7-tap pass
// ((1,6,15,20,15,6,1)/64). 18 kernels/~48 GB per step -> 3 kernels/~8 GB;
// the ppc=1 probe showed the field pipeline was HALF the V4R step cost.
static __global__ void k_filter7(const float* src, float* dst, FieldViews2D v,
                                 Range r) {
    __shared__ float tile[22][22];              // 16×16 block + halo 3
    const int i0 = blockIdx.x * 16, k0 = blockIdx.y * 16;
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < 22 * 22;
         t += blockDim.x * blockDim.y) {
        const int li = t % 22, lk = t / 22;
        tile[lk][li] = src[v.idx(i0 + li - 3, k0 + lk - 3)];
    }
    __syncthreads();
    const int i = i0 + threadIdx.x, k = k0 + threadIdx.y;
    if (i >= r.i1 || k >= r.k1) return;
    const float c7[7] = {1.f / 64, 6.f / 64, 15.f / 64, 20.f / 64,
                         15.f / 64, 6.f / 64, 1.f / 64};
    float col[7];
    for (int dz = 0; dz < 7; ++dz) {
        float acc = 0.f;
        for (int dx2 = 0; dx2 < 7; ++dx2)
            acc += c7[dx2] * tile[threadIdx.y + dz][threadIdx.x + dx2];
        col[dz] = acc;
    }
    float out = 0.f;
    for (int dz = 0; dz < 7; ++dz) out += c7[dz] * col[dz];
    dst[v.idx(i, k)] = out;
}

// ---- energy ledger (double accumulation) -----------------------------------
// WEM = ½Σ(E² + c²B²)dV (ε0 = 1, 1/μ0 = c²); Wc = ½ nc Σ|Vc|² dV.

static __global__ void k_energy(FieldViews2D v, Range r, double* acc) {
    int i, k;
    if (!in_range(r, i, k)) return;
    const int c = v.idx(i, k);
    const double e2 = double(v.ex[c]) * v.ex[c] + double(v.ey[c]) * v.ey[c] +
                      double(v.ez[c]) * v.ez[c];
    const double b2 = double(v.bx[c]) * v.bx[c] + double(v.by[c]) * v.by[c] +
                      double(v.bz[c]) * v.bz[c];
    const double c2v = double(v.vcx[c]) * v.vcx[c] + double(v.vcy[c]) * v.vcy[c] +
                       double(v.vcz[c]) * v.vcz[c];
    const double dV = double(v.dx) * v.dz;
    atomicAdd(&acc[0], 0.5 * (e2 + double(v.c2) * b2) * dV);
    atomicAdd(&acc[1], 0.5 * double(v.nc) * c2v * dV);
}

#endif  // __CUDACC__

}  // namespace f2d

// ---- owner ----------------------------------------------------------------

struct Fields2D {
    arc::DeviceArray<float> ex, ey, ez, bx, by, bz, jx, jy, jz, vcx, vcy, vcz;
    arc::DeviceArray<float> maske, maskb, jtmp, jxr, jyr, jzr;
    arc::DeviceArray<double> energy_acc;
    int nx = 0, nz = 0;
    int jfilter = 3;                       // binomial passes per axis
    int nrep = 1;                          // J replicas (allocate_replicas)
    double dx = 0, dz = 0, dt = 0, cspeed = 1.0, nc = 1.0, x0 = 0, z0 = 0;
    Background2D bg;
    bool masks_on = false;

    void allocate(int nx_, int nz_) {
        nx = nx_; nz = nz_;
        const size_t n = size_t(nx) * nz;
        for (auto* a : {&ex, &ey, &ez, &bx, &by, &bz, &jx, &jy, &jz,
                        &vcx, &vcy, &vcz, &jtmp})
            *a = arc::DeviceArray<float>(n);
        energy_acc = arc::DeviceArray<double>(2);
        for (auto* a : {&ex, &ey, &ez, &bx, &by, &bz, &jx, &jy, &jz,
                        &vcx, &vcy, &vcz, &jtmp})
            a->zero();
    }

    // Deposit contention relief for high-ppc/small-grid runs; memory cost
    // nrep × 3 × cells × 4 B (deck-sized: Li-class 4096 cells × 16 = 0.8 MB;
    // big boxes keep nrep = 1).
    void allocate_replicas(int n) {
        nrep = n;
        if (nrep > 1) {
            const size_t sz = size_t(nrep) * nx * nz;
            jxr = arc::DeviceArray<float>(sz);
            jyr = arc::DeviceArray<float>(sz);
            jzr = arc::DeviceArray<float>(sz);
        }
    }

    FieldViews2D views() {
        return { ex.data(), ey.data(), ez.data(), bx.data(), by.data(), bz.data(),
                 jx.data(), jy.data(), jz.data(), vcx.data(), vcy.data(), vcz.data(),
                 masks_on ? maske.data() : nullptr,
                 masks_on ? maskb.data() : nullptr,
                 nrep > 1 ? jxr.data() : jx.data(),
                 nrep > 1 ? jyr.data() : jy.data(),
                 nrep > 1 ? jzr.data() : jz.data(), nrep,
                 nx, nz, float(dx), float(dz), float(dt),
                 float(cspeed * cspeed), -1.f, float(nc) };
    }

    Range full() const { return {0, nx, 0, nz}; }

    // Umeda-style edge frame: cell-centred damping profile m(d) rising
    // smoothly from (1 − nu_max) at the boundary to 1 over nd cells; applied
    // multiplicatively to E and B every step inside the frame.
    void build_masks(int nd, double nu_max) {
        std::vector<float> h(size_t(nx) * nz);
        auto prof = [&](int d) {
            if (d >= nd) return 1.0;
            const double s = std::sin(0.5 * M_PI * double(d) / nd);
            return 1.0 - nu_max * (1.0 - s * s);
        };
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i < nx; ++i) {
                const int d = std::min(std::min(i, nx - 1 - i),
                                       std::min(k, nz - 1 - k));
                h[size_t(k) * nx + i] = float(prof(d));
            }
        maske = arc::DeviceArray<float>(h.size());
        maskb = arc::DeviceArray<float>(h.size());
        CUDA_CHECK(cudaMemcpy(maske.data(), h.data(), h.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(maskb.data(), h.data(), h.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        masks_on = true;
    }

#ifdef __CUDACC__
    // zero the particle-deposit target (replicas when nrep > 1)
    void zero_j(cudaStream_t s = nullptr) {
        if (nrep > 1) { jxr.zero(s); jyr.zero(s); jzr.zero(s); }
        else          { jx.zero(s);  jy.zero(s);  jz.zero(s); }
    }
    // sum replicas into jx,jy,jz (no-op at nrep = 1)
    void reduce_j(cudaStream_t s = nullptr) {
        if (nrep <= 1) return;
        FieldViews2D v = views();
        const Range r = full();
        f2d::k_reduce_j<<<f2d::blocks_for(r), dim3(f2d::TX, f2d::TZ), 0, s>>>(v, r);
    }

    // binomial-filter the deposited PARTICLE currents (call after the hot
    // deposit, before the cold pass adds its smooth analytic currents)
    void filter_j(cudaStream_t s = nullptr) {
        FieldViews2D v = views();
        const Range r = full();
        const dim3 nb = f2d::blocks_for(r), tb(f2d::TX, f2d::TZ);
        if (jfilter == 3) {   // fused 7-tap; copy-back keeps captured views valid
            for (auto* a : {&jx, &jy, &jz}) {
                f2d::k_filter7<<<nb, tb, 0, s>>>(a->data(), jtmp.data(), v, r);
                CUDA_CHECK(cudaMemcpyAsync(a->data(), jtmp.data(),
                                           size_t(nx) * nz * 4,
                                           cudaMemcpyDeviceToDevice, s));
            }
            return;
        }
        for (int pass = 0; pass < jfilter; ++pass)
            for (auto* a : {&jx, &jy, &jz}) {
                f2d::k_filter_x<<<nb, tb, 0, s>>>(a->data(), jtmp.data(), v, r);
                f2d::k_filter_z<<<nb, tb, 0, s>>>(jtmp.data(), a->data(), v, r);
            }
    }

    // One field+fluid step (no kinetic species yet — P2 inserts the hot
    // deposit between zero-J and the cold pass, same worldline contract as
    // the legacy step_at).
    void step(cudaStream_t s = nullptr) {
        FieldViews2D v = views();
        const Range r = full();
        const dim3 nb = f2d::blocks_for(r), tb(f2d::TX, f2d::TZ);
        f2d::k_faraday<<<nb, tb, 0, s>>>(v, r, float(dt / 2));
        jx.zero(s); jy.zero(s); jz.zero(s);
        f2d::k_cold_step<<<nb, tb, 0, s>>>(v, r, bg, float(x0), float(z0));
        f2d::k_cold_current<<<nb, tb, 0, s>>>(v, r);
        f2d::k_faraday<<<nb, tb, 0, s>>>(v, r, float(dt / 2));
        f2d::k_ampere<<<nb, tb, 0, s>>>(v, r);
        if (masks_on) {
            f2d::k_mask_e<<<nb, tb, 0, s>>>(v, r);
            f2d::k_mask_b<<<nb, tb, 0, s>>>(v, r);
        }
    }

    // returns {W_EM, W_cold}
    void energies(double out[2], cudaStream_t s = nullptr) {
        energy_acc.zero(s);
        FieldViews2D v = views();
        const Range r = full();
        f2d::k_energy<<<f2d::blocks_for(r), dim3(f2d::TX, f2d::TZ), 0, s>>>(
            v, r, energy_acc.data());
        CUDA_CHECK(cudaMemcpy(out, energy_acc.data(), 2 * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
#endif
};

}  // namespace arc2d

#endif  // ARC_PIC2D_FIELDS2D_HPP
