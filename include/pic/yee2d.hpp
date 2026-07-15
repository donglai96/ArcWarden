// ArcWarden — L-shell plan M1: 2D3V Cartesian full-Maxwell (Yee) field engine
// + Esirkepov charge-conserving current deposit.
//
// The third FieldModel branch (plan §11.1): SpectralES / SpectralDarwin keep
// the existing Simulation<Cfg>; YeeMaxwell runs the new MaxwellSimulation
// (simulation_maxwell.hpp). Particles / Grid / RunParams are shared.
//
// Staggering (∂z = 0, all 6 components, periodic; guard cells arrive with the
// M2 boundary work — v1 uses modular indexing):
//     Ex(i+½, j)   Ey(i, j+½)   Ez(i, j)
//     Bx(i, j+½)   By(i+½, j)   Bz(i+½, j+½)
// Update (leapfrog, B at half steps):
//     ∂Bx/∂t = -∂Ez/∂y            ∂Ex/∂t = +c² ∂Bz/∂y - Jx/ε0
//     ∂By/∂t = +∂Ez/∂x            ∂Ey/∂t = -c² ∂Bz/∂x - Jy/ε0
//     ∂Bz/∂t = -∂Ey/∂x + ∂Ex/∂y   ∂Ez/∂t = c²(∂By/∂x - ∂Bx/∂y) - Jz/ε0
//
// Esirkepov (CIC / S1, cell units, one-step displacement < 1 cell):
//   With node shapes S0,S1 over the 3-point stencil {i0-1, i0, i0+1} and
//   ΔS = S1 - S0, the density-decomposition weights are
//     Wx(i,j) = ΔSx(i)·(Sy0(j) + ½ΔSy(j))
//     Wy(i,j) = ΔSy(j)·(Sx0(i) + ½ΔSx(i))
//     Wz(i,j) = Sx0Sy0 + ½ΔSx·Sy0 + ½Sx0·ΔSy + ⅓ΔSx·ΔSy
//   and the currents are the prefix sums
//     Jx(i+½,j) = -(q w /(dt·dy_p)) Σ_{i'≤i} Wx(i',j)
//     Jy(i,j+½) = -(q w /(dt·dx_p)) Σ_{j'≤j} Wy(i,j')
//     Jz(i,j)   =  (q w v_z/(dx_p·dy_p)) · Wz(i,j)
//   which satisfy the DISCRETE continuity equation against the CIC node
//   charge ρ(i,j) = Σ q w Sx(i)Sy(j)/(dx_p dy_p) exactly (test-enforced):
//     (ρ^{n+1}-ρ^n)/dt + (Jx(i+½)-Jx(i-½))/dx_p + (Jy(j+½)-Jy(j-½))/dy_p = 0.

#ifndef ARC_PIC_YEE2D_HPP
#define ARC_PIC_YEE2D_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/pusher.hpp"     // boris_update_full (shared rotation)

#include <cmath>

namespace arc {

struct YeeMaxwell {};          // FieldModel tag (plan §11.1 third branch)

// ---- field views -----------------------------------------------------------

struct YeeViews {
    float *ex = nullptr, *ey = nullptr, *ez = nullptr;
    float *bx = nullptr, *by = nullptr, *bz = nullptr;
    float *jx = nullptr, *jy = nullptr, *jz = nullptr;
    int nx = 0, ny = 0;
    float dxp = 1.f, dyp = 1.f;        // physical cell sizes
    float c2 = 1.f, dt = 0.f;

    __host__ __device__ int idx(int i, int j) const {
        i += (i < 0) * nx - (i >= nx) * nx;      // periodic wrap (|off| ≤ nx)
        j += (j < 0) * ny - (j >= ny) * ny;
        return j * nx + i;
    }
};

namespace yee {

// ---- field update kernels ---------------------------------------------------

// B half-step: B += dt2 * (-curl E)
static __global__ void k_faraday(YeeViews v, float dt2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || j >= v.ny) return;
    const int c = v.idx(i, j);
    const float ez_xp = v.ez[v.idx(i + 1, j)], ez_yp = v.ez[v.idx(i, j + 1)];
    const float ex_yp = v.ex[v.idx(i, j + 1)];
    const float ey_xp = v.ey[v.idx(i + 1, j)];
    v.bx[c] -= dt2 * (ez_yp - v.ez[c]) / v.dyp;
    v.by[c] += dt2 * (ez_xp - v.ez[c]) / v.dxp;
    v.bz[c] -= dt2 * ((ey_xp - v.ey[c]) / v.dxp - (ex_yp - v.ex[c]) / v.dyp);
}

// E full step: E += dt * (c² curl B - J)   (ε0 = 1)
static __global__ void k_ampere(YeeViews v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || j >= v.ny) return;
    const int c = v.idx(i, j);
    const float bz_ym = v.bz[v.idx(i, j - 1)];
    const float bz_xm = v.bz[v.idx(i - 1, j)];
    const float bx_ym = v.bx[v.idx(i, j - 1)];
    const float by_xm = v.by[v.idx(i - 1, j)];
    v.ex[c] += v.dt * ( v.c2 * (v.bz[c] - bz_ym) / v.dyp - v.jx[c]);
    v.ey[c] += v.dt * (-v.c2 * (v.bz[c] - bz_xm) / v.dxp - v.jy[c]);
    v.ez[c] += v.dt * ( v.c2 * ((v.by[c] - by_xm) / v.dxp
                              - (v.bx[c] - bx_ym) / v.dyp) - v.jz[c]);
}

// ---- staggered CIC gather ----------------------------------------------------
// Positions are in CELL units (repo convention). Each component interpolates
// from its own staggered lattice: offset (ox, oy) in half-cells.
__device__ inline float gather_stag(const float* a, const YeeViews& v,
                                    float x, float y, float ox, float oy) {
    const float xs = x - ox, ys = y - oy;
    int i0 = (int)floorf(xs), j0 = (int)floorf(ys);
    const float fx = xs - i0, fy = ys - j0;
    return (1.f - fx) * (1.f - fy) * a[v.idx(i0,     j0)]
         +        fx  * (1.f - fy) * a[v.idx(i0 + 1, j0)]
         + (1.f - fx) *        fy  * a[v.idx(i0,     j0 + 1)]
         +        fx  *        fy  * a[v.idx(i0 + 1, j0 + 1)];
}

// ---- fused push (Boris, staggered gather) + Esirkepov deposit -----------------

// CIC node shape over the 3-point stencil {ic-1, ic, ic+1} for position x
// (cell units): S(ic-1)=..., contributions outside are zero when |x-ic|≤1.
__device__ inline void s1_shape(float x, int ic, float* S) {
    // node n at integer position p = ic-1+n ; S = max(0, 1-|x-p|)
    #pragma unroll
    for (int n = 0; n < 3; ++n) {
        const float d = fabsf(x - (float)(ic - 1 + n));
        S[n] = d < 1.f ? 1.f - d : 0.f;
    }
}

__device__ inline void s1_shape4(float x, int ib, float* S) {
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        const float d = fabsf(x - (float)(ib - 1 + n));
        S[n] = d < 1.f ? 1.f - d : 0.f;
    }
}

__device__ inline double yee_pump_ramp(double t, double trmp, double toff) {
    if (t >= toff) return 0.0;
    if (t < trmp)  return t / trmp;
    if (t > toff - trmp) return (toff - t) / trmp;
    return 1.0;
}

static __global__ void k_push_esirkepov(ParticleViews p, YeeViews v, RunParams rp,
                                        double tnow) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    const float x0 = p.x[t], y0 = p.y[t];

    // gather staggered E, B at the particle (add uniform external B0)
    float Ex = gather_stag(v.ex, v, x0, y0, 0.5f, 0.f);
    float Ey = gather_stag(v.ey, v, x0, y0, 0.f, 0.5f);
    float Ez = gather_stag(v.ez, v, x0, y0, 0.f, 0.f);

    // external whistler pump (An et al. 2019), analytic at the particle:
    // E += Re{ E~_a e^{i(k0 x - w0 t)} } * ramp  — NOT added to the evolved
    // Yee arrays, so the stored E stays purely self-consistent.
    if (rp.pump && tnow < rp.pump_toff) {
        const double th = rp.pump_k0 * (double)x0 * v.dxp - rp.pump_w0 * tnow;
        const double r  = yee_pump_ramp(tnow, rp.pump_trmp, rp.pump_toff);
        Ex += (float)(rp.pump_ex * cos(th) * r);
        Ey += (float)(-rp.pump_ey * sin(th) * r);
        Ez += (float)(rp.pump_ez * cos(th) * r);
    }
    const float Bx = gather_stag(v.bx, v, x0, y0, 0.f, 0.5f) + rp.B0[0];
    const float By = gather_stag(v.by, v, x0, y0, 0.5f, 0.f) + rp.B0[1];
    const float Bz = gather_stag(v.bz, v, x0, y0, 0.5f, 0.5f) + rp.B0[2];

    float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float qmh = (float)(rp.qm * 0.5 * rp.dt);
    detail::boris_update_full(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, qmh);
    p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;

    // move (cell units; non-relativistic v1, matches the spectral path)
    const float x1 = x0 + (float)(ux * rp.dt / (double)v.dxp);
    const float y1 = y0 + (float)(uy * rp.dt / (double)v.dyp);
    p.x[t] = x1; p.y[t] = y1;    // periodic wrap happens in Particles::migrate

    // ---- Esirkepov CIC deposit over the union stencil ----
    // 4-node stencils {ib-1..ib+2}, ib = floor(min(x0,x1)): with |Δ| < 1 the
    // union of the old and new CIC supports always fits.
    const int ib = (int)floorf(fminf(x0, x1));
    const int jb = (int)floorf(fminf(y0, y1));
    float Sx0[4], Sx1[4], Sy0[4], Sy1[4];
    s1_shape4(x0, ib, Sx0); s1_shape4(x1, ib, Sx1);
    s1_shape4(y0, jb, Sy0); s1_shape4(y1, jb, Sy1);

    const float qw   = (float)rp.qm * p.w[t];      // q = qm (m = 1 code units)
    const float fdt  = (float)(1.0 / rp.dt);
    const float cJx  = -qw * fdt / v.dyp;          // prefix-sum prefactor
    const float cJy  = -qw * fdt / v.dxp;
    const float cJz  =  qw * (float)uz / (v.dxp * v.dyp);

    float DSx[4], DSy[4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) { DSx[n] = Sx1[n] - Sx0[n]; DSy[n] = Sy1[n] - Sy0[n]; }

    // Jx: prefix sum along i for each j of Wx = DSx*(Sy0 + DSy/2)
    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        const float sy = Sy0[nj] + 0.5f * DSy[nj];
        float acc = 0.f;
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc += DSx[ni] * sy;
            if (acc != 0.f)
                atomicAdd(&v.jx[v.idx(ib - 1 + ni, jb - 1 + nj)], cJx * acc);
        }
    }
    // Jy: prefix sum along j for each i of Wy = DSy*(Sx0 + DSx/2)
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const float sx = Sx0[ni] + 0.5f * DSx[ni];
        float acc = 0.f;
        #pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            acc += DSy[nj] * sx;
            if (acc != 0.f)
                atomicAdd(&v.jy[v.idx(ib - 1 + ni, jb - 1 + nj)], cJy * acc);
        }
    }
    // Jz: Wz weights (Esirkepov third-direction form)
    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const float wz = Sx0[ni] * Sy0[nj]
                           + 0.5f * (DSx[ni] * Sy0[nj] + Sx0[ni] * DSy[nj])
                           + (1.f / 3.f) * DSx[ni] * DSy[nj];
            if (wz != 0.f)
                atomicAdd(&v.jz[v.idx(ib - 1 + ni, jb - 1 + nj)], cJz * wz);
        }
    }
}

// One pass of 3×3 binomial ([1,2,1]⊗[1,2,1]/16) current smoothing — the OSIRIS
// "smooth" block (order = number of passes). Linear + shift-invariant, so the
// implied ∂ρ/∂t is filtered identically and Gauss consistency is preserved
// against the equally-filtered ρ.
static __global__ void k_binomial3x3(const float* src, float* dst, YeeViews v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || j >= v.ny) return;
    const float w[3] = {0.25f, 0.5f, 0.25f};
    float acc = 0.f;
    #pragma unroll
    for (int dj = -1; dj <= 1; ++dj)
        #pragma unroll
        for (int di = -1; di <= 1; ++di)
            acc += w[di + 1] * w[dj + 1] * src[v.idx(i + di, j + dj)];
    dst[j * v.nx + i] = acc;
}

// CIC node charge deposit (for Gauss/continuity diagnostics)
static __global__ void k_rho_nodes(ParticleViews p, YeeViews v, float* rho, double qsign) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;
    const float x = p.x[t], y = p.y[t];
    const int ib = (int)floorf(x), jb = (int)floorf(y);
    float Sx[3], Sy[3];
    s1_shape(x, ib, Sx); s1_shape(y, jb, Sy);
    const float coef = (float)qsign * p.w[t] / (v.dxp * v.dyp);
    for (int nj = 0; nj < 3; ++nj)
        for (int ni = 0; ni < 3; ++ni) {
            const float s = Sx[ni] * Sy[nj];
            if (s != 0.f)
                atomicAdd(&rho[v.idx(ib - 1 + ni, jb - 1 + nj)], coef * s);
        }
}

// field energy: out[0] += eps0 E²/2 · dA, out[1] += c² B²/2 · dA (FP64 sums)
static __global__ void k_energy(YeeViews v, double* out) {
    __shared__ double sE, sB;
    if (threadIdx.x == 0) { sE = 0; sB = 0; }
    __syncthreads();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0, b = 0;
    if (n < v.nx * v.ny) {
        const double dA = (double)v.dxp * v.dyp;
        e = 0.5 * ((double)v.ex[n] * v.ex[n] + (double)v.ey[n] * v.ey[n]
                 + (double)v.ez[n] * v.ez[n]) * dA;
        b = 0.5 * v.c2 * ((double)v.bx[n] * v.bx[n] + (double)v.by[n] * v.by[n]
                        + (double)v.bz[n] * v.bz[n]) * dA;
    }
    atomicAdd(&sE, e); atomicAdd(&sB, b);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(&out[0], sE); atomicAdd(&out[1], sB); }
}

} // namespace yee

// ---- owning field container --------------------------------------------------

class YeeFields {
public:
    YeeFields(const Grid& g, double c, double dt)
        : nx_(g.nx), ny_(g.ny),
          ex_(g.real_size()), ey_(g.real_size()), ez_(g.real_size()),
          bx_(g.real_size()), by_(g.real_size()), bz_(g.real_size()),
          jx_(g.real_size()), jy_(g.real_size()), jz_(g.real_size()) {
        v_.nx = nx_; v_.ny = ny_;
        v_.dxp = (float)g.dx; v_.dyp = (float)g.dy;
        v_.c2 = (float)(c * c); v_.dt = (float)dt;
        zero();
    }
    void zero(cudaStream_t s = nullptr) {
        ex_.zero(s); ey_.zero(s); ez_.zero(s);
        bx_.zero(s); by_.zero(s); bz_.zero(s);
        zero_j(s);
    }
    void zero_j(cudaStream_t s = nullptr) { jx_.zero(s); jy_.zero(s); jz_.zero(s); }

    YeeViews views() {
        YeeViews v = v_;
        v.ex = ex_.data(); v.ey = ey_.data(); v.ez = ez_.data();
        v.bx = bx_.data(); v.by = by_.data(); v.bz = bz_.data();
        v.jx = jx_.data(); v.jy = jy_.data(); v.jz = jz_.data();
        return v;
    }
    int nx() const { return nx_; }
    int ny() const { return ny_; }

    DeviceArray<float> ex_, ey_, ez_, bx_, by_, bz_, jx_, jy_, jz_;

private:
    int nx_, ny_;
    YeeViews v_;
};

} // namespace arc

#endif // ARC_PIC_YEE2D_HPP
