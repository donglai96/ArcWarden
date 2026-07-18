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

#include "pic/background_b0.hpp"  // M4 parabolic B0(x) + mirror effective field
#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/pusher.hpp"     // boris_update_full + boris_rotate

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

// Boris push + move for one particle on the Yee grid (staggered gather from
// GLOBAL memory + pump), shared verbatim by the flat and tiled deposit kernels
// so the physics is bit-identical on both paths. Returns old/new positions.
__device__ inline void yee_advance_particle(ParticleViews& p, const YeeViews& v,
                                            const RunParams& rp, double tnow, int t,
                                            float& x0, float& y0,
                                            float& x1, float& y1) {
    x0 = p.x[t]; y0 = p.y[t];

    float Ex = gather_stag(v.ex, v, x0, y0, 0.5f, 0.f);
    float Ey = gather_stag(v.ey, v, x0, y0, 0.f, 0.5f);
    float Ez = gather_stag(v.ez, v, x0, y0, 0.f, 0.f);
    if (rp.pump && tnow < rp.pump_toff) {
        const double th = rp.pump_k0 * (double)x0 * v.dxp - rp.pump_w0 * tnow;
        const double r  = yee_pump_ramp(tnow, rp.pump_trmp, rp.pump_toff);
        Ex += (float)(rp.pump_ex * cos(th) * r);
        Ey += (float)(-rp.pump_ey * sin(th) * r);
        Ez += (float)(rp.pump_ez * cos(th) * r);
    }
    // wave-only B kept separate: the M3 delta-f weight equation needs the
    // perturbation force (the uniform B0 rotation conserves the gyrotropic f0)
    const float dBx = gather_stag(v.bx, v, x0, y0, 0.f, 0.5f);
    const float dBy = gather_stag(v.by, v, x0, y0, 0.5f, 0.f);
    const float dBz = gather_stag(v.bz, v, x0, y0, 0.5f, 0.5f);
    const float Bx = dBx + rp.B0[0];
    const float By = dBy + rp.B0[1];
    const float Bz = dBz + rp.B0[2];

    float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float uxo = ux, uyo = uy, uzo = uz;   // u^{n-1/2} (delta-f centering)
    const float qmh = (float)(rp.qm * 0.5 * rp.dt);
    if (rp.b0_prof) {
        // M4 parabolic B0(x) + mirror force (background_b0.hpp): the effective
        // field needs the MID-KICK u, so split the kick-rotate-kick here.
        // Requires B0 ∥ x̂; rp.B0[1] = rp.B0[2] = 0 (validated by callers).
        const float xph = x0 * v.dxp;
        const float b0  = bg::b0x(rp, xph);
        const float mc  = bg::db0dx(rp, xph) / (2.f * b0 * (float)rp.qm);
        ux += qmh * Ex; uy += qmh * Ey; uz += qmh * Ez;
        detail::boris_rotate(ux, uy, uz,
                             dBx + b0, dBy + mc * uz, dBz - mc * uy, qmh);
        ux += qmh * Ex; uy += qmh * Ey; uz += qmh * Ez;
    } else {
        detail::boris_update_full(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, qmh);
    }
    p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;

    // M3 delta-f weight update (chirp1d / Tao PPCF 2017 eq. 19 form):
    //   dwd/dt = -(1-wd)(q/m) F·∂ln f0/∂u,  ∂ln f0/∂u = (-ux/Tpar, -uy/Tperp,
    //   -uz/Tperp) for the bi-Maxwellian reference with B0 ∥ x̂,
    //   F = δE + v×δB (wave fields incl. pump).
    // TIME-CENTERING MATTERS: evaluate u at t^n = (u^{n-1/2}+u^{n+1/2})/2, the
    // same time level as the gathered fields. Using the post-push u alone puts
    // the drive a half step out of quadrature and NUMERICALLY ANTI-DAMPS the
    // fast EM branch (measured: gamma_num ~ +1e-3 at w = 1.28, dt = 0.025,
    // scaling with dt — found via the deltaf_consistency gate).
    if (rp.deltaf) {
        const float uxc = 0.5f * (ux + uxo);
        const float uyc = 0.5f * (uy + uyo);
        const float uzc = 0.5f * (uz + uzo);
        const float Fx = Ex + (uyc * dBz - uzc * dBy);
        const float Fy = Ey + (uzc * dBx - uxc * dBz);
        const float Fz = Ez + (uxc * dBy - uyc * dBx);
        const float S  = Fx * uxc / (float)rp.df_tpar
                       + (Fy * uyc + Fz * uzc) / (float)rp.df_tperp;
        const float wd = p.wd[t];
        // Accumulator precision study (docs/WEIGHT_PRECISION.md). The drive S is
        // FP32 in all modes — the question is roundoff of the ACCUMULATION over
        // ~1e5 steps. Kahan uses __fadd_rn so the compensation term cannot be
        // algebraically folded away; FP64 keeps wdd and mirrors it to wd for the
        // deposit. Default (df_wprec = 0) is bit-identical to the plain update.
        if (rp.df_wprec == 2) {
            const double wo = p.wdd[t];
            const double wn = wo + rp.dt * rp.qm * (1.0 - wo) * (double)S;
            p.wdd[t] = wn;
            p.wd[t]  = (float)wn;
        } else if (rp.df_wprec == 1) {
            const float inc = (float)(rp.dt * rp.qm) * (1.f - wd) * S;
            const float c   = p.wc[t];
            const float yk  = __fadd_rn(inc, -c);
            const float wn  = __fadd_rn(wd, yk);
            p.wc[t] = __fadd_rn(__fadd_rn(wn, -wd), -yk);
            p.wd[t] = wn;
        } else {
            p.wd[t] = wd + (float)(rp.dt * rp.qm) * (1.f - wd) * S;
        }
    }

    x1 = x0 + (float)(ux * rp.dt / (double)v.dxp);
    y1 = y0 + (float)(uy * rp.dt / (double)v.dyp);
    // M2 bounded x: specular reflection at the DOMAIN ends (Umeda scheme —
    // the damping layers may contain plasma; the mask kills the fields, the
    // wall keeps particles out of the periodic x-wrap). Reflect before the
    // deposit so Esirkepov sees the folded path (still |Δ| < 1 cell).
    if (rp.bnd_x) {
        if (x1 < 0.f)               { x1 = -x1;                  p.ux[t] = -ux; }
        else if (x1 >= (float)v.nx) { x1 = 2.f * v.nx - x1;      p.ux[t] = -ux; }
        if (x1 >= (float)v.nx)      // float edge x1 == nx: keep off the wrap
            x1 = nextafterf((float)v.nx, 0.f);
        // hybrid mode: damp the transverse momentum inside the layers so the
        // coherent whistler current dies with the field it would re-radiate
        if (rp.bnd_x == 2) {
            const float nd = (float)rp.bnd_nd;
            float d = 0.f;
            if (x1 < nd)                 d = (nd - x1) / nd;
            else if (x1 > v.nx - nd)     d = (x1 - (v.nx - nd)) / nd;
            if (d > 0.f) {
                const float m = __expf((float)(-rp.bnd_numax * rp.dt) * d * d);
                p.uy[t] = uy * m; p.uz[t] = uz * m;
            }
        }
    }
    // position write-back is the CALLER's job: the flat kernel stores x1
    // unwrapped (Particles::migrate wraps), the tiled kernel stores the
    // wrapped position directly (fused migrate) — avoids a double write.
}

// Esirkepov scatter over the 4x4 union stencil, deposit target abstracted as a
// Sink (global atomics or a shared-memory tile) so both kernels share one
// prefix-sum implementation.
template<class Sink>
__device__ inline void esirkepov_scatter(float x0, float y0, float x1, float y1,
                                         float qw, float uz, const YeeViews& v,
                                         float invdt, int ib, int jb, Sink&& sk) {
    float Sx0[4], Sx1[4], Sy0[4], Sy1[4];
    s1_shape4(x0, ib, Sx0); s1_shape4(x1, ib, Sx1);
    s1_shape4(y0, jb, Sy0); s1_shape4(y1, jb, Sy1);

    const float cJx = -qw * invdt / v.dyp;
    const float cJy = -qw * invdt / v.dxp;
    const float cJz =  qw * uz / (v.dxp * v.dyp);

    float DSx[4], DSy[4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) { DSx[n] = Sx1[n] - Sx0[n]; DSy[n] = Sy1[n] - Sy0[n]; }

    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        const float sy = Sy0[nj] + 0.5f * DSy[nj];
        float acc = 0.f;
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc += DSx[ni] * sy;
            if (acc != 0.f) sk.jx(ib - 1 + ni, jb - 1 + nj, cJx * acc);
        }
    }
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const float sx = Sx0[ni] + 0.5f * DSx[ni];
        float acc = 0.f;
        #pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            acc += DSy[nj] * sx;
            if (acc != 0.f) sk.jy(ib - 1 + ni, jb - 1 + nj, cJy * acc);
        }
    }
    #pragma unroll
    for (int nj = 0; nj < 4; ++nj)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const float wz = Sx0[ni] * Sy0[nj]
                           + 0.5f * (DSx[ni] * Sy0[nj] + Sx0[ni] * DSy[nj])
                           + (1.f / 3.f) * DSx[ni] * DSy[nj];
            if (wz != 0.f) sk.jz(ib - 1 + ni, jb - 1 + nj, cJz * wz);
        }
}

// Global-atomic deposit sink (flat path + tiled-path stray fallback). Uses the
// full modular wrap: the 4-node union stencil {jb-1..jb+2} runs up to TWO
// periods out of range on tiny grids (ny = 1 runs), beyond the one-period
// contract of YeeViews::idx.
struct GlobalJSink {
    YeeViews v;
    __device__ int idx(int i, int j) const {
        return Grid::wrap_far(j, v.ny) * v.nx + Grid::wrap_far(i, v.nx);
    }
    __device__ void jx(int i, int j, float a) { atomicAdd(&v.jx[idx(i, j)], a); }
    __device__ void jy(int i, int j, float a) { atomicAdd(&v.jy[idx(i, j)], a); }
    __device__ void jz(int i, int j, float a) { atomicAdd(&v.jz[idx(i, j)], a); }
};

static __global__ void k_push_esirkepov(ParticleViews p, YeeViews v, RunParams rp,
                                        double tnow) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    float x0, y0, x1, y1;
    yee_advance_particle(p, v, rp, tnow, t, x0, y0, x1, y1);
    p.x[t] = x1; p.y[t] = y1;    // unwrapped; Particles::migrate wraps

    // ---- Esirkepov CIC deposit over the union stencil ----
    // 4-node stencils {ib-1..ib+2}, ib = floor(min(x0,x1)): with |Δ| < 1 the
    // union of the old and new CIC supports always fits.
    const int ib = (int)floorf(fminf(x0, x1));
    const int jb = (int)floorf(fminf(y0, y1));
    float qw = (float)rp.qm * p.w[t];              // q = qm (m = 1 code units)
    if (rp.deltaf) qw *= p.wd[t];                  // DeltaF policy: δJ = q w wd v
    esirkepov_scatter(x0, y0, x1, y1, qw, p.uz[t], v, (float)(1.0 / rp.dt),
                      ib, jb, GlobalJSink{v});
}

// TiledBinnedDeposit for the Yee path (M9): one block per TX*TY spatial tile,
// particles PHYSICALLY tile-ordered (Particles::sort_by_tile) so SoA reads and
// write-backs are coalesced. J is privatized in shared memory over the tile
// plus a (DRIFT+3)-node apron: DRIFT cells of motion since the last sort plus
// the 4-node union stencil always fit. Gathers stay in global memory — a
// tile's particles hit the same few field cells, so L2 serves them; the win
// to be had is the deposit, which replaces ~20N scattered global atomics with
// shared atomics + one apron flush per tile (~SW*SH*3 per tile, ppc-free).
// Particles that moved beyond DRIFT cells since the last sort (or wrapped
// periodically) fall back to global atomics — always correct.
template<int TX, int TY, int DRIFT>
static __global__ void k_push_esirkepov_tiled(ParticleViews p, BinViews b,
                                              YeeViews v, RunParams rp,
                                              double tnow, int blocks_per_tile) {
    constexpr int PAD = DRIFT + 3;               // node reach below tile origin
    constexpr int SW  = TX + 2 * DRIFT + 6;      // local nodes [-PAD, TX+DRIFT+2]
    constexpr int SH  = TY + 2 * DRIFT + 6;
    __shared__ float s_jx[SH * SW], s_jy[SH * SW], s_jz[SH * SW];

    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX;
    const int gj0 = (tile / b.ntx) * TY;

    for (int c = threadIdx.x; c < SH * SW; c += blockDim.x)
        s_jx[c] = s_jy[c] = s_jz[c] = 0.f;
    __syncthreads();

    struct SharedJSink {
        float *jx_, *jy_, *jz_; int i0, j0;
        __device__ void jx(int i, int j, float a) { atomicAdd(&jx_[(j - j0) * SW + (i - i0)], a); }
        __device__ void jy(int i, int j, float a) { atomicAdd(&jy_[(j - j0) * SW + (i - i0)], a); }
        __device__ void jz(int i, int j, float a) { atomicAdd(&jz_[(j - j0) * SW + (i - i0)], a); }
    } shs{s_jx, s_jy, s_jz, gi0 - PAD, gj0 - PAD};

    const float invdt = (float)(1.0 / rp.dt);
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int t = beg + lane * blockDim.x + threadIdx.x; t < end; t += step) {
        float x0, y0, x1, y1;
        yee_advance_particle(p, v, rp, tnow, t, x0, y0, x1, y1);

        const int ib = (int)floorf(fminf(x0, x1));
        const int jb = (int)floorf(fminf(y0, y1));
        float qw = (float)rp.qm * p.w[t];
        if (rp.deltaf) qw *= p.wd[t];              // DeltaF policy: δJ = q w wd v
        // union stencil {ib-1..ib+2} inside the shared apron?
        const int il = ib - gi0, jl = jb - gj0;
        if (il - 1 >= -PAD && il + 2 <= TX + DRIFT + 2 &&
            jl - 1 >= -PAD && jl + 2 <= TY + DRIFT + 2)
            esirkepov_scatter(x0, y0, x1, y1, qw, p.uz[t], v, invdt, ib, jb, shs);
        else                                            // stray since last sort
            esirkepov_scatter(x0, y0, x1, y1, qw, p.uz[t], v, invdt, ib, jb,
                              GlobalJSink{v});

        // fused migrate (same formula as particle_migrate_kernel): the tiled
        // path skips the separate wrap+cell pass — a whole extra SoA sweep.
        float xw = fmodf(x1, (float)v.nx); if (xw < 0.f) xw += (float)v.nx;
        float yw = fmodf(y1, (float)v.ny); if (yw < 0.f) yw += (float)v.ny;
        if (xw >= (float)v.nx) xw = 0.f;   // += wrap of a tiny negative rounds
        if (yw >= (float)v.ny) yw = 0.f;   // to the edge exactly (see migrate)
        p.x[t] = xw; p.y[t] = yw;
        int ci = (int)floorf(xw); if (ci >= v.nx) ci = v.nx - 1;
        int cj = (int)floorf(yw); if (cj >= v.ny) cj = v.ny - 1;
        p.cell[t] = cj * v.nx + ci;
    }
    __syncthreads();

    for (int c = threadIdx.x; c < SH * SW; c += blockDim.x) {
        const float jx = s_jx[c], jy = s_jy[c], jz = s_jz[c];
        if (jx != 0.f || jy != 0.f || jz != 0.f) {
            const int gc = v.idx(gi0 - PAD + (c % SW), gj0 - PAD + (c / SW));
            atomicAdd(&v.jx[gc], jx);
            atomicAdd(&v.jy[gc], jy);
            atomicAdd(&v.jz[gc], jz);
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

// CIC node charge deposit (for Gauss/continuity diagnostics). use_wd = 1 on
// the delta-f branch: deposit the perturbation charge δρ = Σ q w wd S.
static __global__ void k_rho_nodes(ParticleViews p, YeeViews v, float* rho, double qsign,
                                   int use_wd) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;
    const float x = p.x[t], y = p.y[t];
    const int ib = (int)floorf(x), jb = (int)floorf(y);
    float Sx[3], Sy[3];
    s1_shape(x, ib, Sx); s1_shape(y, jb, Sy);
    float coef = (float)qsign * p.w[t] / (v.dxp * v.dyp);
    if (use_wd) coef *= p.wd[t];
    for (int nj = 0; nj < 3; ++nj)
        for (int ni = 0; ni < 3; ++ni) {
            const float s = Sx[ni] * Sy[nj];
            if (s != 0.f)
                atomicAdd(&rho[v.idx(ib - 1 + ni, jb - 1 + nj)], coef * s);
        }
}

// M2: Umeda-style multiplicative damping masks over the two x-end layers,
// applied to ALL wave fields once per step (the grid holds only wave fields;
// the uniform external B0 lives in RunParams and is never damped). mn = mask
// at integer x (Ey, Ez, Bx sites), mh = mask at half-integer x (Ex, By, Bz);
// the y stagger is irrelevant for x-direction masks.
static __global__ void k_damp_x(YeeViews v, const float* __restrict__ mn,
                                const float* __restrict__ mh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || j >= v.ny) return;
    const float a = mn[i], b = mh[i];
    if (a == 1.f && b == 1.f) return;
    const int c = j * v.nx + i;
    v.ey[c] *= a; v.ez[c] *= a; v.bx[c] *= a;
    v.ex[c] *= b; v.by[c] *= b; v.bz[c] *= b;
}

// M2/M10 antenna: add the rotating transverse current column (see RunParams
// ant_*) into J before the Ampère update. Jy site (i, j+½) and Jz site (i, j)
// share integer x, so one Gaussian g(i) serves both.
static __global__ void k_antenna(YeeViews v, RunParams rp, double tnow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || j >= v.ny) return;
    const double dxi = i - rp.ant_x0;
    if (fabs(dxi) > 4.0 * rp.ant_sigma) return;
    if (rp.ant_toff > 0 && tnow >= rp.ant_toff) return;   // toff = 0: always on
    // trapezoidal envelope (yee_pump_ramp shape): linear up over trmp, flat,
    // linear down over the last trmp before toff — a hard turn-off radiates a
    // broadband transient (slow near-ωce whistlers that pollute the domain)
    double ramp = 1.0;
    if (rp.ant_trmp > 0) {
        if (tnow < rp.ant_trmp)                      ramp = tnow / rp.ant_trmp;
        else if (rp.ant_toff > 0 &&
                 tnow > rp.ant_toff - rp.ant_trmp)   ramp = (rp.ant_toff - tnow) / rp.ant_trmp;
    }
    const double g = rp.ant_amp * ramp * exp(-dxi * dxi / (2.0 * rp.ant_sigma * rp.ant_sigma));
    const double ph = rp.ant_w0 * tnow;
    const int c = j * v.nx + i;
    // electron-gyration (R-mode) sense: (Jy + iJz) ∝ e^{+i w0 t}. The
    // opposite sense is an L-mode drive — evanescent below ω_ce, radiates
    // nothing at w0 (verified the hard way: probe spectrum showed only
    // broadband ramp transients near ω_ce).
    v.jy[c] += (float)(g * cos(ph));
    v.jz[c] += (float)(g * sin(ph));
}

// Gauss residual r = divE - rho at NODES (where the CIC rho lives):
//   divE(i,j) = (Ex(i,j)-Ex(i-1,j))/dx + (Ey(i,j)-Ey(i,j-1))/dy
// The uniform neutralizing ion background and the missing initial Poisson
// solve both live in r(t=0); Esirkepov guarantees r(t) ≡ r(0) to round-off,
// so the running diagnostic is the DRIFT rms(r - r0), not rms(r).
static __global__ void k_gauss_residual(YeeViews v, const float* rho, float* r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || j >= v.ny) return;
    const int c = v.idx(i, j);
    const float dive = (v.ex[c] - v.ex[v.idx(i - 1, j)]) / v.dxp
                     + (v.ey[c] - v.ey[v.idx(i, j - 1)]) / v.dyp;
    r[c] = dive - rho[c];
}

// conservation stats: out[0] += divB², out[1] += (r - r0)²  (FP64 sums)
//   divB(i+½,j+½) = (Bx(i+1,j+½)-Bx(i,j+½))/dx + (By(i+½,j+1)-By(i+½,j))/dy
// (Faraday preserves this exactly on the Yee mesh; B starts at 0 and the
// uniform external B0 never touches the grid, so divB itself is the metric.)
static __global__ void k_div_stats(YeeViews v, const float* r, const float* r0,
                                   double* out) {
    __shared__ double sB, sG;
    if (threadIdx.x == 0) { sB = 0; sG = 0; }
    __syncthreads();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    double b2 = 0, g2 = 0;
    if (n < v.nx * v.ny) {
        const int i = n % v.nx, j = n / v.nx;
        const float divb = (v.bx[v.idx(i + 1, j)] - v.bx[n]) / v.dxp
                         + (v.by[v.idx(i, j + 1)] - v.by[n]) / v.dyp;
        b2 = (double)divb * divb;
        const double d = (double)r[n] - r0[n];
        g2 = d * d;
    }
    atomicAdd(&sB, b2); atomicAdd(&sG, g2);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(&out[0], sB); atomicAdd(&out[1], sG); }
}

// M3 weight diagnostics (FP64 sums): out[0] += wd, out[1] += wd²,
// out[2] = max|wd| via atomicMax on the positive-float bit pattern.
static __global__ void k_wd_stats(ParticleViews p, double* out, unsigned int* mx) {
    __shared__ double s1, s2;
    __shared__ unsigned int sm;
    if (threadIdx.x == 0) { s1 = 0; s2 = 0; sm = 0; }
    __syncthreads();
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < p.n) {
        const float w = p.wd[t];
        atomicAdd(&s1, (double)w);
        atomicAdd(&s2, (double)w * w);
        atomicMax(&sm, __float_as_uint(fabsf(w)));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&out[0], s1); atomicAdd(&out[1], s2); atomicMax(mx, sm);
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
