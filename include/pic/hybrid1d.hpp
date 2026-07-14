// ArcWarden — 1D field-aligned electron-hybrid model (L-shell plan, M4 gate).
//
// Physics: whistler-mode chorus generation along a single field line, in the
// Katoh–Omura electron-hybrid formulation:
//   - transverse EM fields (Ey,Ez,By,Bz) on a 1D staggered (Yee) grid in h≡x,
//     parallel propagation only, no E∥ (quasi-neutral cold background);
//   - COLD electrons: linearized magnetized fluid  dVc/dt = -(E + Vc×B0 x̂)
//     (q/m = -1), giving the whistler dispersion medium with zero noise;
//   - HOT electrons: kinetic macro-particles, relativistic Boris push in
//     B = B0(x) x̂ + δB⊥ plus the adiabatic mirror force of the unresolved
//     perpendicular B0 component, folded into the Boris rotation as an
//     effective field  B_mir = -(B0'/(2 B0)) (u × x̂)  — this makes the mirror
//     force exactly energy-conserving in the discrete scheme, since
//     u × B_mir ∝ u × (u × x̂) reproduces  du∥/dt = -(u⊥²/2γB0) ∂B0/∂x.
//   - parabolic background field B0(x) = 1 + a (x-xc)², masked damping
//     regions (Umeda-style multiplicative masks) at both ends, particle
//     reflection at the physical-region edge; optional fully periodic
//     uniform-B0 mode for linear benchmarks.
//
// Units: Ω_e0 = c = m_e = e = ε0 = 1. So B is in electron-gyrofrequency
// units (B0eq = 1), lengths in c/Ω_e0, time in Ω_e0⁻¹, n_cold = (ω_pe/Ω_e0)²,
// and the wave amplitude diagnostic |δB⊥| is directly δB/B0eq.
//
// Field equations (1D, ∂/∂x only):
//   ∂By/∂t =  ∂Ez/∂x          ∂Ey/∂t = -c² ∂Bz/∂x - (Jc+Jh)y
//   ∂Bz/∂t = -∂Ey/∂x          ∂Ez/∂t = +c² ∂By/∂x - (Jc+Jh)z
// Grid: nodes x_i = i·dx carry Ey,Ez,Vc,J; faces x_{i+1/2} carry By,Bz.
//
// Hot equilibrium load: f0(E,μ) mapped from an equatorial bi-Maxwellian:
//   T∥ = const,  1/T⊥(x) = (1-1/b)/T∥ + (1/b)/T⊥eq,  b = B0(x)/B0eq,
//   n_h(x)/n_h,eq = T⊥(x)/T⊥eq   (equal-weight markers, count ∝ n_h(x)).

#ifndef ARC_PIC_HYBRID1D_HPP
#define ARC_PIC_HYBRID1D_HPP

#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace arc {

// ---- parameters -----------------------------------------------------------

struct Chirp1DParams {
    // grid / time
    int    nx     = 4096;      // cells; nodes = nx+1 (bounded) or nx (periodic)
    double dx     = 0.1;       // c/Ω_e0
    double dt     = 0.04;      // Ω_e0⁻¹  (CFL: c·dt < dx)
    long   nsteps = 100000;

    // plasma
    double wpe      = 4.0;     // ω_pe/Ω_e0 (cold);  n_c = wpe²
    double a        = 0.0;     // B0(x) = 1 + a (x-xc)²   [(Ω_e0/c)²]
    double nh       = 0.0;     // n_hot/n_cold at the equator
    double uth_para = 0.2;     // hot parallel thermal momentum (c)
    double uth_perp = 0.4;     // hot perpendicular thermal momentum (c), equator
    int    ppc      = 0;       // hot markers per cell at the equator

    // boundaries
    bool   periodic = false;   // periodic uniform mode (requires a == 0)
    int    nd       = 512;     // damping cells on each side (bounded mode)
    double numax    = 1.0;     // peak damping rate ν_max (Ω_e0)

    bool   nonrel   = false;   // Newtonian push (benchmark vs nonrel theory)
    bool   deltaf   = false;   // nonlinear delta-f hot electrons (Tao PPCF 2017
                               // eq. 19): markers carry w = df/f, deposit df J

    // triggering antenna: external R-polarized current at the equator node,
    // J_ext = ant_amp * env(t) * (cos w0 t, sin w0 t), env = tanh ramp window
    double ant_amp   = 0.0;    // external current density amplitude (0 = off)
    double ant_w0    = 0.25;   // drive frequency (Omega_e0)
    double ant_ton   = 100.0, ant_toff = 600.0, ant_trise = 50.0;

    // init
    double seed_amp = 0.0;     // white-noise seed amplitude on By,Bz
    unsigned long long seed = 20260713ull;

    // diagnostics (0 stride = disabled)
    std::vector<double> probes;        // probe positions RELATIVE to equator
    int  probe_stride  = 0;
    int  frame_stride  = 0;
    int  frame_decim   = 4;
    int  energy_stride = 0;
    int  nphase        = 0;            // # of hot phase-space dumps over the run
    int  phase_decim   = 1024;         // keep every k-th marker in phase dumps
    std::string outdir = ".";
    std::string prefix = "chirp1d";
};

// ---- POD view passed by value into kernels ---------------------------------

struct H1DView {
    // fields / fluid (double)
    double *ey = nullptr, *ez = nullptr;     // nodes [NN]
    double *by = nullptr, *bz = nullptr;     // faces [nx]
    double *vcy = nullptr, *vcz = nullptr;   // nodes [NN]
    double *maskn = nullptr, *maskf = nullptr;
    // hot current deposit (float, zeroed each step)
    float  *jhy = nullptr, *jhz = nullptr;   // nodes [NN]
    // hot particles
    double *xp = nullptr;
    float  *ux = nullptr, *uy = nullptr, *uz = nullptr;
    float  *wd = nullptr;                    // delta-f weight w = df/f
    long    np = 0;

    int    nx = 0, NN = 0;
    int    per = 0;                          // periodic flag
    double dx = 0, dt = 0, c2 = 1.0;
    double a = 0, xc = 0;                    // B0 = 1 + a (x-xc)²
    double nc = 0;                           // cold density  (= wpe²)
    double w0j = 0;                          // macro weight / dx  (deposit coef)
    double xlo = 0, xhi = 0;                 // reflecting walls (bounded mode)
    double L = 0;
    int    nonrel = 0;
    int    deltaf = 0;
    double tpar = 0, tperp_eq = 0;           // hot temperatures (momentum units)
    int    ant_node = -1;                    // triggering antenna (node index)
    double ant_amp = 0, ant_w0 = 0, ant_ton = 0, ant_toff = 0, ant_trise = 0;
};

namespace h1d {

// ---- small device helpers --------------------------------------------------

__device__ __host__ inline unsigned long long splitmix64(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// k-th uniform (0,1] variate of a per-particle stream.
__device__ inline double rng_u01(unsigned long long seed, long p, int k) {
    unsigned long long s = splitmix64(seed ^ (0x100000001b3ull * (unsigned long long)(p + 1)));
    s = splitmix64(s + (unsigned long long)k * 0x9e3779b97f4a7c15ull);
    return ((double)(s >> 11) + 1.0) * (1.0 / 9007199254740992.0);  // (0,1]
}

__device__ inline double b0_of(const H1DView& v, double x) {
    const double xr = x - v.xc;
    return 1.0 + v.a * xr * xr;
}

// local Tperp of the (E,mu)-mapped equatorial bi-Maxwellian
__device__ inline double tperp_of(const H1DView& v, double b0) {
    return 1.0 / ((1.0 - 1.0 / b0) / v.tpar + (1.0 / b0) / v.tperp_eq);
}

// Boris velocity rotation about field B for charge q/m = -1, momentum u, dt.
__device__ inline void boris_rotate(double& ux, double& uy, double& uz,
                                    double Bx, double By, double Bz,
                                    double dt, double gam) {
    const double f  = -0.5 * dt / gam;       // q/m = -1
    const double tx = f * Bx, ty = f * By, tz = f * Bz;
    const double t2 = tx * tx + ty * ty + tz * tz;
    const double sx = 2.0 * tx / (1.0 + t2),
                 sy = 2.0 * ty / (1.0 + t2),
                 sz = 2.0 * tz / (1.0 + t2);
    const double px = ux + (uy * tz - uz * ty);
    const double py = uy + (uz * tx - ux * tz);
    const double pz = uz + (ux * ty - uy * tx);
    ux += py * sz - pz * sy;
    uy += pz * sx - px * sz;
    uz += px * sy - py * sx;
}

// ---- kernels ----------------------------------------------------------------

// Faraday half step: B += dt2 * curl terms.  Faces i+1/2 use nodes i, i+1.
static __global__ void k_faraday(H1DView v, double dt2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.nx) return;
    const int ip = v.per ? (i + 1) % v.NN : i + 1;
    const double dEy = v.ey[ip] - v.ey[i];
    const double dEz = v.ez[ip] - v.ez[i];
    v.by[i] += dt2 * ( dEz / v.dx);
    v.bz[i] += dt2 * (-dEy / v.dx);
}

// Ampère full step: E += dt * (c² curl B - J).  Node i uses faces i-1/2, i+1/2.
static __global__ void k_ampere(H1DView v, double t) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.NN) return;
    if (!v.per && (i == 0 || i == v.NN - 1)) {          // PEC behind absorber
        v.ey[i] = 0.0; v.ez[i] = 0.0;
        return;
    }
    const int fl = v.per ? (i - 1 + v.nx) % v.nx : i - 1;
    const int fr = v.per ? i % v.nx : i;
    const double dBy = v.by[fr] - v.by[fl];
    const double dBz = v.bz[fr] - v.bz[fl];
    double Jy  = (double)v.jhy[i] - v.nc * v.vcy[i];
    double Jz  = (double)v.jhz[i] - v.nc * v.vcz[i];
    if (i == v.ant_node && v.ant_amp != 0.0) {
        const double env = 0.5 * (tanh((t - v.ant_ton) / v.ant_trise)
                                - tanh((t - v.ant_toff) / v.ant_trise));
        Jy += v.ant_amp * env * cos(v.ant_w0 * t);
        Jz += v.ant_amp * env * sin(v.ant_w0 * t);
    }
    v.ey[i] += v.dt * (-v.c2 * dBz / v.dx - Jy);
    v.ez[i] += v.dt * ( v.c2 * dBy / v.dx - Jz);
}

// Cold fluid: half E kick, exact gyro-rotation e^{+i B0 dt}, half E kick.
// (q/m = -1:  d(Vy+iVz)/dt = +i B0 (Vy+iVz) - (Ey+iEz).)
static __global__ void k_cold(H1DView v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.NN) return;
    const double Ey = v.ey[i], Ez = v.ez[i];
    double vy = v.vcy[i] - 0.5 * v.dt * Ey;
    double vz = v.vcz[i] - 0.5 * v.dt * Ez;
    const double th = b0_of(v, i * v.dx) * v.dt;
    const double c = cos(th), s = sin(th);
    const double vy2 = vy * c - vz * s;
    const double vz2 = vy * s + vz * c;
    v.vcy[i] = vy2 - 0.5 * v.dt * Ey;
    v.vcz[i] = vz2 - 0.5 * v.dt * Ez;
}

// Fused hot-electron push (rel. Boris + mirror effective field) + move +
// CIC current deposit at the midpoint position with v^{n+1/2}.
static __global__ void k_push_deposit(H1DView v) {
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= v.np) return;

    const double x = v.xp[p];

    // gather E at nodes
    const double xn = x / v.dx;
    int j = (int)floor(xn);
    double f = xn - j;
    int jp = j + 1;
    if (v.per) { j = ((j % v.NN) + v.NN) % v.NN; jp = (j + 1) % v.NN; }
    else       { if (j < 0) { j = 0; f = 0; } if (j > v.NN - 2) { j = v.NN - 2; f = 1; } jp = j + 1; }
    const double Ey = v.ey[j] * (1.0 - f) + v.ey[jp] * f;
    const double Ez = v.ez[j] * (1.0 - f) + v.ez[jp] * f;

    // gather δB at faces (offset half cell)
    const double xf = xn - 0.5;
    int jf = (int)floor(xf);
    double g = xf - jf;
    int jfp = jf + 1;
    if (v.per) { jf = ((jf % v.nx) + v.nx) % v.nx; jfp = (jf + 1) % v.nx; }
    else       { if (jf < 0) { jf = 0; g = 0; } if (jf > v.nx - 2) { jf = v.nx - 2; g = 1; } jfp = jf + 1; }
    const double By = v.by[jf] * (1.0 - g) + v.by[jfp] * g;
    const double Bz = v.bz[jf] * (1.0 - g) + v.bz[jfp] * g;

    const double b0  = b0_of(v, x);
    const double db0 = 2.0 * v.a * (x - v.xc);

    double Ux = v.ux[p], Uy = v.uy[p], Uz = v.uz[p];

    // half E kick (q = -1, E = (0,Ey,Ez))
    Uy -= 0.5 * v.dt * Ey;
    Uz -= 0.5 * v.dt * Ez;

    const double gam = v.nonrel ? 1.0 : sqrt(1.0 + Ux * Ux + Uy * Uy + Uz * Uz);

    // rotation: B_total = (B0, By, Bz) + B_mir,  B_mir = -(B0'/2B0)(u × x̂)
    const double m  = db0 / (2.0 * b0);
    const double Bxt = b0;
    const double Byt = By - m * Uz;
    const double Bzt = Bz + m * Uy;
    boris_rotate(Ux, Uy, Uz, Bxt, Byt, Bzt, v.dt, gam);

    // half E kick
    Uy -= 0.5 * v.dt * Ey;
    Uz -= 0.5 * v.dt * Ez;

    const double gam2 = v.nonrel ? 1.0 : sqrt(1.0 + Ux * Ux + Uy * Uy + Uz * Uz);
    const double vx = Ux / gam2;

    // nonlinear delta-f weight update (Tao et al. PPCF 2017 eq. 19):
    //   dw/dt = -(1-w) (q/m)(dE + v x dB) . d(ln f0)/du,   q/m = -1,
    //   d(ln f0)/du = (-u_par/T_par, -u_y/Tperp(x), -u_z/Tperp(x)).
    double wgt = 1.0;
    if (v.deltaf) {
        const double vy = Uy / gam2, vz = Uz / gam2;
        const double Tp = tperp_of(v, b0);
        const double Fx = vy * Bz - vz * By;             // (v x dB)_x
        const double Fy = Ey - vx * Bz;
        const double Fz = Ez + vx * By;
        // dw/dt = -(1-w)(q/m) F.dlnf0/du = (1-w) F.dlnf0/du   (q/m = -1)
        //       = -(1-w) [Fx Ux/Tpar + (Fy Uy + Fz Uz)/Tperp]
        const double D  = -(Fx * Ux / v.tpar + (Fy * Uy + Fz * Uz) / Tp);
        double w = v.wd[p];
        w += v.dt * (1.0 - w) * D;
        v.wd[p] = (float)w;
        wgt = w;
    }

    // deposit J at midpoint with v^{n+1/2}  (J = q n v, q = -1; delta-f: w dJ)
    {
        const double xm = (x + 0.5 * vx * v.dt) / v.dx;
        int jd = (int)floor(xm);
        double fd = xm - jd;
        int jdp = jd + 1;
        if (v.per) { jd = ((jd % v.NN) + v.NN) % v.NN; jdp = (jd + 1) % v.NN; }
        else       { if (jd < 0) { jd = 0; fd = 0; } if (jd > v.NN - 2) { jd = v.NN - 2; fd = 1; } jdp = jd + 1; }
        const double vy = Uy / gam2, vz = Uz / gam2;
        const float cy = (float)(-v.w0j * wgt * vy), cz = (float)(-v.w0j * wgt * vz);
        atomicAdd(&v.jhy[jd],  cy * (float)(1.0 - fd));
        atomicAdd(&v.jhy[jdp], cy * (float)fd);
        atomicAdd(&v.jhz[jd],  cz * (float)(1.0 - fd));
        atomicAdd(&v.jhz[jdp], cz * (float)fd);
    }

    // move + boundary
    double xnew = x + vx * v.dt;
    if (v.per) {
        if (xnew < 0)    xnew += v.L;
        if (xnew >= v.L) xnew -= v.L;
    } else {
        if (xnew < v.xlo) { xnew = 2.0 * v.xlo - xnew; Ux = -Ux; }
        if (xnew > v.xhi) { xnew = 2.0 * v.xhi - xnew; Ux = -Ux; }
    }

    v.xp[p] = xnew;
    v.ux[p] = (float)Ux; v.uy[p] = (float)Uy; v.uz[p] = (float)Uz;
}

// Rotate u backward half a step in B0 + mirror field only (leapfrog init).
static __global__ void k_half_back(H1DView v) {
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= v.np) return;
    const double x = v.xp[p];
    const double b0  = b0_of(v, x);
    const double db0 = 2.0 * v.a * (x - v.xc);
    double Ux = v.ux[p], Uy = v.uy[p], Uz = v.uz[p];
    const double gam = v.nonrel ? 1.0 : sqrt(1.0 + Ux * Ux + Uy * Uy + Uz * Uz);
    const double m = db0 / (2.0 * b0);
    boris_rotate(Ux, Uy, Uz, b0, -m * Uz, m * Uy, -0.5 * v.dt, gam);
    v.ux[p] = (float)Ux; v.uy[p] = (float)Uy; v.uz[p] = (float)Uz;
}

// Multiplicative damping masks on wave fields + cold fluid.
static __global__ void k_mask(H1DView v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v.NN) {
        const double mn = v.maskn[i];
        if (mn != 1.0) {
            v.ey[i] *= mn; v.ez[i] *= mn;
            v.vcy[i] *= mn; v.vcz[i] *= mn;
        }
    }
    if (i < v.nx) {
        const double mf = v.maskf[i];
        if (mf != 1.0) { v.by[i] *= mf; v.bz[i] *= mf; }
    }
}

// Hot-particle loader. Cell of marker p found by binary search in offsets.
static __global__ void k_load(H1DView v, const int* __restrict__ off,
                              unsigned long long seed,
                              double uth_para, double uth_perp_eq) {
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= v.np) return;
    // binary search: largest c with off[c] <= p
    int lo = 0, hi = v.nx;                       // off has nx+1 entries
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (off[mid] <= p) lo = mid; else hi = mid;
    }
    const int c = lo;
    const double x = (c + rng_u01(seed, p, 0) * 0.9999999) * v.dx;

    // local T⊥(x) from the (E,μ)-mapped equatorial bi-Maxwellian
    const double b   = b0_of(v, x);
    const double Tpa = uth_para * uth_para;
    const double Tpe = uth_perp_eq * uth_perp_eq;
    const double Tp  = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);

    // u∥ ~ N(0, uth_para): Box-Muller
    const double r1 = rng_u01(seed, p, 1), r2 = rng_u01(seed, p, 2);
    const double upar = uth_para * sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);
    // u⊥ ~ Rayleigh(√Tp), uniform gyrophase
    const double r3 = rng_u01(seed, p, 3), r4 = rng_u01(seed, p, 4);
    const double uperp = sqrt(Tp) * sqrt(-2.0 * log(r3));
    v.xp[p] = x;
    if (v.deltaf) v.wd[p] = 0.0f;
    v.ux[p] = (float)upar;
    v.uy[p] = (float)(uperp * cos(2.0 * M_PI * r4));
    v.uz[p] = (float)(uperp * sin(2.0 * M_PI * r4));
}

// ---- diagnostics kernels ----------------------------------------------------

// out: [0]=WE, [1]=WB, [2]=Wcold, [3] unused; bmax2: max |δB⊥|² (float bits)
static __global__ void k_energy_fields(H1DView v, double* out, int* bmax2) {
    __shared__ double sE, sB, sC;
    if (threadIdx.x == 0) { sE = 0; sB = 0; sC = 0; }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0, b = 0, c = 0;
    float m2 = 0;
    if (i < v.NN) {
        e = 0.5 * (v.ey[i] * v.ey[i] + v.ez[i] * v.ez[i]) * v.dx;
        c = 0.5 * v.nc * (v.vcy[i] * v.vcy[i] + v.vcz[i] * v.vcz[i]) * v.dx;
    }
    if (i < v.nx) {
        const double bb = v.by[i] * v.by[i] + v.bz[i] * v.bz[i];
        b = 0.5 * v.c2 * bb * v.dx;
        m2 = (float)bb;
    }
    atomicAdd(&sE, e); atomicAdd(&sB, b); atomicAdd(&sC, c);
    atomicMax(bmax2, __float_as_int(m2));       // valid for non-negative floats
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&out[0], sE); atomicAdd(&out[1], sB); atomicAdd(&out[2], sC);
    }
}

// out[3] += hot kinetic energy Σ w0 wgt (γ-1); out[4] += Σ wd²  (delta-f)
static __global__ void k_energy_hot(H1DView v, double* out) {
    __shared__ double sK, sW;
    if (threadIdx.x == 0) { sK = 0; sW = 0; }
    __syncthreads();
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    double k = 0, w2 = 0;
    if (p < v.np) {
        const double ux = v.ux[p], uy = v.uy[p], uz = v.uz[p];
        const double u2 = ux * ux + uy * uy + uz * uz;
        const double wgt = v.deltaf ? (double)v.wd[p] : 1.0;
        k = (v.nonrel ? 0.5 * u2 : sqrt(1.0 + u2) - 1.0) * v.w0j * v.dx * wgt;
        if (v.deltaf) w2 = wgt * wgt;
    }
    atomicAdd(&sK, k); atomicAdd(&sW, w2);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(&out[3], sK); atomicAdd(&out[4], sW); }
}

static __global__ void k_probe_gather(H1DView v, const int* __restrict__ idx,
                                      int nprobe, float* buf, int slot) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nprobe) return;
    const int i = idx[k];
    float* o = buf + ((long)slot * nprobe + k) * 4;
    o[0] = (float)v.by[i];
    o[1] = (float)v.bz[i];
    o[2] = (float)v.ey[i];
    o[3] = (float)v.ez[i];
}

} // namespace h1d

// ---- host-side simulation class ---------------------------------------------

class Hybrid1D {
public:
    static constexpr int PROBE_CHUNK = 4096;

    explicit Hybrid1D(const Chirp1DParams& P) : P_(P) {
        if (P_.periodic && P_.a != 0.0)
            throw std::runtime_error("hybrid1d: periodic mode requires a = 0");
        if (!P_.periodic && 2 * P_.nd >= P_.nx)
            throw std::runtime_error("hybrid1d: damping regions cover the whole domain");
        if (P_.dt >= P_.dx)  // c = 1
            throw std::runtime_error("hybrid1d: CFL violated (need c dt < dx)");

        NN_ = P_.periodic ? P_.nx : P_.nx + 1;
        L_  = P_.nx * P_.dx;
        xc_ = 0.5 * L_;

        ey_ = DeviceArray<double>(NN_); ez_ = DeviceArray<double>(NN_);
        vcy_ = DeviceArray<double>(NN_); vcz_ = DeviceArray<double>(NN_);
        by_ = DeviceArray<double>(P_.nx); bz_ = DeviceArray<double>(P_.nx);
        jhy_ = DeviceArray<float>(NN_); jhz_ = DeviceArray<float>(NN_);
        maskn_ = DeviceArray<double>(NN_); maskf_ = DeviceArray<double>(P_.nx);
        diag_ = DeviceArray<double>(5); bmax_ = DeviceArray<int>(1);

        ey_.zero(); ez_.zero(); vcy_.zero(); vcz_.zero();
        by_.zero(); bz_.zero(); jhy_.zero(); jhz_.zero();

        build_masks();
        seed_fields();
        load_particles();
    }

    // one full leapfrog step
    void step() {
        const int TB = 256;
        H1DView v = view();
        const double tn = (nstep_done_ + 0.5) * P_.dt;   // E-update time level
        h1d::k_faraday<<<gN(P_.nx, TB), TB>>>(v, 0.5 * P_.dt);
        if (np_ > 0) {
            CUDA_CHECK(cudaMemsetAsync(jhy_.data(), 0, jhy_.bytes()));
            CUDA_CHECK(cudaMemsetAsync(jhz_.data(), 0, jhz_.bytes()));
            h1d::k_push_deposit<<<gN(np_, TB), TB>>>(v);
        }
        h1d::k_cold<<<gN(NN_, TB), TB>>>(v);
        h1d::k_faraday<<<gN(P_.nx, TB), TB>>>(v, 0.5 * P_.dt);
        h1d::k_ampere<<<gN(NN_, TB), TB>>>(v, tn);
        if (!P_.periodic)
            h1d::k_mask<<<gN(std::max(NN_, P_.nx), TB), TB>>>(v);
        ++nstep_done_;
    }

    // energies: WE, WB, Wcold, Whot, max|δB⊥|
    struct Energies { double we, wb, wc, wh, bmax, wd2; };
    Energies energies() {
        const int TB = 256;
        CUDA_CHECK(cudaMemsetAsync(diag_.data(), 0, diag_.bytes()));
        CUDA_CHECK(cudaMemsetAsync(bmax_.data(), 0, bmax_.bytes()));
        H1DView v = view();
        h1d::k_energy_fields<<<gN(std::max(NN_, P_.nx), TB), TB>>>(v, diag_.data(), bmax_.data());
        if (np_ > 0) h1d::k_energy_hot<<<gN(np_, TB), TB>>>(v, diag_.data());
        double h[5]; int bi;
        CUDA_CHECK(cudaMemcpy(h, diag_.data(), sizeof(h), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&bi, bmax_.data(), sizeof(bi), cudaMemcpyDeviceToHost));
        float b2; std::memcpy(&b2, &bi, sizeof(b2));
        return {h[0], h[1], h[2], h[3], std::sqrt((double)b2), h[4]};
    }

    // replace the hot population with explicit test particles (w0 unchanged)
    void set_particles(const std::vector<double>& x, const std::vector<float>& ux,
                       const std::vector<float>& uy, const std::vector<float>& uz) {
        np_ = (long)x.size();
        xp_ = DeviceArray<double>(np_); ux_ = DeviceArray<float>(np_);
        uy_ = DeviceArray<float>(np_); uz_ = DeviceArray<float>(np_);
        if (P_.deltaf) { wd_ = DeviceArray<float>(np_); wd_.zero(); }
        CUDA_CHECK(cudaMemcpy(xp_.data(), x.data(),  np_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ux_.data(), ux.data(), np_ * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(uy_.data(), uy.data(), np_ * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(uz_.data(), uz.data(), np_ * sizeof(float),  cudaMemcpyHostToDevice));
    }
    void get_particles(std::vector<double>& x, std::vector<float>& ux,
                       std::vector<float>& uy, std::vector<float>& uz) {
        x.resize(np_); ux.resize(np_); uy.resize(np_); uz.resize(np_);
        if (np_ == 0) return;
        CUDA_CHECK(cudaMemcpy(x.data(),  xp_.data(), np_ * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), ux_.data(), np_ * sizeof(float),  cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), uy_.data(), np_ * sizeof(float),  cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), uz_.data(), np_ * sizeof(float),  cudaMemcpyDeviceToHost));
    }
    void half_step_back() {
        if (np_ > 0) h1d::k_half_back<<<gN(np_, 256), 256>>>(view());
    }

    // direct field access for tests (host copies)
    void set_fields(const std::vector<double>& ey, const std::vector<double>& ez,
                    const std::vector<double>& by, const std::vector<double>& bz,
                    const std::vector<double>& vcy, const std::vector<double>& vcz) {
        CUDA_CHECK(cudaMemcpy(ey_.data(), ey.data(), NN_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ez_.data(), ez.data(), NN_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(by_.data(), by.data(), P_.nx * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bz_.data(), bz.data(), P_.nx * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vcy_.data(), vcy.data(), NN_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vcz_.data(), vcz.data(), NN_ * sizeof(double), cudaMemcpyHostToDevice));
    }
    void get_b(std::vector<double>& by, std::vector<double>& bz) {
        by.resize(P_.nx); bz.resize(P_.nx);
        CUDA_CHECK(cudaMemcpy(by.data(), by_.data(), P_.nx * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bz.data(), bz_.data(), P_.nx * sizeof(double), cudaMemcpyDeviceToHost));
    }

    long   np()  const { return np_; }
    double w0()  const { return w0_; }
    double L()   const { return L_; }
    double xc()  const { return xc_; }
    int    NN()  const { return NN_; }
    const Chirp1DParams& params() const { return P_; }

    // ---- full production run with file diagnostics --------------------------
    void run(bool verbose = true) {
        open_outputs();
        const long n_energy = P_.energy_stride > 0 ? P_.energy_stride : 0;
        const long phase_every = (P_.nphase > 1) ? (P_.nsteps / (P_.nphase - 1)) : 0;

        if (P_.nphase > 0) dump_phase(0);
        for (long n = 0; n < P_.nsteps; ++n) {
            step();
            const long n1 = n + 1;
            if (P_.probe_stride > 0 && n1 % P_.probe_stride == 0) sample_probes();
            if (P_.frame_stride > 0 && n1 % P_.frame_stride == 0) dump_frame();
            if (n_energy > 0 && n1 % n_energy == 0) {
                auto e = energies();
                std::fprintf(energy_f_, "%.6f,%.9e,%.9e,%.9e,%.9e,%.6e,%.6e\n",
                             n1 * P_.dt, e.we, e.wb, e.wc, e.wh, e.bmax, e.wd2);
                if (verbose && n1 % (n_energy * 20) == 0) {
                    std::printf("t=%9.1f  WB=%.3e  Whot=%.6e  max|dB|/B0=%.4f\n",
                                n1 * P_.dt, e.wb, e.wh, e.bmax);
                    std::fflush(stdout);
                    std::fflush(energy_f_);
                }
            }
            if (phase_every > 0 && n1 % phase_every == 0) dump_phase(n1);
        }
        flush_probes();
        close_outputs();
    }

private:
    static int gN(long n, int tb) { return (int)((n + tb - 1) / tb); }

    H1DView view() {
        H1DView v;
        v.ey = ey_.data(); v.ez = ez_.data();
        v.by = by_.data(); v.bz = bz_.data();
        v.vcy = vcy_.data(); v.vcz = vcz_.data();
        v.maskn = maskn_.data(); v.maskf = maskf_.data();
        v.jhy = jhy_.data(); v.jhz = jhz_.data();
        if (np_ > 0) {
            v.xp = xp_.data(); v.ux = ux_.data(); v.uy = uy_.data(); v.uz = uz_.data();
            if (P_.deltaf) v.wd = wd_.data();
        }
        v.np = np_;
        v.nx = P_.nx; v.NN = NN_; v.per = P_.periodic ? 1 : 0;
        v.dx = P_.dx; v.dt = P_.dt; v.c2 = 1.0;
        v.a = P_.a; v.xc = xc_;
        v.nc = P_.wpe * P_.wpe;
        v.w0j = w0_ / P_.dx;
        v.xlo = P_.periodic ? 0.0 : P_.nd * P_.dx;
        v.xhi = P_.periodic ? L_ : (P_.nx - P_.nd) * P_.dx;
        v.L = L_;
        v.nonrel = P_.nonrel ? 1 : 0;
        v.deltaf = P_.deltaf ? 1 : 0;
        v.tpar = P_.uth_para * P_.uth_para;
        v.tperp_eq = P_.uth_perp * P_.uth_perp;
        if (P_.ant_amp != 0.0) {
            v.ant_node = (int)std::lround(xc_ / P_.dx);
            v.ant_amp = P_.ant_amp; v.ant_w0 = P_.ant_w0;
            v.ant_ton = P_.ant_ton; v.ant_toff = P_.ant_toff;
            v.ant_trise = P_.ant_trise;
        }
        return v;
    }

    void build_masks() {
        std::vector<double> mn(NN_, 1.0), mf(P_.nx, 1.0);
        if (!P_.periodic) {
            auto mval = [&](double xi) {  // xi = position in cells
                double d = 0.0;
                if (xi < P_.nd)              d = (P_.nd - xi) / P_.nd;
                else if (xi > P_.nx - P_.nd) d = (xi - (P_.nx - P_.nd)) / P_.nd;
                if (d <= 0.0) return 1.0;
                return std::exp(-P_.numax * d * d * P_.dt);
            };
            for (int i = 0; i < NN_; ++i)   mn[i] = mval(i);
            for (int i = 0; i < P_.nx; ++i) mf[i] = mval(i + 0.5);
        }
        CUDA_CHECK(cudaMemcpy(maskn_.data(), mn.data(), NN_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(maskf_.data(), mf.data(), P_.nx * sizeof(double), cudaMemcpyHostToDevice));
    }

    void seed_fields() {
        if (P_.seed_amp <= 0.0) return;
        std::vector<double> by(P_.nx), bz(P_.nx);
        unsigned long long s = P_.seed ^ 0xabcdef1234567890ull;
        for (int i = 0; i < P_.nx; ++i) {
            s = h1d::splitmix64(s);
            by[i] = P_.seed_amp * (2.0 * ((s >> 11) * (1.0 / 9007199254740992.0)) - 1.0);
            s = h1d::splitmix64(s);
            bz[i] = P_.seed_amp * (2.0 * ((s >> 11) * (1.0 / 9007199254740992.0)) - 1.0);
        }
        CUDA_CHECK(cudaMemcpy(by_.data(), by.data(), P_.nx * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bz_.data(), bz.data(), P_.nx * sizeof(double), cudaMemcpyHostToDevice));
    }

    void load_particles() {
        np_ = 0;
        if (P_.ppc <= 0 || P_.nh <= 0.0) return;
        // per-cell counts ∝ n_h(x) = n_h,eq · T⊥(x)/T⊥eq  (equal-weight markers)
        const double Tpa = P_.uth_para * P_.uth_para;
        const double Tpe = P_.uth_perp * P_.uth_perp;
        std::vector<int> off(P_.nx + 1, 0);
        long tot = 0;
        for (int c = 0; c < P_.nx; ++c) {
            const double x = (c + 0.5) * P_.dx;
            const double xr = x - xc_;
            const double b = 1.0 + P_.a * xr * xr;
            const double Tp = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);
            const int cnt = (int)std::lround(P_.ppc * (Tp / Tpe));
            off[c] = (int)tot;
            tot += cnt;
        }
        off[P_.nx] = (int)tot;
        // rebuild as proper prefix (off[c] currently = start; last set above)
        np_ = tot;
        const double nheq = P_.nh * P_.wpe * P_.wpe;
        w0_ = nheq * P_.dx / P_.ppc;

        xp_ = DeviceArray<double>(np_); ux_ = DeviceArray<float>(np_);
        uy_ = DeviceArray<float>(np_); uz_ = DeviceArray<float>(np_);
        if (P_.deltaf) { wd_ = DeviceArray<float>(np_); wd_.zero(); }
        DeviceArray<int> doff(P_.nx + 1);
        CUDA_CHECK(cudaMemcpy(doff.data(), off.data(), (P_.nx + 1) * sizeof(int), cudaMemcpyHostToDevice));
        h1d::k_load<<<gN(np_, 256), 256>>>(view(), doff.data(), P_.seed,
                                           P_.uth_para, P_.uth_perp);
        CUDA_CHECK(cudaDeviceSynchronize());
        half_step_back();
    }

    // ---- outputs -------------------------------------------------------------
    std::string path(const std::string& suffix) const {
        return P_.outdir + "/" + P_.prefix + suffix;
    }

    void open_outputs() {
        if (P_.probe_stride > 0 && !P_.probes.empty()) {
            nprobe_ = (int)P_.probes.size();
            std::vector<int> idx(nprobe_);
            for (int k = 0; k < nprobe_; ++k) {
                int i = (int)std::lround((P_.probes[k] + xc_) / P_.dx);
                if (i < 0) i = 0;
                if (i > P_.nx - 1) i = P_.nx - 1;
                idx[k] = i;
            }
            pidx_ = DeviceArray<int>(nprobe_);
            CUDA_CHECK(cudaMemcpy(pidx_.data(), idx.data(), nprobe_ * sizeof(int), cudaMemcpyHostToDevice));
            pbuf_ = DeviceArray<float>((size_t)PROBE_CHUNK * nprobe_ * 4);
            pslot_ = 0;
            probe_f_ = std::fopen(path("_probes.bin").c_str(), "wb");
        }
        if (P_.frame_stride > 0)
            frame_f_ = std::fopen(path("_frames.bin").c_str(), "wb");
        if (P_.energy_stride > 0) {
            energy_f_ = std::fopen(path("_energy.csv").c_str(), "w");
            std::fprintf(energy_f_, "t,WE,WB,Wcold,Whot,bmax,wd2\n");
        }
        if (P_.nphase > 0)
            phase_f_ = std::fopen(path("_phase.bin").c_str(), "wb");
        write_meta(0, 0);
    }

    void sample_probes() {
        h1d::k_probe_gather<<<1, 64>>>(view(), pidx_.data(), nprobe_, pbuf_.data(), pslot_);
        if (++pslot_ == PROBE_CHUNK) flush_probes();
        ++nprobe_samples_;
    }
    void flush_probes() {
        if (!probe_f_ || pslot_ == 0) return;
        std::vector<float> h((size_t)pslot_ * nprobe_ * 4);
        CUDA_CHECK(cudaMemcpy(h.data(), pbuf_.data(), h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::fwrite(h.data(), sizeof(float), h.size(), probe_f_);
        pslot_ = 0;
    }

    void dump_frame() {
        std::vector<double> by(P_.nx), bz(P_.nx);
        get_b(by, bz);
        const int nout = P_.nx / P_.frame_decim;
        std::vector<float> h(2 * nout);
        for (int i = 0; i < nout; ++i) {
            h[i]        = (float)by[(size_t)i * P_.frame_decim];
            h[nout + i] = (float)bz[(size_t)i * P_.frame_decim];
        }
        std::fwrite(h.data(), sizeof(float), h.size(), frame_f_);
        ++nframes_;
    }

    void dump_phase(long nstep) {
        if (!phase_f_ || np_ == 0) return;
        std::vector<double> x; std::vector<float> ux, uy, uz;
        get_particles(x, ux, uy, uz);
        std::vector<float> w;
        if (P_.deltaf) {
            w.resize(np_);
            CUDA_CHECK(cudaMemcpy(w.data(), wd_.data(), np_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        const int rec = P_.deltaf ? 5 : 4;
        std::vector<float> h;
        h.reserve(rec * (x.size() / P_.phase_decim + 1));
        for (size_t p = 0; p < x.size(); p += P_.phase_decim) {
            h.push_back((float)(x[p] - xc_));
            h.push_back(ux[p]); h.push_back(uy[p]); h.push_back(uz[p]);
            if (P_.deltaf) h.push_back(w[p]);
        }
        const float hdr[2] = {(float)(nstep * P_.dt), (float)(h.size() / rec)};
        std::fwrite(hdr, sizeof(float), 2, phase_f_);
        std::fwrite(h.data(), sizeof(float), h.size(), phase_f_);
        ++nphase_done_;
    }

    void write_meta(int, int) {
        std::ofstream m(path("_meta.txt"));
        m << "nx=" << P_.nx << "\ndx=" << P_.dx << "\ndt=" << P_.dt
          << "\nnsteps=" << P_.nsteps << "\nwpe=" << P_.wpe << "\na=" << P_.a
          << "\nnh=" << P_.nh << "\nuth_para=" << P_.uth_para
          << "\nuth_perp=" << P_.uth_perp << "\nppc=" << P_.ppc
          << "\nperiodic=" << (P_.periodic ? 1 : 0) << "\nnd=" << P_.nd
          << "\nnumax=" << P_.numax << "\nseed_amp=" << P_.seed_amp
          << "\nnp=" << np_ << "\nw0=" << w0_ << "\nL=" << L_ << "\nxc=" << xc_
          << "\nprobe_stride=" << P_.probe_stride
          << "\nframe_stride=" << P_.frame_stride
          << "\nframe_decim=" << P_.frame_decim
          << "\nenergy_stride=" << P_.energy_stride
          << "\nnphase=" << P_.nphase << "\nphase_decim=" << P_.phase_decim
          << "\nnprobe=" << nprobe_
          << "\ndeltaf=" << (P_.deltaf ? 1 : 0)
          << "\nant_amp=" << P_.ant_amp << "\nant_w0=" << P_.ant_w0
          << "\nphase_rec=" << (P_.deltaf ? 5 : 4) << "\n";
        m << "probes=";
        for (size_t k = 0; k < P_.probes.size(); ++k)
            m << (k ? "," : "") << P_.probes[k];
        m << "\n";
    }

    void close_outputs() {
        // final meta rewrite with actual counts
        {
            std::ofstream m(path("_meta.txt"), std::ios::app);
            m << "nprobe_samples=" << nprobe_samples_ << "\nnframes=" << nframes_
              << "\nnphase_done=" << nphase_done_ << "\n";
        }
        if (probe_f_) std::fclose(probe_f_);
        if (frame_f_) std::fclose(frame_f_);
        if (energy_f_) std::fclose(energy_f_);
        if (phase_f_) std::fclose(phase_f_);
        probe_f_ = frame_f_ = energy_f_ = phase_f_ = nullptr;
    }

    Chirp1DParams P_;
    int NN_ = 0;
    double L_ = 0, xc_ = 0, w0_ = 0;
    long np_ = 0;

    DeviceArray<double> ey_, ez_, by_, bz_, vcy_, vcz_, maskn_, maskf_;
    DeviceArray<float>  jhy_, jhz_;
    DeviceArray<double> xp_;
    DeviceArray<float>  ux_, uy_, uz_, wd_;
    DeviceArray<double> diag_;
    DeviceArray<int>    bmax_, pidx_;
    DeviceArray<float>  pbuf_;

    long nstep_done_ = 0;
    int nprobe_ = 0, pslot_ = 0;
    long nprobe_samples_ = 0, nframes_ = 0, nphase_done_ = 0;
    std::FILE *probe_f_ = nullptr, *frame_f_ = nullptr, *energy_f_ = nullptr,
              *phase_f_ = nullptr;
};

} // namespace arc

#endif // ARC_PIC_HYBRID1D_HPP
