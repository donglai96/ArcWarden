// ArcWarden — RSM (Reduced Spectral Mirror): the m = ±1 oblique harmonic.
// Model constitution: docs/RSM_MODEL_DEFINITION.md (v1.0 + amendments).
//
// Fields:   F(x, y, t) = F0(x, t) + 2 Re[ F1(x, t) e^{i k1 y} ]
// with F1 a complex 1D array on the x-nodes and ∂y → i·k1 EXACT (no y grid,
// no y-grid dispersion). m = −1 is the reality conjugate of m = +1, NOT an
// independent counter-propagating mode — propagation direction lives in the
// x-dependence (sign of k_par).
//
// Phase convention (the load-bearing trick): at ny = 1 the particle y is
// already a pure phase coordinate — the legacy pusher advances it by
// v_y·dt/Δy and wraps it mod 1, and the m = 0 deposit/gather are invariant
// to its value (every y-row is the same row). With the deck contract
//     Ly = 2π/k1
// the cell-unit y IS θ/2π with θ = k1·y_phys:  ẏ_cell = v_y/Δy = k1·v_y/2π
// ≡ θ̇/2π, and the periodic y-wrap is exactly phase-periodic. So the phase
// state needs NO new particle array, no sort/migrate/checkpoint plumbing,
// and no pusher change — sort_by_tile and checkpoint carry it for free.
//
// R1 phase trap (constitution §6.1): every ny = 1 loader pins y = 0.5
// (particles.hpp mirror loader; quiet-start loaders are sub-cell coherent),
// so a direct e^{−iθ} deposit off the load is COHERENT — |ρ1| ~ N, a fake
// oblique seed at macroscopic amplitude instead of N^{−1/2} shot noise.
// rsm_theta_init randomizes θ uniformly AFTER the load, on a SALTED seed so
// no loader stream can ever alias it. (2026-07-31 bug: it used stream 11 on
// the bare seed; the conecut loader's trial-0 draw rb — stream 11 — sets
// u∥ ∝ cos(2π·rb), so θ ≡ 2π·rb gave ⟨u∥e^{−iθ}⟩ = |u∥|/2: a macroscopic
// coherent J1x at half the thermal flux, the R1 trap resurrected by stream
// collision. Caught by the x4_rsm_smoke WB 5× anomaly vs rsm-off control.)
// Gate V1 (tests/test_rsm_phaseload.cu) demonstrates the trap and verifies
// the remedy: E|Σ e^{−iθ}|² = N.
//
// Factor-2 ledger (R6): arrays store the COEFFICIENT F1. Physical field
// = 2 Re[F1 e^{iθ}]; Ampère for m = 1 uses J1 with no 2; energy/work use
// W = W0 + 2·W1, P = J0·E0 + 2 Re(J1*·E1).
//
// Feature isolation: nothing here is launched or allocated unless
// RunParams::rsm = 1 — legacy path untouched by construction; every commit
// gated by scripts/regress_case2.py on both deposit paths.

#ifndef ARC_PIC_RSM_OBLIQUE_HPP
#define ARC_PIC_RSM_OBLIQUE_HPP

#include "pic/background_b0.hpp"
#include "pic/config.hpp"
#include "pic/device_array.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/yee2d.hpp"

#include <cmath>
#include <stdexcept>

namespace arc {

// POD handle pack for kernels (complex arrays as float2 = re, im).
//
// Staggering: x-staggered exactly like the legacy Yee lattice — E1x, B1y,
// B1z at i+½; E1y, E1z, B1x at the nodes i — but COLLOCATED in y (y is
// spectral; inheriting the Yee y half-shift would be a phase factor
// e^{i k1 dy/2} = e^{iπ} = −1 at ny = 1, a sign catastrophe). The exact
// ik1 couplings pair only co-located components, so the curls close on
// this lattice with standard staggered Dx and zero y-phase error.
struct RsmViews {
    DeviceView<float2> e1x, e1y, e1z;   // m=1 E coefficient
    DeviceView<float2> b1x, b1y, b1z;   // m=1 B coefficient
    DeviceView<float2> j1x, j1y, j1z;   // m=1 current coefficient
    DeviceView<float2> rho1;            // m=1 charge coefficient
    DeviceView<float2> vc1x, vc1y, vc1z; // m=1 cold-fluid velocity (nodes)
    int   nx  = 0;
    float dxp = 0.f, dyp = 0.f;         // physical spacings (dyp = Ly = 2π/k1)
    float k1  = 0.f;

    __host__ __device__ int wrap(int i) const {
        return i + (i < 0) * nx - (i >= nx) * nx;
    }
};

namespace detail {

// R1 remedy: uniform θ on [0, 1) in cell units (= [0, 2π) in phase), its own
// hash stream, independent of the loader's position/velocity streams. cell is
// untouched — at ny = 1, floor(y) = 0 for every y in [0, 1).
static __global__ void rsm_theta_init_kernel(ParticleViews p, unsigned long seed) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;
    // salted seed: loaders draw on the bare seed, so no (t, stream) tuple
    // here can reproduce a loader draw — θ independent by construction
    p.y[t] = (float)rng_uniform(t, 11, seed ^ 0x9E3779B97F4A7C15ULL);
}

// m = 1 moment deposit (diagnostic / V1 form — the dynamical charge-conserving
// worldline-Esirkepov J1 lands with the m = 1 field update):
//   ρ1(x) = Σ_p qw S(x−x_p) e^{−iθ_p},   J1(x) = Σ_p qw v_p S(x−x_p) e^{−iθ_p}
// with θ_p = 2π·y_p, CIC in x on the nodes, 1/(Δx·Δy) volume normalization
// matching the m = 0 deposit conventions.
static __global__ void rsm_deposit_moments_kernel(ParticleViews p, RsmViews r,
                                                  RunParams rp) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    const float x  = p.x[t];
    int i0 = (int)floorf(x);
    const float fx = x - (float)i0;
    if (i0 >= r.nx) i0 -= r.nx;                    // x == nx float edge
    const int i1 = (i0 + 1 == r.nx) ? 0 : i0 + 1;

    float sth, cth;                                 // e^{−iθ} = cth + i·sth
    __sincosf(-6.283185307179586f * p.y[t], &sth, &cth);

    const float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float gni = rp.rel ? rsqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;

    float qw = (float)rp.qm * p.w[t];
    if (rp.deltaf) qw *= p.wd[t];
    const float a  = qw / (r.dxp * r.dyp);
    const float w0 = (1.f - fx) * a, w1 = fx * a;
    const float vx = ux * gni, vy = uy * gni, vz = uz * gni;

    atomicAdd(&r.rho1[i0].x, w0 * cth);       atomicAdd(&r.rho1[i0].y, w0 * sth);
    atomicAdd(&r.rho1[i1].x, w1 * cth);       atomicAdd(&r.rho1[i1].y, w1 * sth);
    atomicAdd(&r.j1x[i0].x,  w0 * vx * cth);  atomicAdd(&r.j1x[i0].y,  w0 * vx * sth);
    atomicAdd(&r.j1x[i1].x,  w1 * vx * cth);  atomicAdd(&r.j1x[i1].y,  w1 * vx * sth);
    atomicAdd(&r.j1y[i0].x,  w0 * vy * cth);  atomicAdd(&r.j1y[i0].y,  w0 * vy * sth);
    atomicAdd(&r.j1y[i1].x,  w1 * vy * cth);  atomicAdd(&r.j1y[i1].y,  w1 * vy * sth);
    atomicAdd(&r.j1z[i0].x,  w0 * vz * cth);  atomicAdd(&r.j1z[i0].y,  w0 * vz * sth);
    atomicAdd(&r.j1z[i1].x,  w1 * vz * cth);  atomicAdd(&r.j1z[i1].y,  w1 * vz * sth);
}

// ---- m = 1 Maxwell + cold fluid (∂x = staggered Dx, ∂y = i·k1 exact) ----
//
// Same time layout as the legacy step (k_faraday(dt/2) → J → cold →
// k_faraday(dt/2) → k_ampere, ε0 = 1), so the m = 0 and m = 1 systems stay
// time-centered identically. Complex arithmetic is componentwise except the
// ik1 terms:  i·z = (−z.y, z.x),  −i·z = (z.y, −z.x).

// B1 += dt2·(−curl_k E1), curl_k = (Dx, ik1, 0)×:
//   ∂B1x/∂t = −ik1·E1z            (node i, E1z node ✓)
//   ∂B1y/∂t = +Dx E1z             (i+½ from nodes ✓)
//   ∂B1z/∂t = −(Dx E1y − ik1·E1x) (i+½; E1x already at i+½ ✓)
static __global__ void k_rsm_faraday(RsmViews r, float dt2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.nx) return;
    const int ip = r.wrap(i + 1);
    const float idx = 1.f / r.dxp, k1 = r.k1;
    const float2 ez = r.e1z[i], ezp = r.e1z[ip];
    const float2 ey = r.e1y[i], eyp = r.e1y[ip];
    const float2 ex = r.e1x[i];
    r.b1x[i].x += dt2 * k1 * ez.y;                       // −i k1 Ez
    r.b1x[i].y -= dt2 * k1 * ez.x;
    r.b1y[i].x += dt2 * (ezp.x - ez.x) * idx;
    r.b1y[i].y += dt2 * (ezp.y - ez.y) * idx;
    r.b1z[i].x += dt2 * (-(eyp.x - ey.x) * idx - k1 * ex.y);   // +i k1 Ex
    r.b1z[i].y += dt2 * (-(eyp.y - ey.y) * idx + k1 * ex.x);
}

// E1 += dt·(c²·curl_k B1 − J1):
//   ∂E1x/∂t = c²·ik1·B1z − J1x            (i+½ ✓)
//   ∂E1y/∂t = −c²·Dx B1z − J1y            (node from i±½ ✓)
//   ∂E1z/∂t = c²·(Dx B1y − ik1·B1x) − J1z (node ✓)
static __global__ void k_rsm_ampere(RsmViews r, float dt, float c2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.nx) return;
    const int im = r.wrap(i - 1);
    const float idx = 1.f / r.dxp, k1 = r.k1;
    const float2 bz = r.b1z[i], bzm = r.b1z[im];
    const float2 by = r.b1y[i], bym = r.b1y[im];
    const float2 bx = r.b1x[i];
    r.e1x[i].x += dt * (-c2 * k1 * bz.y - r.j1x[i].x);          // +i k1 Bz
    r.e1x[i].y += dt * ( c2 * k1 * bz.x - r.j1x[i].y);
    r.e1y[i].x += dt * (-c2 * (bz.x - bzm.x) * idx - r.j1y[i].x);
    r.e1y[i].y += dt * (-c2 * (bz.y - bzm.y) * idx - r.j1y[i].y);
    r.e1z[i].x += dt * ( c2 * ((by.x - bym.x) * idx + k1 * bx.y) - r.j1z[i].x);
    r.e1z[i].y += dt * ( c2 * ((by.y - bym.y) * idx - k1 * bx.x) - r.j1z[i].y);
}

// m = 1 cold fluid at the NODES — the complex twin of k_cold_fluid_full:
// symmetric E1x gather from i±½ (real transfer function, zero phase error),
// half kick / EXACT rotation about the local B0(x)·x̂ / half kick. The
// rotation is a REAL linear operator, so the Re and Im parts of the complex
// amplitude rotate identically. This vx response IS the E∥ screening — the
// physics the transverse-only legacy fluid cannot do (constitution R3/R5).
static __global__ void k_rsm_cold_fluid(RsmViews r, RunParams rp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.nx) return;
    const int im = r.wrap(i - 1);
    const float2 Ex = { 0.5f * (r.e1x[im].x + r.e1x[i].x),
                        0.5f * (r.e1x[im].y + r.e1x[i].y) };
    const float2 Ey = r.e1y[i], Ez = r.e1z[i];
    const float h = (float)(0.5 * rp.dt * rp.qm);
    float2 vx = r.vc1x[i], vy = r.vc1y[i], vz = r.vc1z[i];
    vx.x += h * Ex.x; vx.y += h * Ex.y;
    vy.x += h * Ey.x; vy.y += h * Ey.y;
    vz.x += h * Ez.x; vz.y += h * Ez.y;
    const float b0 = rp.b0_prof ? bg::b0x(rp, i * r.dxp) : rp.B0[0];
    float sn, cs;
    sincosf((float)(-rp.qm * rp.dt) * b0, &sn, &cs);
    const float2 vy2 = { vy.x * cs - vz.x * sn, vy.y * cs - vz.y * sn };
    const float2 vz2 = { vy.x * sn + vz.x * cs, vy.y * sn + vz.y * cs };
    vy = vy2; vz = vz2;                                  // vx: along b̂, kept
    vx.x += h * Ex.x; vx.y += h * Ex.y;
    vy.x += h * Ey.x; vy.y += h * Ey.y;
    vz.x += h * Ez.x; vz.y += h * Ez.y;
    r.vc1x[i] = vx; r.vc1y[i] = vy; r.vc1z[i] = vz;
}

// Cold m = 1 current J1c = qm·nc·vc1, node → staggered sites with the same
// symmetric averages as k_cold_current_full (J1x between nodes i, i+1).
static __global__ void k_rsm_cold_current(RsmViews r, RunParams rp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.nx) return;
    const int ip = r.wrap(i + 1);
    const float jc = (float)(rp.qm * rp.cold_nc);
    r.j1x[i].x += jc * 0.5f * (r.vc1x[i].x + r.vc1x[ip].x);
    r.j1x[i].y += jc * 0.5f * (r.vc1x[i].y + r.vc1x[ip].y);
    r.j1y[i].x += jc * r.vc1y[i].x;
    r.j1y[i].y += jc * r.vc1y[i].y;
    r.j1z[i].x += jc * r.vc1z[i].x;
    r.j1z[i].y += jc * r.vc1z[i].y;
}

// ---- fused RSM push: ONE Boris push with the total field (R4) + m0
// Esirkepov deposit + charge-conserving modal m1 deposit, same worldline ----

// 1D linear interp from a complex array (i0/i1 pre-wrapped).
__device__ inline float2 rsm_lin(const DeviceView<float2>& a, int i0, int i1,
                                 float f) {
    return float2{ (1.f - f) * a.ptr[i0].x + f * a.ptr[i1].x,
                   (1.f - f) * a.ptr[i0].y + f * a.ptr[i1].y };
}

// Full-f AND δf (δf since rsm-tao17: wd rides the same worldline, deposit
// weights carry wd — see yee_deltaf_update call and the DeltaF policy lines).
// Gather: F_tot = F0 (staggered 2D gather, ny = 1) + 2·Re[F1(x)·e^{iθ}],
// θ = 2π·y (deck contract Ly = 2π/k1). The m = 1 B rides in the WAVE dB so
// the b0_prof mirror branches see the total wave field in one rotation.
//
// m = 1 deposit — exact discrete continuity by construction (R3). With
// S = 1D CIC shape, e = e^{−iθ}, the per-particle charge change factorizes
// EXACTLY:  S¹e¹ − S⁰e⁰ = ē·(S¹−S⁰) + S̄·(e¹−e⁰),  ē = ½(e⁰+e¹),
// S̄ = ½(S⁰+S¹). The first term is the 1D Esirkepov x-flow (prefix sum W →
// J1x links, phase factor ē); the second is the phase flow absorbed by
//   J1y(i) = i·qw·S̄ᵢ·(e¹−e⁰)/(k1·Δt·ΔxΔy)
// — the analytic worldline integral: for Δθ = k1·v_y·Δt this is
// qw·ṽ_y·S̄·e^{−iθ̄}, ṽ_y = 2·sin(Δθ/2)/(k1·Δt) → v_y. Then
//   Δρ1/Δt + Dx J1x + i·k1·J1y = 0   to float roundoff (gate V3).
__device__ inline void rsm_advance_particle(ParticleViews p, YeeViews v,
                                            RsmViews r, RunParams rp, int t,
                                            float& x0, float& y0,
                                            float& x1, float& y1,
                                            float& vz1, float& s0, float& c0) {
    constexpr float TWO_PI = 6.283185307179586f;

    x0 = p.x[t]; y0 = p.y[t];
    __sincosf(TWO_PI * y0, &s0, &c0);                 // e^{iθ0} = c0 + i·s0

    float Ex  = yee::gather_stag(v.ex, v, x0, y0, 0.5f, 0.f);
    float Ey  = yee::gather_stag(v.ey, v, x0, y0, 0.f, 0.5f);
    float Ez  = yee::gather_stag(v.ez, v, x0, y0, 0.f, 0.f);
    float dBx = yee::gather_stag(v.bx, v, x0, y0, 0.f, 0.5f);
    float dBy = yee::gather_stag(v.by, v, x0, y0, 0.5f, 0.f);
    float dBz = yee::gather_stag(v.bz, v, x0, y0, 0.5f, 0.5f);
    {   // m = 1 gather: nodes (E1y, E1z, B1x) and half sites (E1x, B1y, B1z)
        int in0 = (int)floorf(x0);
        const float fn = x0 - (float)in0;
        in0 = r.wrap(in0);
        const int in1 = r.wrap(in0 + 1);
        const float xs = x0 - 0.5f;
        int ih0 = (int)floorf(xs);
        const float fh = xs - (float)ih0;
        ih0 = r.wrap(ih0);
        const int ih1 = r.wrap(ih0 + 1);
        const float2 E1x = rsm_lin(r.e1x, ih0, ih1, fh);
        const float2 E1y = rsm_lin(r.e1y, in0, in1, fn);
        const float2 E1z = rsm_lin(r.e1z, in0, in1, fn);
        const float2 B1x = rsm_lin(r.b1x, in0, in1, fn);
        const float2 B1y = rsm_lin(r.b1y, ih0, ih1, fh);
        const float2 B1z = rsm_lin(r.b1z, ih0, ih1, fh);
        Ex  += 2.f * (E1x.x * c0 - E1x.y * s0);       // 2·Re[F1 e^{iθ}] (R6)
        Ey  += 2.f * (E1y.x * c0 - E1y.y * s0);
        Ez  += 2.f * (E1z.x * c0 - E1z.y * s0);
        dBx += 2.f * (B1x.x * c0 - B1x.y * s0);
        dBy += 2.f * (B1y.x * c0 - B1y.y * s0);
        dBz += 2.f * (B1z.x * c0 - B1z.y * s0);
    }

    // Boris kick-rotate-kick, identical op sequence to yee_advance_particle
    // (minus the [pump] plane-wave driver, still rejected at construction —
    // the [antenna] current column is a pure m = 0 J source and needs nothing
    // here; δf supported since rsm-tao17, see yee_deltaf_update call below).
    float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    const float uxo = ux, uyo = uy, uzo = uz;   // u^{n-1/2} (delta-f centering)
    const float qmh = (float)(rp.qm * 0.5 * rp.dt);
    ux += qmh * Ex; uy += qmh * Ey; uz += qmh * Ez;
    const float gri = rp.rel ? rsqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
    if (rp.b0_prof) {
        // parabolic/dipole B0(x) + effective mirror field (b0_prof = 3 is
        // impossible here: rsm requires ny = 1)
        const float xph = x0 * v.dxp;
        const float b0  = bg::b0x(rp, xph);
        const float mc  = bg::db0dx(rp, xph) / (2.f * b0 * (float)rp.qm);
        detail::boris_rotate(ux, uy, uz,
                             dBx + b0, dBy + mc * uz, dBz - mc * uy, qmh * gri);
    } else {
        detail::boris_rotate(ux, uy, uz, dBx + rp.B0[0], dBy + rp.B0[1],
                             dBz + rp.B0[2], qmh * gri);
    }
    ux += qmh * Ex; uy += qmh * Ey; uz += qmh * Ez;
    p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;
    const float gni = rp.rel ? rsqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
    vz1 = uz * gni;

    // δf weight update (rsm-tao17): the shared legacy kernel, fed the TOTAL
    // wave fields (Ex..dBz above include 2·Re[F1 e^{iθ}]). f0 is θ-independent
    // by construction, so ∂lnf0/∂u is unchanged and the m = 1 force simply
    // rides in F — no new terms. Same call site as the legacy pusher (after
    // the second kick, before the move).
    if (rp.deltaf)
        yee::yee_deltaf_update(p, rp, t, x0, v.dxp, Ex, Ey, Ez, dBx, dBy, dBz,
                               uxo, uyo, uzo, ux, uy, uz);

    x1 = x0 + (float)(ux * gni * rp.dt / (double)v.dxp);
    y1 = y0 + (float)(uy * gni * rp.dt / (double)v.dyp);
    // M2 bounded x (mirror runs): specular wall + optional hybrid transverse
    // damping or atmo loss cone — same scheme as yee_advance_particle, fold
    // BEFORE the deposits so Esirkepov sees the reflected path. y (the
    // phase) is untouched.
    if (rp.bnd_x) {
        bool refl = false;
        if (x1 < 0.f)               { x1 = -x1;             p.ux[t] = ux = -ux; refl = true; }
        else if (x1 >= (float)v.nx) { x1 = 2.f * v.nx - x1; p.ux[t] = ux = -ux; refl = true; }
        if (x1 >= (float)v.nx) x1 = nextafterf((float)v.nx, 0.f);
        // atmo mode (bnd_x = 3): adiabatic return — u_perp KEPT, except the
        // real atmospheric loss cone mapped to the wall: mirror point beyond
        // the atmosphere <=> u⊥²·b_atm < b_l·u² -> precipitate (u⊥ = 0
        // return). Identical operator to yee2d.hpp; only the stored p.uy/uz
        // change — locals (and vz1, already computed) stay pre-zeroed so the
        // deposits match the flat path's op order exactly.
        if (rp.bnd_x == 3 && refl && rp.bnd_batm > 0.0) {
            const float bl = rp.b0_prof
                ? bg::b0x(rp, x1 * (float)v.dxp) / (float)rp.B0[0] : 1.f;
            const float up2 = uy * uy + uz * uz;
            if (up2 * (float)rp.bnd_batm < bl * (up2 + ux * ux))
                { p.uy[t] = 0.f; p.uz[t] = 0.f; }
        }
        if (rp.bnd_x == 2) {
            const float nd = (float)rp.bnd_nd;
            float d = 0.f;
            if (x1 < nd)             d = (nd - x1) / nd;
            else if (x1 > v.nx - nd) d = (x1 - (v.nx - nd)) / nd;
            if (d > 0.f) {
                const float m = __expf((float)(-rp.bnd_numax * rp.dt) * d * d);
                p.uy[t] = uy * m; p.uz[t] = uz * m;
            }
        }
    }
}

// m = 1 modal-deposit sinks. Global: flat path + tiled-path strays (full
// wrap). The shared-tile twin lives inside k_rsm_push_esirkepov_tiled.
struct GlobalJ1Sink {
    RsmViews r;
    __device__ void j1x(int i, float re, float im) {
        const int w = r.wrap(i);
        atomicAdd(&r.j1x[w].x, re); atomicAdd(&r.j1x[w].y, im);
    }
    __device__ void j1y(int i, float re, float im) {
        const int w = r.wrap(i);
        atomicAdd(&r.j1y[w].x, re); atomicAdd(&r.j1y[w].y, im);
    }
    __device__ void j1z(int i, float re, float im) {
        const int w = r.wrap(i);
        atomicAdd(&r.j1z[w].x, re); atomicAdd(&r.j1z[w].y, im);
    }
};

// ---- m = 1 deposit (charge-conserving modal form, header note) ------------
template<class J1Sink>
__device__ inline void rsm_modal_scatter(float x0, float x1, float y1,
                                         float qw, float vz1, float s0, float c0,
                                         const RsmViews& r, const RunParams& rp,
                                         int ib, J1Sink sink) {
    constexpr float TWO_PI = 6.283185307179586f;
    float s1v, c1v;
    __sincosf(TWO_PI * y1, &s1v, &c1v);
    // e⁰ = (c0, −s0), e¹ = (c1, −s1); ē and i·(e¹−e⁰)
    const float ebr = 0.5f * (c0 + c1v), ebi = -0.5f * (s0 + s1v);
    const float pfr = s1v - s0,          pfi = c1v - c0;    // i·(e¹−e⁰)
    float Sx0[4], Sx1[4];
    yee::s1_shape4(x0, ib, Sx0);
    yee::s1_shape4(x1, ib, Sx1);
    const float invV  = 1.f / (r.dxp * r.dyp);
    const float cJx   = -qw * invV * r.dxp * (float)(1.0 / rp.dt);
    const float cJy   =  qw * invV / (r.k1 * (float)rp.dt);
    const float cJz   =  qw * invV * vz1;
    float W = 0.f;
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int i = ib - 1 + n;                      // sink resolves wrap/offset
        if (n < 3) {                                   // links (ib-1+n)+½
            W += Sx1[n] - Sx0[n];
            if (W != 0.f) sink.j1x(i, cJx * W * ebr, cJx * W * ebi);
        }
        const float Sb = 0.5f * (Sx0[n] + Sx1[n]);     // nodes
        if (Sb != 0.f) {
            sink.j1y(i, cJy * Sb * pfr, cJy * Sb * pfi);
            sink.j1z(i, cJz * Sb * ebr, cJz * Sb * ebi);
        }
    }
}

// Flat-path kernel: advance + m0 global scatter + m1 global modal scatter.
static __global__ void k_rsm_push_esirkepov(ParticleViews p, YeeViews v,
                                            RsmViews r, RunParams rp,
                                            double tnow) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;
    float x0, y0, x1, y1, vz1, s0, c0;
    rsm_advance_particle(p, v, r, rp, t, x0, y0, x1, y1, vz1, s0, c0);
    p.x[t] = x1; p.y[t] = y1;         // unwrapped; Particles::migrate wraps

    const int ib = (int)floorf(fminf(x0, x1));
    const int jb = (int)floorf(fminf(y0, y1));
    const float qw = (float)rp.qm * p.w[t];
    yee::esirkepov_scatter(x0, y0, x1, y1, qw, vz1, v, (float)(1.0 / rp.dt),
                           ib, jb, yee::GlobalJSink{v});
    rsm_modal_scatter(x0, x1, y1, qw, vz1, s0, c0, r, rp, ib, GlobalJ1Sink{r});
}

// TiledBinnedDeposit for the RSM path (2026-07-31): the flat path's ~22
// extra global atomics/particle for m1 on top of the unsorted global m0
// scatter ran at 1.9e9 p-steps/s vs 1.6e10 on the tiled m0 path (the first
// x4×RSM arm priced at 26 h — stopped and redone). Same machinery as
// yee::k_push_esirkepov_tiled: one block per TX*TY tile, physical sort,
// DRIFT apron, stray fallback to the global sinks, fused migrate. The m1
// lines are x-only, so their tile adds just 3·SW float2 (~0.6 KB) of
// shared memory beside the m0 tile; the m0 apron test covers the m1
// stencil exactly (same {ib-1..ib+2} x-support).
template<int TX, int TY, int DRIFT>
static __global__ void k_rsm_push_esirkepov_tiled(ParticleViews p, BinViews b,
                                                  YeeViews v, RsmViews r,
                                                  RunParams rp, double tnow,
                                                  int blocks_per_tile) {
    constexpr int PAD = DRIFT + 3;               // node reach below tile origin
    constexpr int SW  = TX + 2 * DRIFT + 6;      // local nodes [-PAD, TX+DRIFT+2]
    constexpr int SH  = TY + 2 * DRIFT + 6;
    __shared__ float  s_jx[SH * SW], s_jy[SH * SW], s_jz[SH * SW];
    __shared__ float2 s_j1x[SW], s_j1y[SW], s_j1z[SW];

    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX;
    const int gj0 = (tile / b.ntx) * TY;

    for (int c = threadIdx.x; c < SH * SW; c += blockDim.x)
        s_jx[c] = s_jy[c] = s_jz[c] = 0.f;
    for (int c = threadIdx.x; c < SW; c += blockDim.x)
        s_j1x[c] = s_j1y[c] = s_j1z[c] = float2{0.f, 0.f};
    __syncthreads();

    struct SharedJSink {
        float *jx_, *jy_, *jz_; int i0, j0;
        __device__ void jx(int i, int j, float a) { atomicAdd(&jx_[(j - j0) * SW + (i - i0)], a); }
        __device__ void jy(int i, int j, float a) { atomicAdd(&jy_[(j - j0) * SW + (i - i0)], a); }
        __device__ void jz(int i, int j, float a) { atomicAdd(&jz_[(j - j0) * SW + (i - i0)], a); }
    } shs{s_jx, s_jy, s_jz, gi0 - PAD, gj0 - PAD};
    struct SharedJ1Sink {
        float2 *jx_, *jy_, *jz_; int i0;
        __device__ void j1x(int i, float re, float im) { atomicAdd(&jx_[i - i0].x, re); atomicAdd(&jx_[i - i0].y, im); }
        __device__ void j1y(int i, float re, float im) { atomicAdd(&jy_[i - i0].x, re); atomicAdd(&jy_[i - i0].y, im); }
        __device__ void j1z(int i, float re, float im) { atomicAdd(&jz_[i - i0].x, re); atomicAdd(&jz_[i - i0].y, im); }
    } sh1{s_j1x, s_j1y, s_j1z, gi0 - PAD};

    const float invdt = (float)(1.0 / rp.dt);
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int t = beg + lane * blockDim.x + threadIdx.x; t < end; t += step) {
        float x0, y0, x1, y1, vz1, s0, c0;
        rsm_advance_particle(p, v, r, rp, t, x0, y0, x1, y1, vz1, s0, c0);

        const int ib = (int)floorf(fminf(x0, x1));
        const int jb = (int)floorf(fminf(y0, y1));
        float qw = (float)rp.qm * p.w[t];
        if (rp.deltaf) qw *= p.wd[t];              // DeltaF policy: δJ = q w wd v
        // union stencil {ib-1..ib+2} inside the shared apron?
        const int il = ib - gi0, jl = jb - gj0;
        if (il - 1 >= -PAD && il + 2 <= TX + DRIFT + 2 &&
            jl - 1 >= -PAD && jl + 2 <= TY + DRIFT + 2) {
            yee::esirkepov_scatter(x0, y0, x1, y1, qw, vz1, v, invdt, ib, jb, shs);
            rsm_modal_scatter(x0, x1, y1, qw, vz1, s0, c0, r, rp, ib, sh1);
        } else {                                        // stray since last sort
            yee::esirkepov_scatter(x0, y0, x1, y1, qw, vz1, v, invdt, ib, jb,
                                   yee::GlobalJSink{v});
            rsm_modal_scatter(x0, x1, y1, qw, vz1, s0, c0, r, rp, ib,
                              GlobalJ1Sink{r});
        }

        // fused migrate (yee tiled scheme; the y-wrap IS the phase wrap)
        float xw = fmodf(x1, (float)v.nx); if (xw < 0.f) xw += (float)v.nx;
        float yw = fmodf(y1, (float)v.ny); if (yw < 0.f) yw += (float)v.ny;
        if (xw >= (float)v.nx) xw = 0.f;
        if (yw >= (float)v.ny) yw = 0.f;
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
    for (int c = threadIdx.x; c < SW; c += blockDim.x) {
        const float2 a = s_j1x[c], e = s_j1y[c], d = s_j1z[c];
        if (a.x != 0.f || a.y != 0.f || e.x != 0.f || e.y != 0.f ||
            d.x != 0.f || d.y != 0.f) {
            const int gc = r.wrap(gi0 - PAD + c);
            atomicAdd(&r.j1x[gc].x, a.x); atomicAdd(&r.j1x[gc].y, a.y);
            atomicAdd(&r.j1y[gc].x, e.x); atomicAdd(&r.j1y[gc].y, e.y);
            atomicAdd(&r.j1z[gc].x, d.x); atomicAdd(&r.j1z[gc].y, d.y);
        }
    }
}

// M2-style x-end damping for the m = 1 fields — the complex twin of
// k_damp_x (same masks: mn at integer-x sites e1y/e1z/b1x + cold vc1, mh at
// half-integer sites e1x/b1y/b1z). Re and Im damp identically.
static __global__ void k_rsm_damp_x(RsmViews r, const float* __restrict__ mn,
                                    const float* __restrict__ mh,
                                    int have_cold) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.nx) return;
    const float a = mn[i], b = mh[i];
    if (a == 1.f && b == 1.f) return;
    r.e1y[i].x *= a; r.e1y[i].y *= a;
    r.e1z[i].x *= a; r.e1z[i].y *= a;
    r.b1x[i].x *= a; r.b1x[i].y *= a;
    r.e1x[i].x *= b; r.e1x[i].y *= b;
    r.b1y[i].x *= b; r.b1y[i].y *= b;
    r.b1z[i].x *= b; r.b1z[i].y *= b;
    if (have_cold) {                     // m1 cold fluid dies with its wave
        r.vc1x[i].x *= a; r.vc1x[i].y *= a;
        r.vc1y[i].x *= a; r.vc1y[i].y *= a;
        r.vc1z[i].x *= a; r.vc1z[i].y *= a;
    }
}

// One pass of 1D binomial [1,2,1]/4 smoothing on a complex array — the m = 1
// twin of the m = 0 k_binomial3x3 (linear + shift-invariant, so the implied
// ∂ρ1/∂t is filtered identically and modal Gauss consistency is preserved).
static __global__ void k_rsm_binomial(const float2* __restrict__ src,
                                      float2* __restrict__ dst, int nx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int im = i > 0 ? i - 1 : nx - 1;
    const int ip = i + 1 < nx ? i + 1 : 0;
    dst[i].x = 0.25f * src[im].x + 0.5f * src[i].x + 0.25f * src[ip].x;
    dst[i].y = 0.25f * src[im].y + 0.5f * src[i].y + 0.25f * src[ip].y;
}

} // namespace detail

// Host-side owner of the m = 1 state. Allocated ONLY when rp.rsm is set.
struct RsmState {
    DeviceArray<float2> e1x, e1y, e1z, b1x, b1y, b1z, j1x, j1y, j1z, rho1;
    DeviceArray<float2> vc1x, vc1y, vc1z;    // m=1 cold-fluid velocity
    DeviceArray<float2> j1tmp;               // binomial-filter scratch
    int    nx  = 0;
    double dxp = 0.0, dyp = 0.0, k1 = 0.0;

    void init(const Grid& g, const RunParams& rp, cudaStream_t s) {
        if (!rp.rsm) throw std::runtime_error("rsm: init called with rp.rsm = 0");
        if (g.ny != 1) throw std::runtime_error("rsm: needs ny = 1");
        if (rp.rsm_k1 <= 0.0) throw std::runtime_error("rsm: rsm_k1 not set");
        const double twopi = 2.0 * 3.14159265358979323846;
        if (std::abs(rp.rsm_k1 * g.dy - twopi) > 1e-6 * twopi)
            throw std::runtime_error("rsm: Ly must equal 2*pi/k1 "
                                     "(particle y IS the phase theta/2*pi)");
        nx = g.nx; dxp = g.dx; dyp = g.dy; k1 = rp.rsm_k1;
        for (auto* a : {&e1x, &e1y, &e1z, &b1x, &b1y, &b1z,
                        &j1x, &j1y, &j1z, &rho1, &vc1x, &vc1y, &vc1z, &j1tmp}) {
            *a = DeviceArray<float2>((std::size_t)nx);
            a->zero(s);
        }
    }

    // rp.jfilter passes of the 1D binomial on the m = 1 currents (the m = 0
    // twin runs in MaxwellSimulation::step_at; both are linear so the two
    // harmonics stay identically smoothed).
    void filter_j1(int passes, cudaStream_t s) {
        constexpr int threads = 128;
        const int blocks = (nx + threads - 1) / threads;
        for (auto* a : {&j1x, &j1y, &j1z}) {
            float2 *src = a->data(), *dst = j1tmp.data();
            for (int pass = 0; pass < passes; ++pass) {
                detail::k_rsm_binomial<<<blocks, threads, 0, s>>>(src, dst, nx);
                float2* t = src; src = dst; dst = t;
            }
            if (src != a->data())
                CUDA_CHECK(cudaMemcpyAsync(a->data(), src,
                                           (std::size_t)nx * sizeof(float2),
                                           cudaMemcpyDeviceToDevice, s));
        }
    }

    RsmViews views() {
        return RsmViews{ e1x.view(), e1y.view(), e1z.view(),
                         b1x.view(), b1y.view(), b1z.view(), j1x.view(),
                         j1y.view(), j1z.view(), rho1.view(),
                         vc1x.view(), vc1y.view(), vc1z.view(),
                         nx, (float)dxp, (float)dyp, (float)k1 };
    }
};

// One m = 1 field step, legacy time layout (faraday-half → [hot J1 deposit
// goes here] → cold → faraday-half → ampere). V2 form: cold fluid only —
// the hot J1 slot is filled when the particle coupling lands.
inline void rsm_cold_step(RsmState& r, const RunParams& rp, cudaStream_t s) {
    constexpr int threads = 128;
    const int blocks = (r.nx + threads - 1) / threads;
    RsmViews v = r.views();
    const float dt2 = (float)(0.5 * rp.dt);
    const float c2  = (float)(rp.c * rp.c);
    r.j1x.zero(s); r.j1y.zero(s); r.j1z.zero(s);
    detail::k_rsm_faraday<<<blocks, threads, 0, s>>>(v, dt2);
    if (rp.cold_nc > 0.0) {
        detail::k_rsm_cold_fluid<<<blocks, threads, 0, s>>>(v, rp);
        detail::k_rsm_cold_current<<<blocks, threads, 0, s>>>(v, rp);
    }
    detail::k_rsm_faraday<<<blocks, threads, 0, s>>>(v, dt2);
    detail::k_rsm_ampere<<<blocks, threads, 0, s>>>(v, (float)rp.dt, c2);
    CUDA_CHECK(cudaPeekAtLastError());
}

namespace detail {
// rsm-tao17: div-free m = 1 field seed — B1z ONLY (div B1 = Dx·B1x + ik1·B1y
// never sees B1z, so ∇·B1 = 0 by construction), per-node uniform random
// phase, white in k∥ (every parallel wavelength seeded equally; the Umeda
// masks eat the layer content). δf REQUIRES this: weights start at 0, so the
// m = 1 system has no wd-weighted shot-noise floor to self-seed from — the
// full-f runs ignite from load noise, an unseeded δf run stays at exactly 0
// (the G1 null gate). Salted hash stream, disjoint from loader and θ-init.
static __global__ void rsm_seed_kernel(RsmViews r, float amp,
                                       unsigned long seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r.nx) return;
    const float ph = 6.283185307179586f
                   * (float)rng_uniform(i, 12, seed ^ 0xC2B2AE3D27D4EB4FULL);
    float sn, cs;
    __sincosf(ph, &sn, &cs);
    r.b1z[i].x = amp * cs;
    r.b1z[i].y = amp * sn;
}
} // namespace detail

// Seed the m = 1 harmonic with rp.rsm_seed ([rsm] seed, amplitude relative to
// wce): coefficient magnitude 0.5·amp·|B0eq| so the physical per-node field
// 2·Re[B1z e^{iθ}] has amplitude amp·wce (factor-2 ledger R6). Call once
// after init, before the first step; no-op when rsm_seed = 0.
inline void rsm_seed_init(RsmState& r, const RunParams& rp, cudaStream_t s) {
    if (rp.rsm_seed <= 0.0) return;
    const float amp = 0.5f * (float)(rp.rsm_seed * std::abs(rp.B0[0]));
    constexpr int threads = 128;
    detail::rsm_seed_kernel<<<(r.nx + threads - 1) / threads, threads, 0, s>>>(
        r.views(), amp, rp.rng_seed);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Randomize the particle phase after the load (R1 remedy). Call once, after
// initialize_mirror / initialize, before the first step.
inline void rsm_theta_init(Particles& parts, const RunParams& rp, cudaStream_t s) {
    constexpr int threads = 256;
    const int blocks = ((int)parts.n + threads - 1) / threads;
    detail::rsm_theta_init_kernel<<<blocks, threads, 0, s>>>(parts.views(),
                                                             rp.rng_seed);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Snapshot the m = 1 moments ρ1, J1 from the current particle state.
inline void rsm_deposit_moments(Particles& parts, RsmState& r,
                                const RunParams& rp, cudaStream_t s) {
    r.j1x.zero(s); r.j1y.zero(s); r.j1z.zero(s); r.rho1.zero(s);
    constexpr int threads = 256;
    const int blocks = ((int)parts.n + threads - 1) / threads;
    detail::rsm_deposit_moments_kernel<<<blocks, threads, 0, s>>>(
        parts.views(), r.views(), rp);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace arc

#endif // ARC_PIC_RSM_OBLIQUE_HPP
