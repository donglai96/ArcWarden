// pic2d — triggering antenna (chirp2d port + user audit hardening).
// Split out of sim2d.hpp 2026-08-20: the orchestrator should not carry
// physics sources. See k_antenna2d comments for the construction contract.

#ifndef ARC_PIC2D_ANTENNA2D_HPP
#define ARC_PIC2D_ANTENNA2D_HPP

#include "pic2d/fields2d.hpp"

namespace arc2d {

// ---- triggering antenna v2 (chirp2d port + user audit 2026-08-19) ----------
// DISCRETELY DIVERGENCE-FREE source: the in-plane (cos-phase) component is
// built from a cell-centre stream function M(x,z) = G_L(L) * H(z),
//   Jx(i+1/2,k) =  cph * [M(i+1/2,k+1/2) - M(i+1/2,k-1/2)]/dz
//   Jz(i,k+1/2) = -cph * [M(i+1/2,k+1/2) - M(i-1/2,k+1/2)]/dx
// so the Yee-node divergence cancels EXACTLY (no external charge, no Gauss
// pollution). H(z) = int envz(z') cos(kpar z') dz' (host-tabulated): the
// cos(kpar z) CARRIER matches the target whistler wavelength, H' = envz*cos
// makes Jx ~ G_L envz cos(kpar z) to leading order, and H(+-inf) ~
// exp(-(kpar*sigz)^2/2) ~ 0 closes the stream function compactly.
// Jy (sin phase, out-of-plane, automatically div-free) copies the same
// shape for circular R polarization. Drive evaluated at t^{n+1/2}.
// W_ant = -sum J_ant.E dV dt accumulated (wacc) for the energy ledger.
struct AntCfg {
    float amp, w0, L0, sigL, sigz, trmp, toff, kpar;
    float tper;   // >0: envelope repeats with this period (element trains);
                  // trmp/toff apply within each cycle, carrier phase runs on
};

namespace a2d {
#ifdef __CUDACC__
static __global__ void k_antenna2d(FieldViews2D v, Background2D bg, AntCfg a,
                                   float x0, float z0, double thalf,
                                   const float* Htab, int nzt, float zt0,
                                   float dzt, double* wacc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= v.nx || k >= v.nz) return;
    const float zn = z0 + k * v.dz;
    if (fabsf(zn) > 6.f * a.sigz + 2.f * v.dz) return;  // 6σ: keeps the
    // discrete curl cancellation below 1e-7 (4σ truncation measured 4.3e-5)
    // envelope time: cycles when tper > 0 (repeated triggering); the
    // carrier phase below stays on thalf so cycles are phase-continuous
    const double te = (a.tper > 0) ? fmod(thalf, double(a.tper)) : thalf;
    if (a.toff > 0 && te >= a.toff) return;
    const float xn = x0 + i * v.dx;
    if (fabsf(lshell_of<float>(bg, xn, zn) - a.L0) > 6.f * a.sigL + 2.f * v.dx)
        return;
    const int c = v.idx(i, k);
    if (c < 0) return;
    double ramp = 1.0;
    if (a.trmp > 0) {
        if (te < a.trmp) ramp = te / a.trmp;
        else if (a.toff > 0 && te > a.toff - a.trmp)
            ramp = (a.toff - te) / a.trmp;
    }
    const float g0 = float(a.amp * ramp);
    const float cph = cosf(float(a.w0 * thalf)), sph = sinf(float(a.w0 * thalf));
    auto GL = [&](float x, float z) {
        const float dL = lshell_of<float>(bg, x, z) - a.L0;
        return expf(-dL * dL / (2.f * a.sigL * a.sigL));
    };
    auto Hz = [&](float z) {           // linear interp of the host table
        const float u = (z - zt0) / dzt;
        const int j = max(0, min(nzt - 2, int(u)));
        const float f = u - j;
        return Htab[j] * (1.f - f) + Htab[j + 1] * f;
    };
    auto M = [&](float x, float z) { return GL(x, z) * Hz(z); };
    const float xh = xn + 0.5f * v.dx, zh = zn + 0.5f * v.dz;
    const float jx = g0 * cph * (M(xh, zh) - M(xh, zh - v.dz)) / v.dz;
    const float jz = -g0 * cph * (M(xh, zh) - M(xh - v.dx, zh)) / v.dx;
    const float envz = expf(-zn * zn / (2.f * a.sigz * a.sigz));
    const float jy = g0 * sph * GL(xn, zn) * envz * cosf(a.kpar * zn);
    atomicAdd(&v.jx[c], jx);
    atomicAdd(&v.jz[c], jz);
    atomicAdd(&v.jy[c], jy);
    if (wacc) {
        const double dV = double(v.dx) * v.dz;
        atomicAdd(wacc, -double(jx * v.ex[c] + jy * v.ey[c] + jz * v.ez[c])
                            * dV * v.dt);
    }
}
#endif
}  // namespace a2d

}  // namespace arc2d

#endif  // ARC_PIC2D_ANTENNA2D_HPP
