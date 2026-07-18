// M4 — analytic field-aligned background B0(x) profile (L-shell plan §M4).
//
// Parabolic near-equator field along the field-aligned coordinate x:
//   B0(x) = B0eq (1 + a (x - xc)^2) x̂,   B0eq = RunParams::B0[0],
// with x, xc, a in PHYSICAL units (a in 1/length^2). M5 replaces the analytic
// form with a dipole B0(s) table behind these same two calls.
//
// The gyration the 1D-in-x grid cannot resolve is folded into the Boris
// rotation as the velocity-dependent effective field (chirp1d scheme):
//   B_mir = [B0'(x) / (2 B0 (q/m))] (u × x̂) = mcoef (0, uz, -uy),
// so that (q/m) u × B_mir reproduces the gyro-averaged mirror force
//   du_par/dt = -(u_perp^2 / 2 B0) dB0/dx
// exactly (charge-independent — the 1/(q/m) factor generalizes chirp1d's
// hard-coded q/m = -1). Being a pure rotation it conserves |u| identically.
//
// Coupling notes:
//  - requires B0 ∥ x̂ (RunParams::B0[1] = B0[2] = 0) — validated by callers.
//  - deltaf + b0_prof is NOT wired yet: the weight drive assumes a uniform
//    bi-Maxwellian f0; the (E,mu)-mapped load + Tperp(x) drive (chirp1d
//    tperp_of) is the next M4 item. Callers must reject the combination.

#pragma once

#include "pic/config.hpp"

namespace arc {
namespace bg {

__host__ __device__ inline float b0x(const RunParams& rp, float xph) {
    const float d = xph - (float)rp.b0_xc;
    return rp.B0[0] * (1.f + (float)rp.b0_a * d * d);
}

__host__ __device__ inline float db0dx(const RunParams& rp, float xph) {
    return rp.B0[0] * 2.f * (float)rp.b0_a * (xph - (float)rp.b0_xc);
}

} // namespace bg
} // namespace arc
