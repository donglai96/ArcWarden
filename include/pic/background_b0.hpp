// M4/M5a — analytic field-aligned background B0(x) profile (L-shell plan).
//
// b0_prof = 1, parabolic near-equator field along the field-aligned coordinate:
//   B0(x) = B0eq (1 + a (x - xc)^2) x̂,   B0eq = RunParams::B0[0],
// with x, xc, a in PHYSICAL units (a in 1/length^2).
//
// b0_prof = 2, dipole B0(s) (M5a): the true dipole magnitude along a field
// line of scale LRE = L·R_E (RunParams::b0_lre, physical units; compressed
// geometries just use a smaller LRE),
//   B(λ)/Beq = sqrt(1 + 3 sin²λ) / cos⁶λ,
//   s(λ)     = LRE·[ t·sqrt(1+3t²) + asinh(sqrt(3) t)/sqrt(3) ] / 2,  t = sinλ.
// fit_dipole() (called from the deck finalize once Lx is known) inverts s(λ)
// by bisection and least-squares fits an EVEN polynomial
//   B/Beq ≈ Σ_{i=0..5} c_i u^i,   u = ((x - xc)/sref)²,  sref = s_max,
// storing c_i in RunParams::b0_c. Device/host code then evaluates the
// polynomial — no table pointer, both sides see identical values. Fit
// residual is checked (< 2e-4 rel) and the fit throws if the domain reaches
// past the field-line apex.
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
//  - the delta-f weight drive and the (E,mu) mirror load only consume b(x) =
//    b0x/B0eq, so they work with any profile behind these two calls.

#pragma once

#include "pic/config.hpp"

#include <cmath>
#include <stdexcept>

namespace arc {
namespace bg {

__host__ __device__ inline float b0x(const RunParams& rp, float xph) {
    const float d = xph - (float)rp.b0_xc;
    if (rp.b0_prof == 2) {
        const float sn = d / (float)rp.b0_sref;
        const float u  = sn * sn;
        float p = (float)rp.b0_c[5];
        for (int i = 4; i >= 0; --i) p = p * u + (float)rp.b0_c[i];
        return rp.B0[0] * p;
    }
    return rp.B0[0] * (1.f + (float)rp.b0_a * d * d);
}

__host__ __device__ inline float db0dx(const RunParams& rp, float xph) {
    const float d = xph - (float)rp.b0_xc;
    if (rp.b0_prof == 2) {
        const float sn = d / (float)rp.b0_sref;
        const float u  = sn * sn;
        float dp = 5.f * (float)rp.b0_c[5];              // d(poly)/du
        for (int i = 4; i >= 1; --i) dp = dp * u + (float)i * (float)rp.b0_c[i];
        return rp.B0[0] * dp * 2.f * d / (float)((double)rp.b0_sref * rp.b0_sref);
    }
    return rp.B0[0] * 2.f * (float)rp.b0_a * d;
}

// ---- host-side dipole reference (exact, used by fit_dipole and the gates) ----

// arc length s(λ) along the field line, in units of LRE (= L·R_E)
inline double dipole_s_of_lambda(double lam) {
    const double t = std::sin(lam);
    return 0.5 * (t * std::sqrt(1.0 + 3.0 * t * t)
                  + std::asinh(std::sqrt(3.0) * t) / std::sqrt(3.0));
}

// B(λ)/Beq on the dipole field line
inline double dipole_b_of_lambda(double lam) {
    const double t = std::sin(lam), c2 = 1.0 - t * t;
    return std::sqrt(1.0 + 3.0 * t * t) / (c2 * c2 * c2);
}

// invert s(λ) by bisection; s_over_lre in [0, s(90°)=1.3801...)
inline double dipole_lambda_of_s(double s_over_lre) {
    if (s_over_lre <= 0.0) return 0.0;
    double lo = 0.0, hi = 0.5 * M_PI;
    if (s_over_lre >= dipole_s_of_lambda(hi))
        throw std::runtime_error("dipole_lambda_of_s: s beyond field-line apex");
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (dipole_s_of_lambda(mid) < s_over_lre) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// exact B(s)/Beq at physical arc distance s from the equator
inline double dipole_b_of_s(double s, double lre) {
    return dipole_b_of_lambda(dipole_lambda_of_s(std::fabs(s) / lre));
}

// Fit the even polynomial (RunParams::b0_c, b0_sref) over |s| <= smax.
// Call at deck-finalize time with smax = the half-domain (plus margin for
// float round-off at the reflecting wall). Requires rp.b0_lre > 0.
inline void fit_dipole(RunParams& rp, double smax) {
    if (rp.b0_lre <= 0.0)
        throw std::runtime_error("fit_dipole: b0_lre must be > 0");
    constexpr int NC = 6, M = 400;
    rp.b0_sref = smax;

    // least squares in u = (s/smax)^2 over Chebyshev-like sampling of s
    double A[NC][NC] = {}, rhs[NC] = {};
    for (int k = 0; k < M; ++k) {
        const double s = smax * (k + 0.5) / M;
        const double u = (s / smax) * (s / smax);
        const double f = dipole_b_of_s(s, rp.b0_lre);
        double pw[NC];
        pw[0] = 1.0;
        for (int i = 1; i < NC; ++i) pw[i] = pw[i - 1] * u;
        for (int i = 0; i < NC; ++i) {
            rhs[i] += pw[i] * f;
            for (int j = 0; j < NC; ++j) A[i][j] += pw[i] * pw[j];
        }
    }
    // Gaussian elimination with partial pivoting (6x6)
    for (int col = 0; col < NC; ++col) {
        int best = col;
        for (int r = col + 1; r < NC; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[best][col])) best = r;
        if (best != col) {
            for (int j = 0; j < NC; ++j) std::swap(A[col][j], A[best][j]);
            std::swap(rhs[col], rhs[best]);
        }
        for (int r = col + 1; r < NC; ++r) {
            const double m = A[r][col] / A[col][col];
            for (int j = col; j < NC; ++j) A[r][j] -= m * A[col][j];
            rhs[r] -= m * rhs[col];
        }
    }
    for (int i = NC - 1; i >= 0; --i) {
        double v = rhs[i];
        for (int j = i + 1; j < NC; ++j) v -= A[i][j] * rp.b0_c[j];
        rp.b0_c[i] = v / A[i][i];
    }

    // verify the fit against the exact profile
    double maxrel = 0.0;
    for (int k = 0; k <= 100; ++k) {
        const double s = smax * k / 100.0;
        const double u = (s / smax) * (s / smax);
        double p = rp.b0_c[NC - 1];
        for (int i = NC - 2; i >= 0; --i) p = p * u + rp.b0_c[i];
        const double f = dipole_b_of_s(s, rp.b0_lre);
        maxrel = std::max(maxrel, std::fabs(p - f) / f);
    }
    if (maxrel > 2e-4)
        throw std::runtime_error("fit_dipole: fit residual " +
                                 std::to_string(maxrel) + " exceeds 2e-4");
}

} // namespace bg
} // namespace arc
