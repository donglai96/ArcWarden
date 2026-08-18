// pic2d — analytic 2D background-field family (PLAN_2D_REBORN §2.1, D1).
//
// Every member is defined by a flux function ψ(x,z), B0 = ∇ψ × ŷ, so
// ∇·B0 = 0 holds EXACTLY in the 2D wave metric (the property Lu 2019's
// 3D-dipole-in-(p,q) embedding cannot guarantee; see PLAN_2D_REBORN §1.2.3).
// B0 is analytic in the pusher and never gridded: Faraday sees wave fields
// only, so no discrete solenoidality error can enter through the background.
//
// Members:
//   linedipole  ψ = M x/r²  (production geometry)
//     Bx = −2Mxz/r⁴, Bz = M(x²−z²)/r⁴, |B| = M/r² (exact identity).
//     Field lines are circles through the origin: r = L cosλ with
//     λ = atan2(z,x) the latitude and L = r²/x the line's equatorial
//     crossing ("L-shell"). Arc length s = L·λ exactly; mirror profile
//     B(s)/B_eq = sec²(s/L) = 1 + (s/L)² + …, i.e. the equatorial
//     parabolic scale is l_re = L, matching the 1D dipole family's knob.
//     Vacuum field (∇×B0 = 0): no background current anywhere in the box.
//     |∇L| = sec²λ = B/B_eq (flux conservation), so the width of a shell
//     ΔL measured across lines is ΔL cos²λ and the shell area to ±λ_w is
//     ΔL·L·(λ_w + sinλ_w cosλ_w) — used by the memory pre-flight.
//   kemirror    ψ = −B0eq(x̃ + a x̃ z̃²)  (Ke 2017 / legacy prof=3 slab,
//     axes rotated to the pic2d convention: field ∥ ẑ at the equator,
//     mirroring along z). Bz = B0eq(1+a z̃²), Bx = −2a B0eq x̃ z̃.
//     Solenoidal exactly; NOT curl-free (carries the external coil
//     current, as in the legacy code) — fine, B0 never enters Faraday.
//   uniform / tilted   B0 = B0eq(sinθ, 0, cosθ) — Li-2019 arm and
//     dispersion gates (θ = 0 gives B0 ∥ ẑ).
//
// Conventions: (x,z) meridional plane, ŷ out of plane; latitude λ from
// the +x axis; the "equator" of the production geometry is the +x axis.
// All functions are __host__ __device__ and templated on Real so the
// orbit gates run in double on the host while kernels run float.

#ifndef ARC_PIC2D_BACKGROUND2D_HPP
#define ARC_PIC2D_BACKGROUND2D_HPP

#ifndef __CUDACC__
#define ARC2D_HD
#else
#define ARC2D_HD __host__ __device__
#endif

#include <cmath>

namespace arc2d {

enum class B0Prof : int { uniform = 0, tilted = 1, kemirror = 2, linedipole = 3,
                          dipole2d = 4 };
// dipole2d (2026-08-18 realism upgrade, user-directed): flux function
//     ψ = (B0eq L0³/2) · x⁴/r⁶  =  g(L),  L = r³/x²,
// whose level curves ARE the real 3D dipole's meridional line shapes
// r = L cos²λ (equatorial curvature radius L/3, not the circle's L/2).
// Along a line: B/B_eq = sec²λ·√(1+4tan²λ) — within ~5% of the true
// dipole √(1+3sin²λ)/cos⁶λ for λ ≤ 45° (the circle was ~40% low), and
// B_eq(L) = B0eq (L0/L)³, the real radial scaling. Exactly 2D-solenoidal;
// NOT curl-free (a distributed supporting current, as in Ke's slab
// mirror) — harmless: B0 is force-only, the wave equations never see it.

// POD, copied by value into kernels (repo Views pattern).
struct Background2D {
    int    prof   = int(B0Prof::linedipole);
    double B0eq   = 0.2;     // |B| at the reference equator point
    double L0     = 1330.5;  // linedipole: reference L (= l_re); kemirror: unused
    double M      = 0.0;     // linedipole moment, derived: B0eq·L0²
    double a      = 0.0;     // kemirror curvature (l_re = 1/sqrt(a))
    double xc     = 0.0;     // kemirror equator offsets (x̃ = x−xc, z̃ = z−zc)
    double zc     = 0.0;
    double theta  = 0.0;     // tilted: angle of B0 from ẑ toward x̂ (radians)

    void finalize() { M = B0eq * L0 * L0; }
};

template <typename Real>
struct Vec2 { Real x, z; };

// B0 components at (x,z).
template <typename Real>
ARC2D_HD inline Vec2<Real> b0_field(const Background2D& bg, Real x, Real z) {
    switch (B0Prof(bg.prof)) {
        case B0Prof::linedipole: {
            const Real r2 = x * x + z * z;
            const Real ir4 = Real(1) / (r2 * r2);
            return { Real(-2.0 * bg.M) * x * z * ir4,
                     Real(bg.M) * (x * x - z * z) * ir4 };
        }
        case B0Prof::dipole2d: {
            // ψ = C x⁴/r⁶, C = B0eq L0³/2:
            // Bx = ∂ψ/∂z = −6C x⁴ z / r⁸ ;  Bz = −∂ψ/∂x = C x³(2x²−4z²)/r⁸
            // (signs fixed so Bz(eq) = +B0eq(L0/x)³, matching linedipole)
            const Real r2 = x * x + z * z;
            const Real ir8 = Real(1) / (r2 * r2 * r2 * r2);
            const Real C = Real(0.5 * bg.B0eq * bg.L0 * bg.L0 * bg.L0);
            const Real x3 = x * x * x;
            return { Real(-6) * C * x3 * x * z * ir8,
                     C * x3 * (Real(2) * x * x - Real(4) * z * z) * ir8 };
        }
        case B0Prof::kemirror: {
            const Real xt = x - Real(bg.xc), zt = z - Real(bg.zc);
            return { Real(-2.0 * bg.a * bg.B0eq) * xt * zt,
                     Real(bg.B0eq) * (Real(1) + Real(bg.a) * zt * zt) };
        }
        case B0Prof::tilted:
            return { Real(bg.B0eq * std::sin(bg.theta)),
                     Real(bg.B0eq * std::cos(bg.theta)) };
        case B0Prof::uniform:
        default:
            return { Real(0), Real(bg.B0eq) };
    }
}

template <typename Real>
ARC2D_HD inline Real b0_abs(const Background2D& bg, Real x, Real z) {
    if (B0Prof(bg.prof) == B0Prof::linedipole)
        return Real(bg.M) / (x * x + z * z);       // exact, no sqrt of squares
    const Vec2<Real> b = b0_field(bg, x, z);
    return std::sqrt(b.x * b.x + b.z * b.z);
}

// Unit vector along B0 (in-plane; b_y = 0 for the whole family).
template <typename Real>
ARC2D_HD inline Vec2<Real> b0_bhat(const Background2D& bg, Real x, Real z) {
    const Vec2<Real> b = b0_field(bg, x, z);
    const Real ib = Real(1) / std::sqrt(b.x * b.x + b.z * b.z);
    return { b.x * ib, b.z * ib };
}

// ---- linedipole field-line geometry (analytic; used by the δf weight
// equation, the shell loader, and region-tagged diagnostics) ----------------

// L-shell of the field line through (x,z): L = r²/x.  (x > 0 in the box.)
template <typename Real>
ARC2D_HD inline Real lshell(Real x, Real z) { return (x * x + z * z) / x; }

// Latitude λ = atan2(z,x); arc length along the local line s = L·λ.
template <typename Real>
ARC2D_HD inline Real latitude(Real x, Real z) { return std::atan2(z, x); }

template <typename Real>
ARC2D_HD inline Real arc_s(Real x, Real z) { return lshell(x, z) * latitude(x, z); }

// ∇L = ((x²−z²)/x², 2z/x); b̂·∇L = 0 exactly (moving along a line keeps L).
template <typename Real>
ARC2D_HD inline Vec2<Real> grad_lshell(Real x, Real z) {
    return { (x * x - z * z) / (x * x), Real(2) * z / x };
}

// ∇|B0| (needed by the δf weight equation's mapping term). linedipole:
// |B| = M/r² ⇒ ∇|B| = −2|B| r̂/r (exact). kemirror/uniform: from the
// closed forms; tilted/uniform have ∇|B| = 0.
template <typename Real>
ARC2D_HD inline Vec2<Real> grad_babs(const Background2D& bg, Real x, Real z) {
    switch (B0Prof(bg.prof)) {
        case B0Prof::linedipole: {
            const Real r2 = x * x + z * z;
            const Real f = Real(-2.0 * bg.M) / (r2 * r2);
            return { f * x, f * z };
        }
        case B0Prof::kemirror: {
            // |B|² = B0²[(1+az̃²)² + 4a²x̃²z̃²]; ∇|B| = ∇|B|²/(2|B|)
            const Real xt = x - Real(bg.xc), zt = z - Real(bg.zc);
            const Real B0 = Real(bg.B0eq), a = Real(bg.a);
            const Real oz = Real(1) + a * zt * zt;
            const Real Babs = B0 * std::sqrt(oz * oz + Real(4) * a * a * xt * xt * zt * zt);
            const Real dx2 = B0 * B0 * Real(8) * a * a * xt * zt * zt;
            const Real dz2 = B0 * B0 * (Real(4) * a * zt * oz +
                                        Real(8) * a * a * xt * xt * zt);
            return { dx2 / (Real(2) * Babs), dz2 / (Real(2) * Babs) };
        }
        default:
            return { Real(0), Real(0) };
    }
}

// Mirror ratio at latitude λ on any linedipole line: B/B_eq = sec²λ.
ARC2D_HD inline double mirror_ratio(double lambda) {
    const double c = std::cos(lambda);
    return 1.0 / (c * c);
}

// Position on the reference line L at latitude λ (deck/geometry helpers).
ARC2D_HD inline void line_point(double L, double lambda, double& x, double& z) {
    const double c = std::cos(lambda);
    x = L * c * c;
    z = L * c * std::sin(lambda);
}

// ---- profile-dispatched field-line geometry (linedipole | dipole2d) -------
// Every consumer (loader ζ-mapping, deck gates, diagnostics line) goes
// through these; adding a background member means extending them, nothing
// else.

ARC2D_HD inline bool has_lines(const Background2D& bg) {
    return B0Prof(bg.prof) == B0Prof::linedipole ||
           B0Prof(bg.prof) == B0Prof::dipole2d;
}

template <typename Real>
ARC2D_HD inline Real lshell_of(const Background2D& bg, Real x, Real z) {
    const Real r2 = x * x + z * z;
    if (B0Prof(bg.prof) == B0Prof::dipole2d)
        return r2 * Real(std::sqrt(double(r2))) / (x * x);   // L = r³/x²
    return r2 / x;                                           // circle: L = r²/x
}

ARC2D_HD inline double beq_of(const Background2D& bg, double L) {
    if (B0Prof(bg.prof) == B0Prof::dipole2d) {
        const double q = bg.L0 / L;
        return bg.B0eq * q * q * q;               // real L⁻³ scaling
    }
    return bg.M / (L * L);
}

ARC2D_HD inline double mirror_ratio_of(const Background2D& bg, double lam) {
    const double c = std::cos(lam), t = std::tan(lam);
    if (B0Prof(bg.prof) == B0Prof::dipole2d)
        return std::sqrt(1.0 + 4.0 * t * t) / (c * c);
    return 1.0 / (c * c);
}

ARC2D_HD inline void line_point_of(const Background2D& bg, double L, double lam,
                                   double& x, double& z) {
    const double c = std::cos(lam);
    if (B0Prof(bg.prof) == B0Prof::dipole2d) {
        x = L * c * c * c;                        // r = L cos²λ (true dipole shape)
        z = L * c * c * std::sin(lam);
    } else {
        x = L * c * c;
        z = L * c * std::sin(lam);
    }
}

// arc length along the line from the equator to λ (host; Simpson)
inline double arc_s_of(const Background2D& bg, double L, double lam) {
    const int n = 64;
    double s = 0;
    for (int i = 0; i < n; ++i) {
        const double a = lam * i / n, b = lam * (i + 1) / n, m = 0.5 * (a + b);
        auto ds = [&](double u) {
            const double c = std::cos(u);
            if (B0Prof(bg.prof) == B0Prof::dipole2d)
                return L * c * std::sqrt(1.0 + 3.0 * (1 - c * c));
            return L;                             // circle: ds = L dλ exactly
        };
        s += (b - a) / 6.0 * (ds(a) + 4.0 * ds(m) + ds(b));
    }
    return s;
}

}  // namespace arc2d

#endif  // ARC_PIC2D_BACKGROUND2D_HPP
