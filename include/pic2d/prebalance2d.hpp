// pic2d — diamagnetic pre-balance (P1 v2 fix, user-directed 2026-08-19).
//
// The (E,μ) shell loader drops a β⊥ ≈ 6% hot shell into the VACUUM dipole2d
// field. The missing ingredient of the true anisotropic equilibrium is the
// pressure dent δB∥ ≈ −p⊥/B0; without it the plasma digs the dent itself on
// the bounce timescale (measured on the first P1 run: B∥ dent deepening
// linearly to −1.4e-3 by t = 4500/Ωe, heading for the force-balance value
// −p⊥/B0 ≈ −6e-3, plus a saturated N/S-antisymmetric FAC By loop ~1e-3 —
// a quasi-static structure 3× the whistler amplitude, poisoning wave maps
// and the W_EM gauge).
//
// Fix: seed the dent in the WAVE B field at t = 0 (B0 stays the analytic
// vacuum field everywhere). Construction: node stream function
//     A_y(i,k) = F(L)·T(λ),
//     F'(L) = p⊥,eq(L)/B_eq(L)   (p⊥,eq = Σ_sp n0 T⊥ prof_sp(L)),
//     T(λ)  = 1/(ζ(λ)² b(λ)²)   (mapped p⊥ decay along the line, b = B/Beq,
//                                 ζ = 1 + A_eq(1 − 1/b), species-0 A_eq),
// differenced onto the Yee sites exactly as Faraday's curl expects:
//     bx(i,k+½) += (A(i,k+1) − A(i,k))/dz
//     bz(i+½,k) −= (A(i+1,k) − A(i,k))/dx
// so the discrete ∇·B stays identically zero. On-line check: at the equator
// δBz = −∂A/∂x = −F'(L) = −p⊥,eq/B_eq < 0 — the dent. The A = F·T ansatz is
// exact in δB∥ and leaves a spurious ⊥ component F·T'·|∇λ| ~ 1e-6 (shell-
// integrated F ≈ 3e-4 × T' ≈ 4 × |∇λ| ≈ 1e-3) — negligible, and the masks
// own everything outside the band anyway.
//
// Caveats (recorded): loader ζ-mapping still uses vacuum B0 (3% mismatch at
// shell centre = second-order residual relaxation); the FAC/By loop part of
// the equilibrium is NOT seeded (it saturated at ~1e-3 within 2 bounce
// periods unfixed — expected to re-form small and fast); δf runs would need
// the dent reflected in the weight-equation base state — prebalance is
// intended for FULL-F arms (deck gate warns).

#ifndef ARC_PIC2D_PREBALANCE2D_HPP
#define ARC_PIC2D_PREBALANCE2D_HPP

#include "pic2d/sim2d.hpp"

#include <cstdio>
#include <vector>

namespace arc2d {

inline void apply_prebalance(Sim2D& S, const Deck2D& d) {
    if (!has_lines(d.bg)) {
        std::fprintf(stderr, "prebalance: needs a field-line background — skipped\n");
        return;
    }
    Fields2D& F = S.F;
    const int nx = F.nx, nz = F.nz;

    // ---- F(L) table: cumulative trapezoid of p⊥,eq(L)/B_eq(L) ------------
    const int NL = 8192;
    double Llo = 1e30, Lhi = 0;
    for (int c = 0; c < 4; ++c) {
        const double xx = (c & 1) ? d.x1 : d.x0, zz = (c >> 1) ? d.z1 : d.z0;
        const double L = lshell_of<double>(d.bg, xx, zz);
        Lhi = std::max(Lhi, L);
    }
    Llo = std::max(1.0, d.x0 * 0.999);     // L >= x everywhere in the box
    Lhi *= 1.001;
    std::vector<double> Ftab(NL, 0.0);
    auto peq = [&](double L) {
        double p = 0;
        for (const auto& sp : S.sp)
            p += sp.C.n0 * sp.C.tperp * k2d::shell_prof(float(L), sp.C);
        return p;
    };
    const double dL = (Lhi - Llo) / (NL - 1);
    for (int i = 1; i < NL; ++i) {
        const double La = Llo + (i - 1) * dL, Lb = Llo + i * dL;
        Ftab[i] = Ftab[i - 1] + 0.5 * dL * (peq(La) / beq_of(d.bg, La) +
                                            peq(Lb) / beq_of(d.bg, Lb));
    }
    auto Fof = [&](double L) {
        const double u = (L - Llo) / dL;
        const int i = std::max(0, std::min(NL - 2, int(u)));
        const double f = u - i;
        return Ftab[i] * (1 - f) + Ftab[i + 1] * f;
    };
    const double Aeq0 = S.sp.empty() ? 0.0
        : double(S.sp[0].C.tperp) / S.sp[0].C.tpar - 1.0;

    // ---- node stream function A(i,k) → staggered δB ----------------------
    std::vector<double> A(size_t(nx + 1) * (nz + 1));
    for (int k = 0; k <= nz; ++k)
        for (int i = 0; i <= nx; ++i) {
            const double x = d.x0 + i * d.dx, z = d.z0 + k * d.dz;
            const double L = lshell_of<double>(d.bg, x, z);
            const double lam = std::atan2(z, x);
            const double b = mirror_ratio_of(d.bg, lam);
            const double zeta = 1.0 + Aeq0 * (1.0 - 1.0 / b);
            A[size_t(k) * (nx + 1) + i] = Fof(L) / (zeta * zeta * b * b);
        }
    std::vector<float> dbx(size_t(nx) * nz), dbz(size_t(nx) * nz);
    double dbmax = 0;
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            const size_t n0 = size_t(k) * (nx + 1) + i;
            const double bxv = (A[n0 + (nx + 1)] - A[n0]) / d.dz;
            const double bzv = -(A[n0 + 1] - A[n0]) / d.dx;
            dbx[size_t(k) * nx + i] = float(bxv);
            dbz[size_t(k) * nx + i] = float(bzv);
            dbmax = std::max(dbmax, std::abs(bxv) + std::abs(bzv));
        }
    std::vector<float> stor;
    F.pack_host(dbx, stor);
    CUDA_CHECK(cudaMemcpy(F.bx.data(), stor.data(), stor.size() * 4,
                          cudaMemcpyHostToDevice));
    F.pack_host(dbz, stor);
    CUDA_CHECK(cudaMemcpy(F.bz.data(), stor.data(), stor.size() * 4,
                          cudaMemcpyHostToDevice));
    std::printf("  prebalance: diamagnetic dent seeded, max|δB| %.2e "
                "(%.1f%% of B0eq), eq dent %.2e\n",
                dbmax, 100.0 * dbmax / d.bg.B0eq,
                -peq(d.bg.L0) / beq_of(d.bg, d.bg.L0));
    for (const auto& sp : S.sp)
        if (sp.C.deltaf)
            std::fprintf(stderr, "prebalance: WARNING δf species '%s' — the "
                                 "weight equation does not know the dent\n",
                         sp.name.c_str());
}

}  // namespace arc2d

#endif  // ARC_PIC2D_PREBALANCE2D_HPP
