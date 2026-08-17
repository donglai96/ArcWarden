// V2c-lite absorber gate (PLAN_2D_REBORN §5, P1) — the Umeda edge-frame
// masks must swallow outgoing waves in the PRODUCTION geometry (linedipole
// background) without interior damping or reflection blow-ups. The
// quantitative WKB flux-conservation gate (the test Lu's metric ambiguity
// fails) is scheduled with the antenna in P3; this gate establishes the
// absorbing boundary the flagship deck relies on.
//
// Setup: linedipole L = 200 (strong-B inner corner exercises the EXACT
// Rodrigues cold rotation at Ωe·dt ≈ 0.11 — the reason the fluid uses exact
// rotation, not the Boris tan-approximation), uniform cold nc = 1, box
// x ∈ [60, 240], z ∈ ±140 at dx = 0.25, masks nd = 80 cells, nu = 0.05.
//
// Seed: an R-polarized whistler packet ON the L = 200 line at the equator,
// k₀ = 1.0 ∥ B0 (ω ≈ 0.5 Ω_e,eq, v_g ≈ 0.1c), Gaussian envelope σ = 8.
// The in-plane component comes from a vector potential Ay (div-free EXACT);
// By is out-of-plane (div-free trivially). An isotropic blob is the WRONG
// seed for this gate (first-run lesson, kept for the record): its
// quasi-perpendicular content has v_g → 0 and its L-sense half is
// evanescent below Ω_e — 25% of the energy simply never propagates. The
// packet splits ±v_g along the curved line and must exit through the frame
// near λ ≈ ±57° (where the L = 200 line meets the x-wall).
//
// Energy split, by construction: a spatial B⊥ helix at k₀ projects 50/50
// onto (i) the R-whistler branch propagating one way (v_g ≈ 0.1c, absorbed
// at t ≈ 2000 after the s ≈ 198 transit — measured plateau 0.4999 → 0.4993
// over t = 400–1200) and (ii) the high EM branch ω ≈ √(ωpe²+k₀²) ≈ 1.4
// (the polarization that is evanescent below Ω_e reappears there), which
// exits at ≈ c before t = 400 — the prompt loss of the other half.
//
// Gates: (a) whistler half in flight at t=400: W/W0 > 0.4; (b) interior
// truly undamped during transit: W(1200)/W(400) > 0.99; (c) absorption:
// W(T=6000)/W0 < 2e-2; (d) no wall pile-up (reflection forensics):
// late-time energy within mask frame + 10 cells < 5e-3 of W0; (e) finite.

#include "pic2d/fields2d.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-26s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

int main() {
    std::printf("test_absorber2d — Umeda frame masks in the linedipole box\n");

    const double DX = 0.25, DT = 0.05;
    const double X0 = 60.0, X1 = 240.0, Z0 = -140.0, Z1 = 140.0;
    const int NX = int((X1 - X0) / DX), NZ = int((Z1 - Z0) / DX);
    const long NSTEP = 120000;                 // T = 6000/ωpe
    const int ND = 80;                         // mask frame cells

    Fields2D F;
    F.allocate(NX, NZ);
    F.dx = DX; F.dz = DX; F.dt = DT; F.cspeed = 1.0; F.nc = 1.0;
    F.x0 = X0; F.z0 = Z0;
    F.bg.prof = int(B0Prof::linedipole);
    F.bg.B0eq = 0.2; F.bg.L0 = 200.0;
    F.bg.finalize();
    F.build_masks(ND, 0.05);

    // R-polarized whistler packet at the equator: k0 ∥ B0 (∥ ẑ there).
    // In-plane part from Ay (Bx = ∂Ay/∂z, Bz = −∂Ay/∂x — div-free exact);
    // out-of-plane By set directly, phased 90° for circular polarization.
    {
        const double k0 = 1.0, sig = 8.0, amp = 1e-6, xc = 200.0;
        std::vector<float> hby(size_t(NX) * NZ, 0.f);
        std::vector<double> ay(size_t(NX + 1) * (NZ + 1), 0.0);
        auto G = [&](double x, double z) {
            return std::exp(-((x - xc) * (x - xc) + z * z) / (2.0 * sig * sig));
        };
        for (int k = 0; k <= NZ; ++k)          // Ay at in-plane B sites' corners
            for (int i = 0; i <= NX; ++i) {
                const double x = X0 + i * DX, z = Z0 + k * DX;
                ay[size_t(k) * (NX + 1) + i] = -(amp / k0) * std::cos(k0 * z) * G(x, z);
            }
        std::vector<float> hbx(size_t(NX) * NZ, 0.f), hbz(size_t(NX) * NZ, 0.f);
        for (int k = 0; k < NZ; ++k)
            for (int i = 0; i < NX; ++i) {
                const size_t c = size_t(k) * NX + i;
                // Bx at (i, k+½): ∂z Ay along the i-column edge
                hbx[c] = float((ay[size_t(k + 1) * (NX + 1) + i] -
                                ay[size_t(k) * (NX + 1) + i]) / DX);
                // Bz at (i+½, k): −∂x Ay along the k-row edge
                hbz[c] = float(-(ay[size_t(k) * (NX + 1) + i + 1] -
                                 ay[size_t(k) * (NX + 1) + i]) / DX);
                const double x = X0 + (i + 0.5) * DX, z = Z0 + (k + 0.5) * DX;
                hby[c] = float(amp * std::cos(k0 * z) * G(x, z));
            }
        CUDA_CHECK(cudaMemcpy(F.bx.data(), hbx.data(), hbx.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(F.bz.data(), hbz.data(), hbz.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(F.by.data(), hby.data(), hby.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    double W0[2];
    F.energies(W0);
    const double Wtot0 = W0[0] + W0[1];

    bool finite = true;
    double W400 = -1, W1200 = -1;
    std::printf("  energy trajectory (t, W_EM+W_cold, /W0):\n");
    for (long n = 0; n < NSTEP; ++n) {
        F.step();
        if (n == long(400.0 / DT) - 1 || n % 24000 == 23999) {
            double W[2];
            F.energies(W);
            const double Wt = W[0] + W[1];
            if (!std::isfinite(Wt)) finite = false;
            if (n == long(400.0 / DT) - 1) W400 = Wt;
            if (n == 23999) W1200 = Wt;
            std::printf("    t=%6.0f  %.6e  %.3e\n", (n + 1) * DT, Wt, Wt / Wtot0);
        }
    }
    double W1[2];
    F.energies(W1);
    const double Wend = W1[0] + W1[1];

    // reflection forensics: energy sitting in/near the mask frame at the end
    double Wframe = 0;
    {
        std::vector<float> he(size_t(NX) * NZ);
        double Wtail = 0;
        for (auto* arr : {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz}) {
            CUDA_CHECK(cudaMemcpy(he.data(), arr->data(),
                                  he.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (int k = 0; k < NZ; ++k)
                for (int i = 0; i < NX; ++i) {
                    const int d = std::min(std::min(i, NX - 1 - i),
                                           std::min(k, NZ - 1 - k));
                    if (d < ND + 10)
                        Wtail += 0.5 * double(he[size_t(k) * NX + i]) *
                                 he[size_t(k) * NX + i] * DX * DX;
                }
        }
        Wframe = Wtail;
    }

    gate("whistler half in flight", W400 / Wtot0 > 0.4, W400 / Wtot0, 0.4);
    gate("interior undamped (transit)", W1200 / W400 > 0.99, W1200 / W400, 0.99);
    gate("packet absorbed (t=6000)", Wend / Wtot0 < 2e-2, Wend / Wtot0, 2e-2);
    gate("no wall pile-up", Wframe / Wtot0 < 5e-3, Wframe / Wtot0, 5e-3);
    gate("all energies finite", finite && std::isfinite(Wend), finite ? 0.0 : 1.0, 0.5);

    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
