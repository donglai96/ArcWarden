// V1 quiet-hold gate (PLAN_2D_REBORN §5, P2) — an (E,μ)-loaded ISOTROPIC
// δf shell in the linedipole box must sit still: the equilibrium is exact
// (isotropic ⇒ all mapping terms vanish; only the shell-edge term acts, and
// it only carries the bounded gyro-ripple of L(x,z)), so wd must show NO
// secular growth, no marker may escape the field-aligned reflecting walls,
// kinetic energy must hold (the δf answer to the ρ⊥/dz WARN ruling), and
// the wave fields must stay at the δf noise floor.
//
// Pre-registered expectation (kinetic2d.hpp header): wd_rms rises from
// wdnoise to a bounded PLATEAU set by the coherent shell-edge gyro-ripple
// (markers in the Gaussian edge see dlnf ≈ (v·∇L)·a/σ² oscillating at the
// gyro-period); the gate is plateau, not absolute level — secular growth
// (late/mid ratio) is the failure signature.
//
// Setup: linedipole L0 = 200, B0eq = 0.2 (ωpe/Ωe = 5), box [60,240]×[±140]
// at dx = 0.25, masks nd = 80, walls at the mask interior edge
// [80,220]×[±120] (the L = 200 line meets x = 80 at λ ≈ 51°). Shell:
// flat 20 + Gaussian σ = 5 in L, isotropic uth = 0.14 (ρ⊥/dz = 2.8 at the
// equator), ppc 50 → 6.6M markers, wdnoise = 1e-3. Run 2 bounce periods
// (T_b ≈ 9000 at v⊥ ≈ 0.14, T = 18000, dt = 0.15).
//
// Gates: (a) zero escapes; (b) wd_rms plateau: rms(T)/rms(T/2) < 1.3;
// (c) hot kinetic energy drift < 1%; (d) W_EM stays below 100× its early
// (t = T/8) value — the δf field floor does not grow.

#include "pic2d/deck2d.hpp"
#include "pic2d/kinetic2d.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-26s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

int main(int argc, char** argv) {
    // V1b (arg "aniso"): T⊥/T∥ = 2 at physically-inert density n0 = 1e-5
    // (γ ∝ n0 → no real instability inside 2 T_b; the weight equation has
    // no n0, so the μ-slot machinery and the ζ-mapped loader are exercised
    // at full strength). Pre-registered expectation: a load-vs-f₀ mismatch
    // transient (the loader samples the local mapped bi-Max at the particle
    // position, f₀ lives on gc invariants), then plateau — same gates.
    const bool aniso = argc > 1 && std::string(argv[1]) == "aniso";
    std::printf("test_deltaf2d_hold — V1%s δf shell quiet hold\n",
                aniso ? "b ANISOTROPIC (T⊥/T∥ = 2)" : " isotropic");

    const double DX = 0.25, DT = 0.15;
    const double X0 = 60, X1 = 240, Z0 = -140, Z1 = 140;
    const int NX = int((X1 - X0) / DX), NZ = int((Z1 - Z0) / DX);
    const long NSTEP = 120000;                    // T = 18000 ≈ 2 T_b
    const int ND = 80;

    Fields2D F;
    F.allocate(NX, NZ);
    F.dx = DX; F.dz = DX; F.dt = DT; F.cspeed = 1.0; F.nc = 1.0;
    F.x0 = X0; F.z0 = Z0;
    F.bg.prof = int(B0Prof::linedipole);
    F.bg.B0eq = 0.2; F.bg.L0 = 200.0;
    F.bg.finalize();
    F.build_masks(ND, 0.05);

    KineticCfg C;
    C.qm = -1.f; C.deltaf = 1; C.rel = 0;
    C.tpar = 0.14f * 0.14f;
    C.tperp = aniso ? 2.f * C.tpar : C.tpar;
    C.n0 = aniso ? 1e-5f : 0.01f;
    C.L0 = 200.f; C.dL = 20.f; C.edge = 5.f;
    C.wdnoise = 1e-3f;
    C.wx0 = float(X0 + ND * DX); C.wx1 = float(X1 - ND * DX);
    C.wz0 = float(Z0 + ND * DX); C.wz1 = float(Z1 - ND * DX);

    // loader bounding box: shell support ∩ wall rectangle
    const float bx0 = C.wx0, bx1 = std::min(C.wx1, float(C.L0 + 0.5 * C.dL + 3 * C.edge + 1));
    const float bz0 = C.wz0, bz1 = C.wz1;

    const double nint = k2d::shell_density_integral(C, F.bg, bx0, bx1, bz0, bz1, 0.5);
    const uint64_t NMARK = 6600000;
    const float wmark = float(C.n0 * nint / double(NMARK));
    std::printf("  shell ∫(n/n0)dA = %.1f → %.2e markers, w = %.3e\n",
                nint, double(NMARK), wmark);

    MarkerStore mk;
    mk.allocate(NMARK);
    mk.n = NMARK;
    {
        MarkerViews mv = mk.views();
        const int tb = 256;
        k2d::k_load<<<int((NMARK + tb - 1) / tb), tb>>>(
            mv, C, F.bg, bx0, bx1, bz0, bz1, wmark, 20260817u, NMARK, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double* dacc = nullptr;
    CUDA_CHECK(cudaMalloc(&dacc, sizeof(double)));
    auto wd_rms = [&]() {
        CUDA_CHECK(cudaMemset(dacc, 0, sizeof(double)));
        MarkerViews mv = mk.views();
        k2d::k_wd_stats<<<int((NMARK + 255) / 256), 256>>>(mv, dacc, NMARK);
        double h;
        CUDA_CHECK(cudaMemcpy(&h, dacc, sizeof(double), cudaMemcpyDeviceToHost));
        return std::sqrt(h / double(NMARK));
    };
    auto hot_ke_and_escapes = [&](long& esc) {
        std::vector<float> hx(NMARK), hz(NMARK), a(NMARK), b(NMARK), c2(NMARK);
        CUDA_CHECK(cudaMemcpy(hx.data(), mk.x.data(), NMARK * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hz.data(), mk.z.data(), NMARK * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(a.data(), mk.ux.data(), NMARK * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), mk.uy.data(), NMARK * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c2.data(), mk.uz.data(), NMARK * 4, cudaMemcpyDeviceToHost));
        double ke = 0;
        esc = 0;
        for (uint64_t i = 0; i < NMARK; ++i) {
            ke += 0.5 * (double(a[i]) * a[i] + double(b[i]) * b[i] +
                         double(c2[i]) * c2[i]);
            // escape = GUIDING CENTER beyond a REAL wall by > 2 c/ωpe (the
            // outer-radial edge is not a wall — gc cannot cross L; gyro
            // excursions of the particle position are not escapes)
            float xg, zg;
            k2d::gc_pos(hx[i], hz[i], b[i], 1.f, C, F.bg, xg, zg);
            if (xg < C.wx0 - 2 || zg < C.wz0 - 2 || zg > C.wz1 + 2 ||
                hx[i] < float(X0) || hx[i] > float(X1) || hz[i] < float(Z0) ||
                hz[i] > float(Z1))
                ++esc;
        }
        return ke;
    };

    const double wd0 = wd_rms();
    long esc0;
    const double ke0 = hot_ke_and_escapes(esc0);
    std::printf("  t=0: wd_rms %.3e, KE %.6e, escapes %ld\n", wd0, ke0, esc0);

    double wd_half = 0, wem_early = -1;
    const dim3 tb(f2d::TX, f2d::TZ);
    for (long n = 0; n < NSTEP; ++n) {
        FieldViews2D v = F.views();
        const Range r = F.full();
        const dim3 nb = f2d::blocks_for(r);
        f2d::k_faraday<<<nb, tb>>>(v, r, float(DT / 2));
        F.zero_j();
        {
            MarkerViews mv = mk.views();
            k2d::k_push_deposit<<<int((NMARK + 255) / 256), 256>>>(
                mv, C, v, F.bg, float(X0), float(Z0), NMARK);
        }
        F.reduce_j();
        F.filter_j();                    // jfilter=3, the production recipe
        f2d::k_cold_step<<<nb, tb>>>(v, r, F.bg, float(X0), float(Z0));
        f2d::k_cold_current<<<nb, tb>>>(v, r);
        f2d::k_faraday<<<nb, tb>>>(v, r, float(DT / 2));
        f2d::k_ampere<<<nb, tb>>>(v, r);
        f2d::k_mask_e<<<nb, tb>>>(v, r);
        f2d::k_mask_b<<<nb, tb>>>(v, r);

        if (n == NSTEP / 8) {
            double W[2];
            F.energies(W);
            wem_early = W[0];
        }
        if (n == NSTEP / 2) wd_half = wd_rms();
        if (n % 20000 == 19999) {
            double W[2];
            F.energies(W);
            std::printf("    t=%7.0f  wd_rms %.3e  W_EM %.3e\n", (n + 1) * DT,
                        wd_rms(), W[0]);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double wd1 = wd_rms();
    long esc1;
    const double ke1 = hot_ke_and_escapes(esc1);
    double W[2];
    F.energies(W);

    gate("zero escapes", esc1 == 0, double(esc1), 0.5);
    gate("wd plateau (T vs T/2)", wd1 / wd_half < 1.3, wd1 / wd_half, 1.3);
    gate("hot KE drift", std::fabs(ke1 / ke0 - 1.0) < 1e-2,
         std::fabs(ke1 / ke0 - 1.0), 1e-2);
    gate("field floor no growth", W[0] < 100.0 * wem_early, W[0] / wem_early, 100.0);
    cudaFree(dacc);

    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
