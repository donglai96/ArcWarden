// RSM gate V3 — particle coupling conservation ledger
// (docs/RSM_MODEL_DEFINITION.md ladder V3, risks R3/R4/R6).
//
// A) Deterministic m = 1 gather/kick: one particle at known phase θ in a
//    uniform complex E1 (B = 0) must receive Δu = qm·dt·2·Re[E1·e^{iθ}]
//    exactly — the factor-2 ledger (R6) and the phase convention, kernel-level.
// B) Complex continuity: after one full step of a warm plasma,
//    (ρ1ⁿ⁺¹ − ρ1ⁿ)/Δt + Dx J1x + i·k1·J1y = 0 to float roundoff — the
//    charge-conserving modal deposit (R3) is exact BY CONSTRUCTION, this
//    verifies the construction.
// C) div B1 = Dx B1x + i·k1·B1y stays at roundoff (the m = 1 Faraday
//    update conserves it identically).
// D) Energy ledger through a REAL instability: whistler-unstable anisotropic
//    plasma, m0 + m1 both grow from noise to saturation; the total
//    W = KE + W_m0 + 2·W_m1 must be conserved to a few % of the energy
//    actually transferred. A factor-2 mismatch anywhere in the
//    gather/deposit/Ampère chain shows up here at O(1) of the transfer.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

namespace {

double copy_sum_sq(DeviceArray<float2>& a, std::vector<float2>& h) {
    CUDA_CHECK(cudaMemcpy(h.data(), a.data(), a.size() * sizeof(float2),
                          cudaMemcpyDeviceToHost));
    double s = 0;
    for (auto& z : h) s += (double)z.x * z.x + (double)z.y * z.y;
    return s;
}

// ---- A: deterministic kick --------------------------------------------------
bool test_kick() {
    const double k1 = 0.4, dt = 0.02;
    const int nx = 64;
    Grid g(nx, 1, nx * 0.5, 2.0 * M_PI / k1);
    RunParams rp;
    rp.dt = dt; rp.c = 1.0; rp.qm = -1.0; rp.rsm = 1; rp.rsm_k1 = k1;

    MaxwellSimulation sim(g, rp);
    Particles& parts = sim.particles();
    parts.allocate_n(1);
    const float px = 17.3f, py = 0.37f;
    const float zero = 0.f, one = 1.f;
    const int cell0 = 17;
    CUDA_CHECK(cudaMemcpy(parts.x.data(),  &px,   4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.y.data(),  &py,   4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.ux.data(), &zero, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.uy.data(), &zero, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.uz.data(), &zero, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.w.data(),  &one,  4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.cell.data(), &cell0, 4, cudaMemcpyHostToDevice));

    const float2 a_ey = {3.0e-4f, -1.7e-4f};       // uniform complex E1y
    const float2 a_ex = {-2.1e-4f, 0.9e-4f};       // uniform complex E1x
    std::vector<float2> h(nx, a_ey);
    CUDA_CHECK(cudaMemcpy(sim.rsm().e1y.data(), h.data(), nx * 8, cudaMemcpyHostToDevice));
    std::fill(h.begin(), h.end(), a_ex);
    CUDA_CHECK(cudaMemcpy(sim.rsm().e1x.data(), h.data(), nx * 8, cudaMemcpyHostToDevice));

    sim.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    float ux, uy, uz;
    CUDA_CHECK(cudaMemcpy(&ux, parts.ux.data(), 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&uy, parts.uy.data(), 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&uz, parts.uz.data(), 4, cudaMemcpyDeviceToHost));

    const double th = 2.0 * M_PI * py;
    const double ey_want = rp.qm * dt * 2.0 * (a_ey.x * std::cos(th) - a_ey.y * std::sin(th));
    const double ex_want = rp.qm * dt * 2.0 * (a_ex.x * std::cos(th) - a_ex.y * std::sin(th));
    const double ex_err = std::fabs(ux - ex_want) / std::fabs(ex_want);
    const double ey_err = std::fabs(uy - ey_want) / std::fabs(ey_want);
    std::printf("A kick: dux %.6e want %.6e (err %.1e), duy %.6e want %.6e "
                "(err %.1e), duz %.1e\n", ux, ex_want, ex_err, uy, ey_want,
                ey_err, (double)uz);
    if (ex_err > 1e-4 || ey_err > 1e-4 || std::fabs(uz) > 1e-12) {
        std::printf("A FAIL\n"); return false;
    }
    return true;
}

// ---- B + C: continuity and div B1 ------------------------------------------
bool test_continuity() {
    const double k1 = 0.4, dt = 0.02, dx = 0.5;
    const int nx = 128, ppc = 200;
    Grid g(nx, 1, nx * dx, 2.0 * M_PI / k1);
    RunParams rp;
    rp.dt = dt; rp.c = 1.0; rp.qm = -1.0;
    rp.B0[0] = 0.2f; rp.wce = 0.2;
    rp.ppc = ppc; rp.noisy_load = true; rp.rng_seed = 424242;
    rp.weight = dx * g.dy / ppc;                  // n0 = 1
    rp.vth = 0.15;
    rp.rsm = 1; rp.rsm_k1 = k1;

    MaxwellSimulation sim(g, rp);
    // isotropic warm load through the legacy loader, then randomize θ (R1)
    RunParams rpl = rp;
    rpl.vth = 0.15;
    sim.particles().initialize(rpl, g, sim.stream());
    rsm_theta_init(sim.particles(), rp, sim.stream());
    {   // small div-free m1 seed
        std::mt19937 gen(7u);
        std::uniform_real_distribution<float> un(-1e-5f, 1e-5f);
        std::vector<float2> bz(nx);
        for (auto& z : bz) z = float2{un(gen), un(gen)};
        CUDA_CHECK(cudaMemcpy(sim.rsm().b1z.data(), bz.data(), nx * 8,
                              cudaMemcpyHostToDevice));
    }
    for (int n = 0; n < 3; ++n) sim.step();       // settle transients
    CUDA_CHECK(cudaDeviceSynchronize());

    RsmState& r = sim.rsm();
    std::vector<float2> rho_a(nx), rho_b(nx), j1x(nx), j1y(nx);
    rsm_deposit_moments(sim.particles(), r, rp, sim.stream());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(rho_a.data(), r.rho1.data(), nx * 8, cudaMemcpyDeviceToHost));

    sim.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    // save the step's J1 BEFORE the moment deposit clobbers the arrays
    CUDA_CHECK(cudaMemcpy(j1x.data(), r.j1x.data(), nx * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(j1y.data(), r.j1y.data(), nx * 8, cudaMemcpyDeviceToHost));
    rsm_deposit_moments(sim.particles(), r, rp, sim.stream());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(rho_b.data(), r.rho1.data(), nx * 8, cudaMemcpyDeviceToHost));

    double res2 = 0, drho2 = 0, djx2 = 0, djy2 = 0;
    for (int i = 0; i < nx; ++i) {
        const int im = (i + nx - 1) % nx;
        const double drho_re = (rho_b[i].x - rho_a[i].x) / dt;
        const double drho_im = (rho_b[i].y - rho_a[i].y) / dt;
        const double djx_re = (j1x[i].x - j1x[im].x) / dx;
        const double djx_im = (j1x[i].y - j1x[im].y) / dx;
        const double kj_re = -k1 * j1y[i].y;      // i·k1·J1y
        const double kj_im =  k1 * j1y[i].x;
        const double rr = drho_re + djx_re + kj_re;
        const double ri = drho_im + djx_im + kj_im;
        res2  += rr * rr + ri * ri;
        drho2 += drho_re * drho_re + drho_im * drho_im;
        djx2  += djx_re * djx_re + djx_im * djx_im;
        djy2  += kj_re * kj_re + kj_im * kj_im;
    }
    const double scale = std::sqrt(std::max(std::max(drho2, djx2), djy2));
    const double rel = std::sqrt(res2) / scale;
    std::printf("B continuity: rms residual / rms largest term = %.3e\n", rel);
    if (rel > 1e-3) { std::printf("B FAIL\n"); return false; }

    // C: div B1 after 200 more steps (seeded div-free)
    for (int n = 0; n < 200; ++n) sim.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float2> b1x(nx), b1y(nx);
    CUDA_CHECK(cudaMemcpy(b1x.data(), r.b1x.data(), nx * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b1y.data(), r.b1y.data(), nx * 8, cudaMemcpyDeviceToHost));
    double div2 = 0, b2 = 0;
    for (int i = 0; i < nx; ++i) {
        const int ip = (i + 1) % nx;
        const double dr = (b1x[ip].x - b1x[i].x) / dx - k1 * b1y[i].y;
        const double di = (b1x[ip].y - b1x[i].y) / dx + k1 * b1y[i].x;
        div2 += dr * dr + di * di;
        b2 += (double)b1x[i].x * b1x[i].x + (double)b1x[i].y * b1x[i].y
            + (double)b1y[i].x * b1y[i].x + (double)b1y[i].y * b1y[i].y;
    }
    const double dnorm = std::sqrt(div2) / (k1 * std::sqrt(std::max(b2, 1e-300)) + 1e-300);
    std::printf("C divB1: |div B1| / (k1·|B1|) = %.3e\n", dnorm);
    if (dnorm > 1e-4) { std::printf("C FAIL\n"); return false; }
    return true;
}

// ---- D: energy ledger through the anisotropy instability -------------------
bool test_energy() {
    const double k1 = 0.4, dt = 0.02, dx = 0.5;
    const int nx = 128, ppc = 400;
    Grid g(nx, 1, nx * dx, 2.0 * M_PI / k1);
    RunParams rp;
    rp.dt = dt; rp.c = 1.0; rp.qm = -1.0;
    rp.B0[0] = 0.25f; rp.wce = 0.25;
    rp.ppc = ppc; rp.noisy_load = true; rp.rng_seed = 991;
    rp.weight = dx * g.dy / ppc;
    rp.rsm = 1; rp.rsm_k1 = k1;

    MaxwellSimulation sim(g, rp);
    // anisotropic bi-Maxwellian: T_perp/T_par = 12 → whistlers grow m0 AND m1
    {
        RunParams rpl = rp;
        rpl.vth = 0.1;                            // parallel (x)
        sim.particles().initialize(rpl, g, sim.stream());
        // stretch the perpendicular components on the host (loader is isotropic)
        const std::size_t N = sim.particles().n;
        std::vector<float> u(N);
        for (auto* arr : { &sim.particles().uy, &sim.particles().uz }) {
            CUDA_CHECK(cudaMemcpy(u.data(), arr->data(), N * 4, cudaMemcpyDeviceToHost));
            for (auto& x : u) x *= 3.5f;          // uth_perp = 0.35
            CUDA_CHECK(cudaMemcpy(arr->data(), u.data(), N * 4, cudaMemcpyHostToDevice));
        }
    }
    rsm_theta_init(sim.particles(), rp, sim.stream());
    {   // m1 seed above shot noise so both harmonics engage early
        std::mt19937 gen(13u);
        std::uniform_real_distribution<float> un(-1e-4f, 1e-4f);
        std::vector<float2> bz(nx);
        for (auto& z : bz) z = float2{un(gen), un(gen)};
        CUDA_CHECK(cudaMemcpy(sim.rsm().b1z.data(), bz.data(), nx * 8,
                              cudaMemcpyHostToDevice));
    }

    const std::size_t N = sim.particles().n;
    const double dV = dx * g.dy;
    std::vector<float> hu(N), hv(N), hw(N), hwt(N);
    std::vector<float> f0(g.real_size());
    std::vector<float2> h1(nx);
    CUDA_CHECK(cudaMemcpy(hwt.data(), sim.particles().w.data(), N * 4,
                          cudaMemcpyDeviceToHost));

    auto energies = [&](double& ke, double& w0, double& w1) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hu.data(), sim.particles().ux.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hv.data(), sim.particles().uy.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hw.data(), sim.particles().uz.data(), N * 4, cudaMemcpyDeviceToHost));
        ke = 0;
        for (std::size_t i = 0; i < N; ++i)
            ke += 0.5 * hwt[i] * ((double)hu[i] * hu[i] + (double)hv[i] * hv[i]
                                  + (double)hw[i] * hw[i]);
        w0 = 0;
        YeeFields& f = sim.fields();
        const double c2 = rp.c * rp.c;
        for (auto* arr : { &f.ex_, &f.ey_, &f.ez_ }) {
            CUDA_CHECK(cudaMemcpy(f0.data(), arr->data(), f0.size() * 4, cudaMemcpyDeviceToHost));
            for (float x : f0) w0 += 0.5 * (double)x * x * dV;
        }
        for (auto* arr : { &f.bx_, &f.by_, &f.bz_ }) {
            CUDA_CHECK(cudaMemcpy(f0.data(), arr->data(), f0.size() * 4, cudaMemcpyDeviceToHost));
            for (float x : f0) w0 += 0.5 * c2 * (double)x * x * dV;
        }
        w1 = 0;
        RsmState& r = sim.rsm();
        for (auto* arr : { &r.e1x, &r.e1y, &r.e1z })
            w1 += 0.5 * copy_sum_sq(*arr, h1) * dV;
        for (auto* arr : { &r.b1x, &r.b1y, &r.b1z })
            w1 += 0.5 * c2 * copy_sum_sq(*arr, h1) * dV;
        w1 *= 2.0;                                // the ±k1 pair (R6 ledger)
    };

    double ke0, wm0_0, wm1_0;
    energies(ke0, wm0_0, wm1_0);
    const double wtot0 = ke0 + wm0_0 + wm1_0;

    const int nsteps = 15000, every = 500;
    double worst = 0, transfer = 0, w1max = 0;
    for (int n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n % every) continue;
        double ke, w0, w1;
        energies(ke, w0, w1);
        worst    = std::max(worst, std::fabs(ke + w0 + w1 - wtot0));
        transfer = std::max(transfer, std::fabs(ke - ke0));
        w1max    = std::max(w1max, w1);
        if (n % 2500 == 0)
            std::printf("D t=%6.0f  KE=%.8e  Wm0=%.3e  2Wm1=%.3e  "
                        "dWtot/W1max=%.3f\n", n * dt, ke, w0, w1,
                        (ke + w0 + w1 - wtot0) / std::max(w1max, 1e-30));
    }
    const double denom = std::max(std::max(transfer, w1max), 1e-30);
    std::printf("D ledger: worst |dWtot| = %.3e, transferred = %.3e, "
                "W1max = %.3e, ratio = %.4f\n", worst, transfer, w1max,
                worst / denom);
    if (w1max < 10.0 * wm1_0) {
        std::printf("D WARNING: m1 never grew (w1max %.2e vs seed %.2e) — "
                    "ledger not exercised\n", w1max, wm1_0);
    }
    if (worst / denom > 0.10) { std::printf("D FAIL\n"); return false; }
    return true;
}

} // namespace

int main() {
    bool ok = test_kick();
    ok = test_continuity() && ok;
    ok = test_energy() && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
