// RSM gate V1 — phase load (docs/RSM_MODEL_DEFINITION.md §6.1 R1, ladder V1).
//
// The trap: every ny = 1 loader pins y = 0.5, so a direct e^{−iθ} deposit off
// the load is COHERENT — |Σ_p e^{−iθ_p}| = N, a macroscopic fake oblique seed.
// The remedy: rsm_theta_init randomizes θ on its own RNG stream. This gate
//   (a) DEMONSTRATES the trap: pinned y → |Σ e^{−iθ}| ≥ 0.999·N;
//   (b) verifies zero-mean shot noise: over K seeds, E|Σ e^{−iθ}|²/N = 1
//       (complex-Gaussian sum ⇒ |S|²/N ~ Exp(1); K = 32 ⇒ ±3σ ≈ [0.5, 1.6]);
//   (c) verifies the N^{−1/2} law: the same normalized power at 4N;
//   (d) same for the velocity moment J1z: E|Σ u_z e^{−iθ}|² = N·⟨u_z²⟩
//       (θ independent of the velocity draw).
// Totals are recovered as Σ_i F1(x_i)·Δx·Δy = Σ_p qw (v_p) e^{−iθ_p}, which
// also exercises the CIC x-scatter + volume normalization round trip.

#include "pic/config.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/rsm_oblique.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

namespace {

struct Totals { double rho_re, rho_im, jz_re, jz_im; };

Totals moment_totals(RsmState& r) {
    std::vector<float2> hrho(r.nx), hjz(r.nx);
    CUDA_CHECK(cudaMemcpy(hrho.data(), r.rho1.data(), r.nx * sizeof(float2),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hjz.data(), r.j1z.data(), r.nx * sizeof(float2),
                          cudaMemcpyDeviceToHost));
    Totals t{0, 0, 0, 0};
    const double dV = r.dxp * r.dyp;
    for (int i = 0; i < r.nx; ++i) {
        t.rho_re += dV * hrho[i].x; t.rho_im += dV * hrho[i].y;
        t.jz_re  += dV * hjz[i].x;  t.jz_im  += dV * hjz[i].y;
    }
    return t;
}

// Mean normalized modal power over nseeds independent θ draws.
void seed_sweep(Particles& parts, RsmState& r, RunParams rp, int nseeds,
                double uz2, double& p_rho, double& p_jz) {
    const double N = (double)parts.n;
    p_rho = p_jz = 0.0;
    for (int k = 0; k < nseeds; ++k) {
        rp.rng_seed = 1000 + 7919UL * k;
        rsm_theta_init(parts, rp, 0);
        rsm_deposit_moments(parts, r, rp, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const Totals t = moment_totals(r);
        p_rho += (t.rho_re * t.rho_re + t.rho_im * t.rho_im) / N;
        p_jz  += (t.jz_re * t.jz_re + t.jz_im * t.jz_im) / (N * uz2);
    }
    p_rho /= nseeds; p_jz /= nseeds;
}

int run_size(std::size_t N, const Grid& g, RunParams rp, bool check_trap) {
    Particles parts;
    parts.allocate_n(N);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> px(0.f, (float)g.nx);
    std::normal_distribution<float> pv(0.f, 0.2f);
    std::vector<float> hx(N), hy(N, 0.5f), hu(N), hv(N), hw(N), hwt(N, 1.f);
    std::vector<int>   hc(N, 0);
    double uz2 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        hx[i] = px(gen); hu[i] = pv(gen); hv[i] = pv(gen); hw[i] = pv(gen);
        uz2 += (double)hw[i] * hw[i];
    }
    uz2 /= (double)N;
    CUDA_CHECK(cudaMemcpy(parts.x.data(),  hx.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.y.data(),  hy.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.ux.data(), hu.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.uy.data(), hv.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.uz.data(), hw.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.w.data(),  hwt.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.cell.data(), hc.data(), N * 4, cudaMemcpyHostToDevice));

    RsmState rsm;
    rsm.init(g, rp, 0);

    int fails = 0;
    if (check_trap) {
        // (a) pinned y = 0.5 (the loader state) → coherent, |S| = N
        rsm_deposit_moments(parts, rsm, rp, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const Totals t = moment_totals(rsm);
        const double coh = std::hypot(t.rho_re, t.rho_im) / (double)N;
        std::printf("  pinned-load coherence |S|/N = %.6f (trap demo, want ~1)\n", coh);
        if (coh < 0.999) { std::printf("  FAIL: pinned load should be coherent\n"); ++fails; }
    }

    // (b)/(d) randomized θ → normalized modal power = 1 over seeds
    double p_rho, p_jz;
    seed_sweep(parts, rsm, rp, 32, uz2, p_rho, p_jz);
    std::printf("  N = %zu: E|S_rho|^2/N = %.3f, E|S_jz|^2/(N<uz^2>) = %.3f "
                "(want 1.0 +/- 3sigma = [0.5, 1.6])\n", N, p_rho, p_jz);
    if (p_rho < 0.5 || p_rho > 1.6) { std::printf("  FAIL: rho1 shot-noise power off\n"); ++fails; }
    if (p_jz  < 0.5 || p_jz  > 1.6) { std::printf("  FAIL: j1z shot-noise power off\n"); ++fails; }
    return fails;
}

} // namespace

int main() {
    const int    nx = 64;
    const double k1 = 0.16;                       // 15-deg-class k_perp
    const double Ly = 2.0 * M_PI / k1;            // the RSM deck contract
    Grid g(nx, 1, nx * 0.26, Ly);

    RunParams rp;
    rp.qm = -1.0; rp.dt = 0.1;
    rp.rsm = 1; rp.rsm_k1 = 2.0 * M_PI / Ly;

    int fails = 0;
    std::printf("V1 phase-load gate (theta = 2*pi*y, Ly = 2*pi/k1):\n");
    fails += run_size(1u << 18, g, rp, true);      // trap demo + shot noise
    fails += run_size(1u << 20, g, rp, false);     // (c) N^{-1/2} law at 4N

    std::printf(fails ? "test_rsm_phaseload: FAIL (%d)\n"
                      : "test_rsm_phaseload: PASS\n", fails);
    return fails ? 1 : 0;
}
