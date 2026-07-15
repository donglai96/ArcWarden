// M9 gate [yee_tiled_deposit]: the tiled shared-memory Esirkepov path
// (tile_sort > 0) must produce the same J as the flat global-atomic kernel to
// atomic-order float roundoff, including stray particles that drifted beyond
// the tile apron between sorts, and must keep the discrete continuity residual
// at roundoff over a multi-step run with a stale sort.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

static void load_random(Particles& p, int N, int nx, int ny, unsigned seed) {
    p.allocate_n(N);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux_(0, (float)nx), uy_(0, (float)ny);
    std::uniform_real_distribution<float> uv(-1.f, 1.f);
    std::vector<float> hx(N), hy(N), hux(N), huy(N), huz(N), hw(N);
    std::vector<int> hcell(N);
    for (int t = 0; t < N; ++t) {
        hx[t] = ux_(rng); hy[t] = uy_(rng);
        // |u| ≤ 1.21 keeps Δx < 0.77 cell/step EVEN after B0 rotates the full
        // vector into x (Esirkepov needs < 1 cell/step in every direction);
        // ~0.77 cell/step × 4 stale steps ≈ 3 cells > the DRIFT=2 apron, so
        // the stray fallback is exercised.
        hux[t] = uv(rng) * 0.7f; huy[t] = uv(rng) * 0.7f; huz[t] = uv(rng) * 0.7f;
        // scale weights so the loaded density is ~1: full steps EVOLVE the
        // fields, and unresolved plasma oscillations (ωpe·dt ≥ 2) would grow
        // velocities past the 1-cell/step Esirkepov bound within a few steps
        hw[t] = (0.5f + 0.5f * (t % 7) / 7.0f) * (float)(nx * (double)ny) / N;
        hcell[t] = (int)hy[t] * nx + (int)hx[t];
    }
    CUDA_CHECK(cudaMemcpy(p.x.data(),  hx.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.y.data(),  hy.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.ux.data(), hux.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.uy.data(), huy.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.uz.data(), huz.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.w.data(),  hw.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.cell.data(), hcell.data(), N * 4, cudaMemcpyHostToDevice));
}

static std::vector<float> host(DeviceArray<float>& a) {
    std::vector<float> h(a.size());
    CUDA_CHECK(cudaMemcpy(h.data(), a.data(), a.bytes(), cudaMemcpyDeviceToHost));
    return h;
}

int main() {
    // 100x60: nx NOT divisible by 16 -> a 4-cell edge-tile column (ntx = 7)
    const int nx = 100, ny = 60, N = 200000;
    const double dx = 0.7, dy = 1.3;
    Grid g(nx, ny, nx * dx, ny * dy);
    // full steps evolve the Yee fields, so the LIGHT CFL must hold:
    // c·dt < dx/√2 (the flat-path continuity test dodges this by calling the
    // deposit kernel alone; here dt = 0.9·dx/(c·√2)).
    RunParams rp; rp.qm = -1.0; rp.c = 1.0;
    rp.dt = 0.9 * dx / (rp.c * std::sqrt(2.0));
    rp.B0[0] = 0.3f; rp.B0[2] = 0.2f;

    int fails = 0;

    // ---- gate 1: one-step J identity, flat vs tiled (fresh sort) ----
    RunParams rpf = rp; rpf.tile_sort = 0;
    RunParams rpt = rp; rpt.tile_sort = 5;   // sort at step 0; steps 1-4 stale
    MaxwellSimulation simf(g, rpf), simt(g, rpt);
    load_random(simf.particles(), N, nx, ny, 20260714);
    load_random(simt.particles(), N, nx, ny, 20260714);
    simf.step(); simt.step();
    CUDA_CHECK(cudaDeviceSynchronize());

    const char* nm[3] = {"Jx", "Jy", "Jz"};
    DeviceArray<float>* jf[3] = {&simf.fields().jx_, &simf.fields().jy_, &simf.fields().jz_};
    DeviceArray<float>* jt[3] = {&simt.fields().jx_, &simt.fields().jy_, &simt.fields().jz_};
    for (int c = 0; c < 3; ++c) {
        auto a = host(*jf[c]), b = host(*jt[c]);
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            num += (a[i] - b[i]) * (double)(a[i] - b[i]);
            den += a[i] * (double)a[i];
        }
        const double rel = std::sqrt(num / (den + 1e-300));
        std::printf("[yee_tiled] one-step %s rel diff = %.3e (tol 5e-5)\n", nm[c], rel);
        if (!(rel < 5e-5)) ++fails;
    }

    // ---- gate 2: continuity residual with a STALE sort (strays exercised) ----
    // the measured step is the 4th since the sort: drift up to ~3 cells >
    // DRIFT=2 apron, so the global-atomic fallback must carry the strays.
    YeeViews v = simt.fields().views();
    Particles& p = simt.particles();
    DeviceArray<float> rho0(g.real_size()), rho1(g.real_size());
    for (int s = 0; s < 3; ++s) simt.step();          // stale-ify the sort
    CUDA_CHECK(cudaDeviceSynchronize());
    rho0.zero(); rho1.zero();
    yee::k_rho_nodes<<<((int)p.n + 255) / 256, 256>>>(p.views(), v, rho0.data(), rp.qm);
    simt.step();                                       // deposits J^{n+1/2}
    CUDA_CHECK(cudaDeviceSynchronize());
    yee::k_rho_nodes<<<((int)p.n + 255) / 256, 256>>>(p.views(), v, rho1.data(), rp.qm);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto r0 = host(rho0), r1 = host(rho1);
    auto jx = host(simt.fields().jx_), jy = host(simt.fields().jy_);
    double worst = 0, scale = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int c  = j * nx + i;
            const int im = j * nx + (i + nx - 1) % nx;
            const int jm = ((j + ny - 1) % ny) * nx + i;
            const double res = (r1[c] - r0[c]) / rp.dt
                             + (jx[c] - jx[im]) / dx + (jy[c] - jy[jm]) / dy;
            worst = std::max(worst, std::abs(res));
            scale = std::max(scale, std::abs(r1[c] - r0[c]) / rp.dt);
        }
    const double rel = worst / (scale + 1e-300);
    std::printf("[yee_tiled] stale-sort continuity residual = %.3e (tol 2e-4)\n", rel);
    if (!(rel < 2e-4)) ++fails;

    std::printf(fails ? "[yee_tiled] FAIL (%d)\n" : "[yee_tiled] PASS\n", fails);
    return fails ? 1 : 0;
}
