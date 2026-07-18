// M1 gate [esirkepov_continuity]: the Esirkepov deposit must satisfy the
// DISCRETE continuity equation exactly (to float roundoff):
//   (ρ^{n+1} - ρ^n)/Δt + (Jx(i+½,j)-Jx(i-½,j))/Δx + (Jy(i,j+½)-Jy(i,j-½))/Δy = 0
// with ρ the CIC node charge. 20k particles with random positions and
// velocities (|v Δt| up to 0.9 cell, all three components, non-square cells),
// one fused push+deposit step in zero fields (straight-line motion).

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

int main() {
    const int nx = 64, ny = 48;
    const double dx = 0.7, dy = 1.3;
    Grid g(nx, ny, nx * dx, ny * dy);
    RunParams rp; rp.qm = -1.0; rp.c = 5.0; rp.dt = 0.9 * dx;  // vmax ~ 1 cell/step at v=1
    MaxwellSimulation sim(g, rp);

    const int N = 20000;
    std::mt19937 rng(20260713);
    std::uniform_real_distribution<float> ux_(0, (float)nx), uy_(0, (float)ny);
    std::uniform_real_distribution<float> uv(-1.f, 1.f);

    Particles& p = sim.particles();
    p.allocate_n(N);
    std::vector<float> hx(N), hy(N), hux(N), huy(N), huz(N), hw(N);
    std::vector<int> hcell(N, 0);
    for (int t = 0; t < N; ++t) {
        hx[t] = ux_(rng); hy[t] = uy_(rng);
        hux[t] = uv(rng) * 0.7f; huy[t] = uv(rng) * 1.3f; huz[t] = uv(rng);
        hw[t] = 0.5f + 0.5f * (t % 7) / 7.0f;      // non-uniform weights
    }
    CUDA_CHECK(cudaMemcpy(p.x.data(),  hx.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.y.data(),  hy.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.ux.data(), hux.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.uy.data(), huy.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.uz.data(), huz.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.w.data(),  hw.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.cell.data(), hcell.data(), N * 4, cudaMemcpyHostToDevice));

    YeeViews v = sim.fields().views();
    DeviceArray<float> rho0(g.real_size()), rho1(g.real_size());
    rho0.zero(); rho1.zero();

    yee::k_rho_nodes<<<(N + 255) / 256, 256>>>(p.views(), v, rho0.data(), rp.qm, 0);
    sim.fields().zero_j();
    yee::k_push_esirkepov<<<(N + 255) / 256, 256>>>(p.views(), v, rp, 0.0);
    yee::k_rho_nodes<<<(N + 255) / 256, 256>>>(p.views(), v, rho1.data(), rp.qm, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> r0(g.real_size()), r1(g.real_size()),
                       jx(g.real_size()), jy(g.real_size());
    CUDA_CHECK(cudaMemcpy(r0.data(), rho0.data(), r0.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r1.data(), rho1.data(), r1.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(jx.data(), sim.fields().jx_.data(), jx.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(jy.data(), sim.fields().jy_.data(), jy.size() * 4, cudaMemcpyDeviceToHost));

    auto at = [&](std::vector<float>& a, int i, int j) -> float {
        return a[((j + ny) % ny) * nx + ((i + nx) % nx)];
    };
    double max_res = 0, max_rdot = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double rdot = (at(r1, i, j) - at(r0, i, j)) / rp.dt;
            const double divj = (at(jx, i, j) - at(jx, i - 1, j)) / dx
                              + (at(jy, i, j) - at(jy, i, j - 1)) / dy;
            max_res = std::fmax(max_res, std::fabs(rdot + divj));
            max_rdot = std::fmax(max_rdot, std::fabs(rdot));
        }
    const double rel = max_res / max_rdot;
    std::printf("max |drho/dt + div J| = %.3e ; max|drho/dt| = %.3e ; rel = %.3e\n",
                max_res, max_rdot, rel);
    const bool ok = rel < 5e-5;      // float atomics roundoff scale
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
