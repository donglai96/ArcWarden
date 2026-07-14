// M1 gate [yee_langmuir]: cold Langmuir oscillation on the FULL-MAXWELL path,
// the same physics the spectral ES test validates — dual-path check.
// Cold electrons on a quiet lattice, velocity perturbation ux = v0 sin(kx x),
// E = B = 0 initially: the charge-conserving current makes Ex develop exactly
// as Poisson would give, and the mode oscillates at w = wpe (= 1 in code
// units, n0 = 1, |q| = m = 1). Measured from mode-amplitude zero crossings.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int nx = 64, ny = 8, ppc_side = 8;          // 64 particles/cell, quiet
    const double dx = 1.0, dy = 1.0, c = 10.0;
    Grid g(nx, ny, nx * dx, ny * dy);
    RunParams rp; rp.qm = -1.0; rp.c = c;
    rp.dt = 0.4 / (c * std::sqrt(2.0));               // CFL 0.4
    MaxwellSimulation sim(g, rp);

    const double kx = 2.0 * M_PI / (nx * dx);
    const double v0 = 0.01;

    // quiet ppc_side × ppc_side sub-lattice per cell, w = dx dy / ppc (n0 = 1)
    const int ppc = ppc_side * ppc_side;
    const int N = nx * ny * ppc;
    Particles& p = sim.particles();
    p.allocate_n(N);
    std::vector<float> hx(N), hy(N), hux(N), huy(N, 0), huz(N, 0), hw(N);
    std::vector<int> hcell(N, 0);
    int t = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            for (int b = 0; b < ppc_side; ++b)
                for (int a = 0; a < ppc_side; ++a, ++t) {
                    hx[t] = i + (a + 0.5f) / ppc_side;
                    hy[t] = j + (b + 0.5f) / ppc_side;
                    hux[t] = (float)(v0 * std::sin(kx * hx[t] * dx));
                    hw[t] = (float)(dx * dy / ppc);
                }
    CUDA_CHECK(cudaMemcpy(p.x.data(),  hx.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.y.data(),  hy.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.ux.data(), hux.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.uy.data(), huy.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.uz.data(), huz.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.w.data(),  hw.data(),  N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.cell.data(), hcell.data(), N * 4, cudaMemcpyHostToDevice));

    // measure Ex mode-1 amplitude
    std::vector<float> hex(g.real_size());
    const int nsteps = (int)(10.0 * 2.0 * M_PI / rp.dt);   // ~10 plasma periods
    double prev = 0; int ncross = 0; double t_first = 0, t_last = 0;
    for (int n = 1; n <= nsteps; ++n) {
        sim.step();
        CUDA_CHECK(cudaMemcpy(hex.data(), sim.fields().ex_.data(),
                              hex.size() * 4, cudaMemcpyDeviceToHost));
        double A = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                A += hex[j * nx + i] * std::sin(kx * (i + 0.5) * dx);  // Ex at i+1/2
        if (n > 1 && ((prev < 0 && A >= 0) || (prev > 0 && A <= 0))) {
            const double f = prev / (prev - A);
            const double tc = (n - 1 + f) * rp.dt;
            if (!ncross) t_first = tc;
            t_last = tc; ++ncross;
        }
        prev = A;
    }
    if (ncross < 5) { std::printf("FAIL: only %d crossings\n", ncross); return 1; }
    const double w_meas = M_PI * (ncross - 1) / (t_last - t_first);
    const double err = std::fabs(w_meas - 1.0);
    std::printf("w_meas = %.5f (wpe = 1)  err = %.2e  [%d crossings]\n",
                w_meas, err, ncross);
    const bool ok = err < 0.02;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
