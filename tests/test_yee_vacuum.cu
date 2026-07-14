// M1 gate [yee_vacuum_dispersion]: vacuum EM waves on the 2D Yee grid must
// oscillate at the YEE dispersion relation
//   sin²(ωΔt/2)/(cΔt)² = sin²(kxΔx/2)/Δx² + sin²(kyΔy/2)/Δy²
// (which → ω = c|k| as Δ→0). A standing Ez mode Ez = cos(kx x + ky y) splits
// into ± travelling waves; the projected amplitude A(t) oscillates as cos(ωt).
// ω is measured from zero crossings over ~40 periods; tolerance 3e-4 relative
// (float fields). Checked for an axis mode and an oblique mode.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace arc;

static double run_mode(int mx, int my, int nx, int ny, double dx, double dy,
                       double c, double dt, int nsteps) {
    Grid g(nx, ny, nx * dx, ny * dy);
    RunParams rp; rp.c = c; rp.dt = dt; rp.qm = -1.0;
    MaxwellSimulation sim(g, rp);

    const double kx = 2.0 * M_PI * mx / (nx * dx);
    const double ky = 2.0 * M_PI * my / (ny * dy);

    // init Ez = cos(kx x + ky y) on Ez sites (nodes)
    std::vector<float> ez(g.real_size());
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            ez[j * nx + i] = (float)std::cos(kx * i * dx + ky * j * dy);
    CUDA_CHECK(cudaMemcpy(sim.fields().ez_.data(), ez.data(),
                          ez.size() * 4, cudaMemcpyHostToDevice));

    // project A(t) = Σ Ez cos(kx x + ky y); count zero crossings
    std::vector<float> h(g.real_size());
    double prev = 0; int ncross = 0; double t_first = 0, t_last = 0;
    for (int n = 1; n <= nsteps; ++n) {
        sim.step();
        CUDA_CHECK(cudaMemcpy(h.data(), sim.fields().ez_.data(),
                              h.size() * 4, cudaMemcpyDeviceToHost));
        double A = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                A += h[j * nx + i] * std::cos(kx * i * dx + ky * j * dy);
        if (n > 1 && ((prev < 0 && A >= 0) || (prev > 0 && A <= 0))) {
            const double f = prev / (prev - A);
            const double tc = (n - 1 + f) * dt;
            if (!ncross) t_first = tc;
            t_last = tc;
            ++ncross;
        }
        prev = A;
    }
    return M_PI * (ncross - 1) / (t_last - t_first);
}

static double w_yee(int mx, int my, int nx, int ny, double dx, double dy,
                    double c, double dt) {
    const double kx = 2.0 * M_PI * mx / (nx * dx);
    const double ky = 2.0 * M_PI * my / (ny * dy);
    const double s2 = std::pow(std::sin(0.5 * kx * dx) / dx, 2)
                    + std::pow(std::sin(0.5 * ky * dy) / dy, 2);
    return 2.0 / dt * std::asin(c * dt * std::sqrt(s2));
}

int main() {
    bool ok = true;
    struct Case { int mx, my; } cases[] = {{3, 0}, {2, 5}};
    const int nx = 64, ny = 64;
    const double dx = 1.0, dy = 0.8, c = 5.0;
    const double dt = 0.4 / (c * std::sqrt(1 / (dx * dx) + 1 / (dy * dy)));

    for (auto cs : cases) {
        const double wth = w_yee(cs.mx, cs.my, nx, ny, dx, dy, c, dt);
        const int nsteps = (int)(40.0 * 2.0 * M_PI / wth / dt);
        const double wm = run_mode(cs.mx, cs.my, nx, ny, dx, dy, c, dt, nsteps);
        const double kmag = std::hypot(2 * M_PI * cs.mx / (nx * dx),
                                       2 * M_PI * cs.my / (ny * dy));
        const double err = std::fabs(wm - wth) / wth;
        std::printf("mode (%d,%d): w_meas=%.6f w_yee=%.6f (ck=%.6f) err=%.2e\n",
                    cs.mx, cs.my, wm, wth, c * kmag, err);
        if (err > 3e-4) { std::printf("FAIL: dispersion\n"); ok = false; }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
