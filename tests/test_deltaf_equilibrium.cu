// M3 gate [deltaf_equilibrium_hold]: an UNPERTURBED delta-f load on a stable
// (isotropic Maxwellian) equilibrium must not develop systematic weight
// growth — the classic trap that exposes sign/force errors in the weight
// equation. Quiet start, wd(0) = 0, no seed: fields stay at deposit noise
// (which is itself wd-scaled, so near machine floor), and wd may only jitter
// at the noise level driven by the tiny fields.
//
// A small random transverse-B seed (1e-6) closes the loop: the stable plasma
// supports propagating whistlers/oscillations, wd responds linearly and must
// stay BOUNDED — no secular growth (second half vs first half), on both the
// flat and tiled deposit paths. (With zero seed everything stays exactly 0,
// which only catches a B0-leaking-into-F class of bug.)

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

// rms + max of wd (host reduce; test-sized runs)
static void wd_stats(Particles& P, double& rms, double& mx) {
    const size_t n = P.n;
    std::vector<float> wd(n);
    CUDA_CHECK(cudaMemcpy(wd.data(), P.wd.data(), n * 4, cudaMemcpyDeviceToHost));
    double s2 = 0; mx = 0;
    for (float v : wd) { s2 += (double)v * v; mx = std::max(mx, (double)std::fabs(v)); }
    rms = std::sqrt(s2 / n);
}

static bool run_path(int tile_sort) {
    const int nx = 64, ny = 16, ppc = 100;
    const double c = 10.0, wce = 0.5, uth = 0.05, dx = 0.1;
    Grid g(nx, ny, nx * dx, ny * dx);
    RunParams rp;
    rp.dt = 0.4 * dx / c; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.noisy_load = false; rp.dump_every = 0;
    rp.tile_sort = tile_sort;
    rp.deltaf = 1; rp.df_tpar = uth * uth; rp.df_tperp = uth * uth;

    SpeciesList sp = { Species{"e", 1.0, ppc, {uth, uth, uth}, {0, 0, 0}, true} };
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    sim.particles().enable_deltaf(sim.stream());
    sim.stream().synchronize();

    // small transverse-B seed so the loop actually runs (stable -> bounded wd)
    std::mt19937 gen(7u);
    std::uniform_real_distribution<float> un(-1e-6f, 1e-6f);
    std::vector<float> seed(g.real_size());
    for (auto& v : seed) v = un(gen);
    CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), seed.data(), seed.size() * 4, cudaMemcpyHostToDevice));
    for (auto& v : seed) v = un(gen);
    CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), seed.data(), seed.size() * 4, cudaMemcpyHostToDevice));

    const long nsteps = 4000;
    double rms_half = 0, mx_half = 0, rms_end = 0, mx_end = 0;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n == nsteps / 2) { cudaDeviceSynchronize(); wd_stats(sim.particles(), rms_half, mx_half); }
    }
    cudaDeviceSynchronize();
    wd_stats(sim.particles(), rms_end, mx_end);

    const double growth = rms_half > 0 ? rms_end / rms_half : 1.0;
    std::printf("  %s path: rms(wd) %.3e -> %.3e (x%.2f)  max|wd| %.3e -> %.3e\n",
                tile_sort ? "tiled" : "flat", rms_half, rms_end, growth, mx_half, mx_end);
    bool ok = true;
    if (mx_end > 1e-2)   { std::printf("  FAIL: max|wd| %.3e > 1e-2\n", mx_end); ok = false; }
    if (growth > 3.0)    { std::printf("  FAIL: rms(wd) grew x%.2f > 3 in the second half\n", growth); ok = false; }
    return ok;
}

int main() {
    std::printf("deltaf equilibrium hold (isotropic Maxwellian, quiet, no seed):\n");
    bool ok = run_path(0);
    ok = run_path(20) && ok;
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
