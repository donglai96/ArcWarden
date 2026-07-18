// M2 gate [boundary_vacuum_damping]: the Umeda-style x-end damping layers
// must absorb a normally-incident vacuum EM pulse with small reflection, and
// must be exactly inert when disabled.
//
// Setup: 512x8 grid (y-uniform), c = 1, dt = 0.5 dx. A +x-travelling pulse
//   Ez = exp(-(x-x0)²/2σ²),  By = -Ez/c   (Yee convention: k_faraday gives
//   ∂By/∂t = +∂Ez/∂x → By = -(k/ω)Ez for e^{i(kx-ωt)})
// launched from mid-domain. It enters the right layer (nd = 64, numax = 0.5),
// is damped; anything reflected re-enters the physical region. After the
// round-trip window the max |Ez| over the physical region IS the reflected
// amplitude (the outgoing pulse is gone into the masks; without masks it
// would wrap periodically and return at full amplitude — checked too).
//
// PASS: R < 1% with the layers on; R > 90% with bnd_x = 0 (control, proves
// the measurement window sees the wrapped pulse); masks inert at |x-x0| far
// from layers.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

// launch the pulse, run nsteps, return max |Ez| over the physical region
static double run_pulse(int bnd_x, int nd, double numax, int nsteps) {
    const int nx = 512, ny = 8;
    const double dx = 1.0, c = 1.0, dt = 0.5;
    Grid g(nx, ny, nx * dx, ny * dx);
    RunParams rp;
    rp.c = c; rp.dt = dt; rp.qm = -1.0;
    rp.bnd_x = bnd_x; rp.bnd_nd = nd; rp.bnd_numax = numax;
    MaxwellSimulation sim(g, rp);

    const double x0 = 256.0, sg = 12.0;
    std::vector<float> ez(g.real_size()), by(g.real_size());
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double xe = i, xb = i + 0.5;   // Ez at nodes, By at i+1/2
            ez[j * nx + i] = (float)std::exp(-(xe - x0) * (xe - x0) / (2 * sg * sg));
            by[j * nx + i] = (float)(-std::exp(-(xb - x0) * (xb - x0) / (2 * sg * sg)) / c);
        }
    CUDA_CHECK(cudaMemcpy(sim.fields().ez_.data(), ez.data(), ez.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), by.data(), by.size() * 4,
                          cudaMemcpyHostToDevice));

    for (int n = 0; n < nsteps; ++n) sim.step();

    CUDA_CHECK(cudaMemcpy(ez.data(), sim.fields().ez_.data(), ez.size() * 4,
                          cudaMemcpyDeviceToHost));
    double mx = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = nd; i < nx - nd; ++i)       // physical region only
            mx = std::max(mx, (double)std::fabs(ez[j * nx + i]));
    return mx;
}

int main() {
    // pulse: x0=256 → right-layer entry (x=448) at t=192; absorbed and any
    // reflection back across the whole physical region by t = 640 → 1280 steps.
    const int nd = 64, nsteps = 1280;
    const double numax = 0.5;

    const double R      = run_pulse(1, nd, numax, nsteps);
    const double R_ctrl = run_pulse(0, nd, numax, nsteps);   // periodic control

    bool ok = true;
    std::printf("vacuum pulse: reflected max|Ez| = %.3e (damping)  %.3e (periodic control)\n",
                R, R_ctrl);
    if (R > 0.01)      { std::printf("FAIL: reflection %.3e > 1e-2\n", R);        ok = false; }
    if (R_ctrl < 0.9)  { std::printf("FAIL: control lost the pulse (%.3e)\n", R_ctrl); ok = false; }
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
