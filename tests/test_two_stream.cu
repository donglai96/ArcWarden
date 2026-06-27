// ArcWarden — Step 14 test [S9]: two-stream instability.
//
// Two counter-streaming cold electron beams (±v0, total density n0) are unstable.
// For the most unstable mode, (k v0)² = (3/8) ω_pe², the growth rate is
//   γ_max = ω_pe / (2√2) ≈ 0.35355 ω_pe.
// With ωpe=1 we size the box so the fundamental mode k1 = 2π/Lx is that mode:
//   k1 v0 = √(3/8)  =>  Lx = 2π / √(3/8)  (with v0 = 1).
//
// Checks:
//   1. Field energy grows exponentially early; fitted γ ≈ 0.3536 (within ~15%).
//   2. Particle trapping: the fraction with |vx| < v0/2 grows from 0 (cold beams)
//      to a sizable value — the phase-space vortex / BGK hole signature.
//   3. Dumps (x, vx) at saturation to two_stream_phase.csv for visual inspection.
//
// Exits non-zero on failure.

#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace arc;

int main() {
    const double v0 = 1.0;
    const double kv = std::sqrt(3.0 / 8.0);     // most-unstable k*v0 (ωpe=1)
    const double Lx = 2.0 * M_PI / kv;          // fundamental == most unstable
    const int    nx = 64, ny = 4;
    Grid g(nx, ny, Lx, 1.0);

    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.vth = 0.0;                 // cold beams
    rp.vd  = v0;
    rp.two_stream = true;
    rp.ppc = 64;                  // 32 per beam per cell
    rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;
    rp.dt = 0.05;
    rp.nsteps = 600;              // ~ t = 30, several e-foldings + saturation
    rp.dump_every = 1;
    rp.perturb_amp = 0.005;
    rp.perturb_kx  = 1;

    const double gamma_analytic = 1.0 / (2.0 * std::sqrt(2.0));   // 0.35355

    Simulation<> sim(g, rp);
    sim.init();
    sim.run();

    const auto& H = sim.diagnostics().history();
    const int M = int(H.size());
    bool ok = true;

    // ---- 1. growth rate from ln(EE) in the linear phase ----
    std::vector<double> lnE(M);
    int isat = 0; double eemax = 0;
    for (int i = 0; i < M; ++i) {
        lnE[i] = std::log(std::max(H[i].ee, 1e-300));
        if (H[i].ee > eemax) { eemax = H[i].ee; isat = i; }
    }
    const int i0 = std::max(1, isat / 4), i1 = std::max(i0 + 5, (isat * 4) / 5);
    // least-squares slope of lnE vs time over [i0,i1]
    double sx=0, sy=0, sxx=0, sxy=0; int npt=0;
    for (int i = i0; i <= i1 && i < M; ++i) {
        const double x = i * rp.dt, y = lnE[i];
        sx += x; sy += y; sxx += x*x; sxy += x*y; ++npt;
    }
    const double slope = (npt*sxy - sx*sy) / (npt*sxx - sx*sx);
    const double gamma = slope / 2.0;           // EE ∝ exp(2γt)
    std::printf("two-stream: saturation step=%d  fit [%d,%d]  gamma=%.4f (analytic %.4f)\n",
                isat, i0, i1, gamma, gamma_analytic);
    if (std::fabs(gamma - gamma_analytic) > 0.06) { std::printf("FAIL: growth rate\n"); ok = false; }

    // ---- 2. trapping signature (vx fills the gap between the beams) ----
    const Particles& P = sim.particles();
    const int N = int(P.n);
    std::vector<float> vx(N), x(N), y(N);
    CUDA_CHECK(cudaMemcpy(vx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(),  P.y.data(),  P.y.bytes(),  cudaMemcpyDeviceToHost));
    int trapped = 0;
    for (int i = 0; i < N; ++i) if (std::fabs(vx[i]) < 0.5 * v0) ++trapped;
    const double frac_trapped = double(trapped) / N;
    std::printf("trapped fraction (|vx|<v0/2) = %.3f (expect > 0.10; starts at 0)\n", frac_trapped);
    if (frac_trapped < 0.10) { std::printf("FAIL: no trapping/vortex\n"); ok = false; }

    // ---- 3. dump phase space (x in cell units -> physical) for plotting ----
    // All particles, so the saturated vortices are fully resolved.
    {
        std::ofstream f("two_stream_phase.csv");
        f << "x,vx\n";
        for (int i = 0; i < N; ++i) f << x[i] * g.dx << ',' << vx[i] << '\n';
    }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
