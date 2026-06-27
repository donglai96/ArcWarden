// ArcWarden — Step 14 test [S8]: long-run energy & charge conservation.
//
// A thermal, Debye-resolved plasma (vth=1 => λ_D = vth/ωpe = 1, dx « λ_D) is a
// stable equilibrium. Over a long run the code must:
//   1. conserve total charge essentially exactly (fixed particle number, CIC),
//   2. keep total energy bounded (no secular numerical heating / cooling).
//
// Exits non-zero on failure.

#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int nx = 64, ny = 8;
    Grid g(nx, ny, 2.0 * M_PI, 2.0 * M_PI);   // dx ≈ 0.098 « λ_D = 1

    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.vth = 1.0; rp.vd = 0.0;                // warm, no drift
    rp.ppc = 64;
    rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;
    rp.dt = 0.05;
    rp.nsteps = 1000;                         // t = 50
    rp.dump_every = 10;

    Simulation<> sim(g, rp);
    sim.init();
    sim.run();

    const auto& H = sim.diagnostics().history();
    const int M = int(H.size());
    if (M < 10) { std::printf("FAIL: too few samples\n"); return 1; }

    double qmin=1e30, qmax=-1e30, qmean=0, tmin=1e30, tmax=-1e30, tmean=0;
    for (const auto& s : H) {
        qmin = std::fmin(qmin, s.charge); qmax = std::fmax(qmax, s.charge); qmean += s.charge;
        tmin = std::fmin(tmin, s.total);  tmax = std::fmax(tmax, s.total);  tmean += s.total;
    }
    qmean /= M; tmean /= M;

    const double q_expect   = double(sim.particles().n) * rp.qm * rp.weight;
    const double q_var      = std::fabs(qmax - qmin) / std::fabs(qmean);
    const double e_drift    = std::fabs(tmax - tmin) / tmean;

    std::printf("charge: mean=%.6f (expect %.6f) rel-variation=%.2e\n", qmean, q_expect, q_var);
    std::printf("total energy: mean=%.4f drift=(max-min)/mean=%.3f\n", tmean, e_drift);

    bool ok = true;
    if (std::fabs(qmean - q_expect) > 1e-4 * std::fabs(q_expect)) { std::printf("FAIL: charge value\n"); ok = false; }
    if (q_var   > 1e-4) { std::printf("FAIL: charge not constant\n"); ok = false; }
    if (e_drift > 0.05) { std::printf("FAIL: energy drift too large\n"); ok = false; }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
