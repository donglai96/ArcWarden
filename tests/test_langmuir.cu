// ArcWarden — Step 13 test [S7]: cold Langmuir oscillation.
//
// A cold (vth=0) plasma with a small single-k density perturbation oscillates at
// the plasma frequency ω_pe (independent of k for a cold plasma). With the ωpe=1
// normalization (n0=1, eps0=1, me=1, |e|=1) => ω_pe = 1.
//
// The field energy EE ∝ E² oscillates at 2·ω_pe, so its peaks are spaced π/ω_pe.
// Checks:
//   1. ω_pe measured from EE-peak spacing ≈ 1 (within ~5%).
//   2. KE and EE are anti-correlated (energy sloshes between them).
//   3. Total energy is bounded (no secular blow-up).
//
// Exits non-zero on failure.

#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int    nx = 64, ny = 8;
    const double TwoPi = 2.0 * M_PI;
    Grid g(nx, ny, TwoPi, TwoPi);     // Lx = 2π

    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.vth = 0.0;                     // cold
    rp.ppc = 64;
    rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;
    rp.dt = 0.05;
    rp.nsteps = 600;                  // ~4.8 plasma periods
    rp.dump_every = 1;
    rp.perturb_amp = 0.3;             // single-k seed (~3% density)
    rp.perturb_kx  = 1;

    Simulation<> sim(g, rp);
    sim.init();
    sim.run();

    const auto& H = sim.diagnostics().history();
    const int M = static_cast<int>(H.size());
    if (M < 100) { std::printf("FAIL: too few samples (%d)\n", M); return 1; }

    // ---- gather series ----
    std::vector<double> ee(M), ke(M), tot(M);
    double ee_max = 0.0, tot_mean = 0.0;
    for (int i = 0; i < M; ++i) {
        ee[i] = H[i].ee; ke[i] = H[i].ke; tot[i] = H[i].total;
        ee_max = std::fmax(ee_max, ee[i]);
        tot_mean += tot[i];
    }
    tot_mean /= M;

    // ---- 1. ω_pe from EE-peak spacing (peaks every π/ω_pe) ----
    std::vector<int> peaks;
    const double thresh = 0.25 * ee_max;
    for (int i = 1; i < M - 1; ++i)
        if (ee[i] > thresh && ee[i] >= ee[i-1] && ee[i] > ee[i+1]) peaks.push_back(i);

    bool ok = true;
    if (peaks.size() < 4) { std::printf("FAIL: only %zu EE peaks found\n", peaks.size()); return 1; }
    // average spacing of interior peaks (skip first transient)
    double dsum = 0; int dn = 0;
    for (size_t k = 2; k < peaks.size(); ++k) { dsum += (peaks[k] - peaks[k-1]); ++dn; }
    const double dT   = (dsum / dn) * rp.dt;     // EE period
    const double wpe  = M_PI / dT;               // ω_pe
    std::printf("EE peaks=%zu  EE period=%.4f  =>  w_pe=%.4f (expect 1.0)\n",
                peaks.size(), dT, wpe);
    if (std::fabs(wpe - 1.0) > 0.05) { std::printf("FAIL: w_pe off\n"); ok = false; }

    // ---- 2. KE<->EE anti-correlation (Pearson) ----
    double mk = 0, me = 0;
    for (int i = 0; i < M; ++i) { mk += ke[i]; me += ee[i]; }
    mk /= M; me /= M;
    double cov = 0, vk = 0, ve = 0;
    for (int i = 0; i < M; ++i) {
        cov += (ke[i]-mk)*(ee[i]-me); vk += (ke[i]-mk)*(ke[i]-mk); ve += (ee[i]-me)*(ee[i]-me);
    }
    const double corr = cov / std::sqrt(vk * ve);
    std::printf("corr(KE,EE) = %.4f (expect strongly negative)\n", corr);
    if (corr > -0.7) { std::printf("FAIL: KE/EE not anti-correlated\n"); ok = false; }

    // ---- 3. total energy bounded ----
    double tmin = 1e30, tmax = -1e30;
    for (int i = 0; i < M; ++i) { tmin = std::fmin(tmin, tot[i]); tmax = std::fmax(tmax, tot[i]); }
    const double spread = (tmax - tmin) / tot_mean;
    std::printf("total energy: mean=%.4e spread=(max-min)/mean=%.3f\n", tot_mean, spread);
    if (spread > 0.20) { std::printf("FAIL: total energy not bounded\n"); ok = false; }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
