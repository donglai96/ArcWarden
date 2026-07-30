// REFRESH Gate 1 — the null test (docs/REFRESH_DESIGN.md):
// "the bath must be invisible when f = f0."
//
// Twin runs of a Chen-2026-case-2-like deck (dipole, loss-cone hot full-f,
// x = damping boundary — fields damped, particles reflect; refresh rejects
// the hybrid particle absorber, see refresh.hpp header — SAME seed), 3000
// steps inside the noise window (t = 450/wpe = 90/We0, two decades below
// element onset), one with the refresh bath ON (lambda_R = 15 deg), one
// OFF. Criteria:
//   (a) INTERIOR (|s| < s_R) n(x), Tpar(x), Tperp(x) agree bin-wise within
//       shot noise — the bath must not touch the generation region,
//   (b) net injected energy ~ 0 (random walk, no systematic pump/drain),
//   (c) redraws happen (the bath is actually active, not silently off),
//       draw failures are a small fraction.
// The BATH bins are reported but NOT gated tightly: the OFF steady state
// there is f0 + the absorber-carved cold-parallel population, and the bath
// deliberately converts those stragglers back into trapped f0 samples on
// their first outward pass — a real, wanted difference (see refresh.hpp).
// A WRONG local distribution (e.g. equatorial widths instead of the (E,mu)
// mapping, or a broken loss-cone T2) shows up as a systematic bath-region
// T distortion PLUS interior leakage and a monotonic dE ramp — far above
// these tolerances.
//
// Gate 2 rides along: lambda_R = 12/15/18 deg short runs must agree in the
// INTERIOR (|s| < s(12 deg)) within the same envelope (geometric choice,
// not a physics knob).

#include "pic/deck.hpp"
#include "pic/refresh.hpp"
#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace arc;

namespace {

const char* DECK_FMT = R"([grid]
nx = 5000
ny = 1
Lx = 1300.0
Ly = 0.26

[time]
dt = 0.15
nsteps = 1

[field]
model = yee
jfilter = 3

[plasma]
qm = -1.0
cold_nc = 0.9822
rel = true
noisy = true
seed = 20260726

[background]
wce = 0.2
profile = dipole
lre = 1330.504

[boundary]
x = damping
nd = 300
numax = 0.1

[antenna]
amp = 0.0

[refresh]
enable = %s
lambda_deg = %.1f

[species hot]
density = 0.0178
ppc = 400
uth = 0.19783565 0.24390817 0.24390817
rep = fullf
dist = losscone
rho = 1.0
kappa = 0.3
)";

struct Profile {
    std::vector<double> n, tpar, tperp;   // per x-bin
    double ke = 0.0;                      // total hot kinetic energy
};

struct RunResult {
    Profile prof;
    double  redraws = 0.0, de = 0.0, fails = 0.0;   // refresh totals over the run
};

RunResult run_case(bool refresh_on, double lambda_deg, int nsteps, int nbin) {
    char text[4096];
    std::snprintf(text, sizeof text, DECK_FMT, refresh_on ? "true" : "false",
                  lambda_deg);
    const std::string path = "refresh_null_deck.tmp.ini";
    { std::ofstream f(path); f << text; }
    Deck d = load_deck(path);
    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    const Species& q = d.species[0];

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(q, g, rp, sim.stream());
    RefreshState rfr;
    rfr.init(q, g, rp, sim.stream());
    sim.stream().synchronize();

    RunResult rr;
    for (int n = 1; n <= nsteps; ++n) {
        sim.step();
        if (rfr.on) rfr.apply(sim.particles(), g, rp, n, sim.stream());
    }
    if (rfr.on) {
        const auto s = rfr.drain(sim.stream());
        rr.redraws = s.redraws; rr.de = s.de; rr.fails = s.fails;
    }
    sim.stream().synchronize();

    // host moments, nbin equal x-bins
    const size_t N = sim.particles().n;
    std::vector<float> x(N), ux(N), uy(N), uz(N), w(N);
    CUDA_CHECK(cudaMemcpy(x.data(),  sim.particles().x.data(),  N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ux.data(), sim.particles().ux.data(), N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(), sim.particles().uy.data(), N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz.data(), sim.particles().uz.data(), N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w.data(),  sim.particles().w.data(),  N * 4, cudaMemcpyDeviceToHost));

    std::vector<double> sw(nbin, 0), sux(nbin, 0), sux2(nbin, 0), sup2(nbin, 0);
    for (size_t t = 0; t < N; ++t) {
        int b = (int)(x[t] * nbin / g.nx);
        if (b < 0) b = 0; if (b >= nbin) b = nbin - 1;
        const double W = w[t];
        sw[b]   += W;
        sux[b]  += W * ux[t];
        sux2[b] += W * (double)ux[t] * ux[t];
        sup2[b] += W * 0.5 * ((double)uy[t] * uy[t] + (double)uz[t] * uz[t]);
        const double u2 = (double)ux[t] * ux[t] + (double)uy[t] * uy[t]
                        + (double)uz[t] * uz[t];
        rr.prof.ke += W * (rp.rel ? std::sqrt(1.0 + u2) - 1.0 : 0.5 * u2);
    }
    rr.prof.n.resize(nbin); rr.prof.tpar.resize(nbin); rr.prof.tperp.resize(nbin);
    for (int b = 0; b < nbin; ++b) {
        const double m = sw[b] > 0 ? sw[b] : 1.0;
        const double mu = sux[b] / m;
        rr.prof.n[b]     = sw[b];
        rr.prof.tpar[b]  = sux2[b] / m - mu * mu;
        rr.prof.tperp[b] = sup2[b] / m;
    }
    return rr;
}

double worst_rel(const std::vector<double>& a, const std::vector<double>& b,
                 int lo, int hi) {
    double worst = 0.0;
    for (int i = lo; i < hi; ++i)
        if (a[i] > 0.0)
            worst = std::fmax(worst, std::fabs(a[i] - b[i]) / a[i]);
    return worst;
}

} // namespace

// Probe: apply ONE refresh to the fresh load (zero dynamics) and compare
// bath-bin moments against the untouched load — isolates the sampler from
// the push. A correct sampler leaves every moment inside shot noise.
static void sampler_probe() {
    char text[4096];
    std::snprintf(text, sizeof text, DECK_FMT, "true", 15.0);
    const std::string path = "refresh_null_deck.tmp.ini";
    { std::ofstream f(path); f << text; }
    Deck d = load_deck(path);
    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    const Species& q = d.species[0];

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(q, g, rp, sim.stream());
    sim.stream().synchronize();

    const size_t N = sim.particles().n;
    auto moments = [&](std::vector<double>& tpar, std::vector<double>& tperp,
                       int nbin) {
        std::vector<float> x(N), ux(N), uy(N), uz(N), w(N);
        CUDA_CHECK(cudaMemcpy(x.data(),  sim.particles().x.data(),  N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), sim.particles().ux.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), sim.particles().uy.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), sim.particles().uz.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w.data(),  sim.particles().w.data(),  N * 4, cudaMemcpyDeviceToHost));
        std::vector<double> sw(nbin, 0), s2(nbin, 0), sp(nbin, 0);
        for (size_t t = 0; t < N; ++t) {
            int b = (int)(x[t] * nbin / g.nx);
            if (b < 0) b = 0; if (b >= nbin) b = nbin - 1;
            sw[b] += w[t];
            s2[b] += w[t] * (double)ux[t] * ux[t];
            sp[b] += w[t] * 0.5 * ((double)uy[t] * uy[t] + (double)uz[t] * uz[t]);
        }
        tpar.resize(nbin); tperp.resize(nbin);
        for (int b = 0; b < nbin; ++b) {
            tpar[b]  = s2[b] / sw[b];
            tperp[b] = sp[b] / sw[b];
        }
    };

    const int NB = 50;
    std::vector<double> tpar0, tperp0, tpar1, tperp1;
    moments(tpar0, tperp0, NB);

    RefreshState rfr;
    rfr.init(q, g, rp, sim.stream());
    rfr.apply(sim.particles(), g, rp, 1, sim.stream());
    const auto s = rfr.drain(sim.stream());
    sim.stream().synchronize();
    moments(tpar1, tperp1, NB);

    std::printf("--- sampler probe: ONE refresh on the fresh load ---\n");
    std::printf("redraws %.3e  fails %.3e  dE %.3e\n", s.redraws, s.fails, s.de);
    std::printf("bin   x-range      Tpar0     Tpar1     dTp%%   Tperp0    Tperp1    dTe%%\n");
    for (int b = 0; b < NB; ++b) {
        const double xlo = b * 1300.0 / NB, xhi = (b + 1) * 1300.0 / NB;
        if (std::fabs(0.5 * (xlo + xhi) - 650.0) < 330.0) continue;   // bath only
        std::printf("%3d %6.0f-%-6.0f %.6f  %.6f  %+5.1f  %.6f  %.6f  %+5.1f\n",
                    b, xlo, xhi, tpar0[b], tpar1[b],
                    100.0 * (tpar1[b] / tpar0[b] - 1.0),
                    tperp0[b], tperp1[b],
                    100.0 * (tperp1[b] / tperp0[b] - 1.0));
    }
}

int main() {
    const int NSTEP = 3000, NBIN = 50;
    bool ok = true;

    sampler_probe();

    std::printf("=== refresh Gate 1: null test (f = f0, bath ON vs OFF) ===\n");
    const RunResult off = run_case(false, 15.0, NSTEP, NBIN);
    const RunResult on  = run_case(true,  15.0, NSTEP, NBIN);

    // interior bins: |s| < s(15 deg) = 349 c/wpe, equator at 650
    const double s15 = 1330.504 * bg::dipole_s_of_lambda(15.0 * M_PI / 180.0);
    const int ilo = (int)((650.0 - s15) / 1300.0 * NBIN) + 1;
    const int ihi = (int)((650.0 + s15) / 1300.0 * NBIN);
    const double dn  = worst_rel(off.prof.n,     on.prof.n,     ilo, ihi);
    const double dtp = worst_rel(off.prof.tpar,  on.prof.tpar,  ilo, ihi);
    const double dte = worst_rel(off.prof.tperp, on.prof.tperp, ilo, ihi);
    std::printf("INTERIOR bins [%d,%d) worst rel diff: n %.4f  Tpar %.4f  "
                "Tperp %.4f\n", ilo, ihi, dn, dtp, dte);
    const double bn  = worst_rel(off.prof.n,     on.prof.n,     0, NBIN);
    const double btp = worst_rel(off.prof.tpar,  on.prof.tpar,  0, NBIN);
    const double bte = worst_rel(off.prof.tperp, on.prof.tperp, 0, NBIN);
    std::printf("(all-bin incl. bath/absorber, informational: n %.4f  "
                "Tpar %.4f  Tperp %.4f)\n", bn, btp, bte);
    if (dn  > 0.03) { std::printf("FAIL: interior n(x) moved %.1f%% > 3%%\n",  100 * dn);  ok = false; }
    if (dtp > 0.05) { std::printf("FAIL: interior Tpar(x) moved %.1f%% > 5%%\n", 100 * dtp); ok = false; }
    if (dte > 0.05) { std::printf("FAIL: interior Tperp(x) moved %.1f%% > 5%%\n", 100 * dte); ok = false; }

    std::printf("redraws total %.3e (%.3e/step)  fails %.3e  net dE %.3e  "
                "hot KE %.3e  |dE|/KE %.2e\n", on.redraws, on.redraws / NSTEP,
                on.fails, on.de, on.prof.ke, std::fabs(on.de) / on.prof.ke);
    if (on.redraws <= 0.0) { std::printf("FAIL: bath never fired\n"); ok = false; }
    if (on.fails > 0.05 * on.redraws) {
        std::printf("FAIL: draw failures %.1f%% of redraws > 5%%\n",
                    100 * on.fails / on.redraws);
        ok = false;
    }
    if (std::fabs(on.de) > 0.02 * on.prof.ke) {
        std::printf("FAIL: net injected energy %.2f%% of hot KE > 2%% "
                    "(systematic pump/drain)\n", 100 * std::fabs(on.de) / on.prof.ke);
        ok = false;
    }

    std::printf("=== refresh Gate 2: lambda_R geometric insensitivity ===\n");
    const RunResult r12 = run_case(true, 12.0, NSTEP, NBIN);
    const RunResult r18 = run_case(true, 18.0, NSTEP, NBIN);
    // interior = |s| < s(12 deg) on both sides; bins are uniform in x over
    // [0, Lx], equator at Lx/2. s(12 deg)/lre = 0.2100 -> 279 c/wpe.
    const double s12 = 1330.504 * bg::dipole_s_of_lambda(12.0 * M_PI / 180.0);
    const int lo = (int)((650.0 - s12) / 1300.0 * NBIN) + 1;
    const int hi = (int)((650.0 + s12) / 1300.0 * NBIN);
    const double g2n  = std::fmax(worst_rel(r12.prof.n, on.prof.n, lo, hi),
                                  worst_rel(r18.prof.n, on.prof.n, lo, hi));
    const double g2tp = std::fmax(worst_rel(r12.prof.tpar, on.prof.tpar, lo, hi),
                                  worst_rel(r18.prof.tpar, on.prof.tpar, lo, hi));
    const double g2te = std::fmax(worst_rel(r12.prof.tperp, on.prof.tperp, lo, hi),
                                  worst_rel(r18.prof.tperp, on.prof.tperp, lo, hi));
    std::printf("interior bins [%d,%d) worst rel diff vs 15deg: n %.4f  "
                "Tpar %.4f  Tperp %.4f\n", lo, hi, g2n, g2tp, g2te);
    if (g2n > 0.03 || g2tp > 0.05 || g2te > 0.05) {
        std::printf("FAIL: interior depends on lambda_R (knob, not geometry)\n");
        ok = false;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
