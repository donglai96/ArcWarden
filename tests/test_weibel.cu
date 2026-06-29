// ArcWarden — Darwin GATE 2: Weibel / filamentation instability.
//
// Two cold counter-streaming electron beams along z (uz = ±v0), neutralizing ion
// background. A transverse current perturbation Jz(x,y) seeds a magnetic field B;
// the v×B force bunches the beams into current filaments → B grows exponentially.
// This exercises the FULL Darwin loop: ρ,J deposit → E_L,B solve → dcu,amu + the
// transverse E_T iteration → magnetized push.
//
// Cold-beam filamentation growth rate (ω_pe=1):  γ(k) = k v0 / sqrt(1+(k c)²),
// saturating to γ_max = v0/c at large k. With v0=1, c=5 → γ_max = 0.2, and every
// resolved box mode (k≥1) grows at ≈0.19–0.20. Magnetic ENERGY grows as e^{2γt}.
//
// PASS: magnetic energy grows exponentially over a wide dynamic range, the fitted
// γ lands in a band around the analytic γ_max, and it saturates.

#include "pic/config.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const double TwoPi = 6.28318530717958647692;
    const int nx = 64, ny = 64;
    Grid g(nx, ny, TwoPi, TwoPi);
    const int N = g.real_size();

    const double v0 = 1.0, c = 5.0;
    RunParams rp;
    rp.dt = 0.1; rp.c = c; rp.ndc = 2;
    rp.qm = -1.0; rp.eps0 = 1.0;
    rp.noisy_load = true; rp.rng_seed = 20260628UL;
    rp.dump_every = 0;

    // two cold counter-streaming beams along z (each half density, equal ppc)
    const double uth = 0.05;
    SpeciesList sp = {
        Species{ "beam_p", 0.5, 64, {uth, uth, uth}, {0.0, 0.0,  v0} },
        Species{ "beam_m", 0.5, 64, {uth, uth, uth}, {0.0, 0.0, -v0} },
    };

    Simulation<CfgDarwin> sim(g, rp, sp);
    sim.init();

    auto mag_energy = [&]() {
        cudaDeviceSynchronize();
        const Fields& f = sim.fields();
        std::vector<float> bx(N), by(N), bz(N);
        cudaMemcpy(bx.data(), f.Bx.data(), f.Bx.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(by.data(), f.By.data(), f.By.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(bz.data(), f.Bz.data(), f.Bz.bytes(), cudaMemcpyDeviceToHost);
        double e = 0;
        for (int i = 0; i < N; ++i)
            e += (double)bx[i]*bx[i] + (double)by[i]*by[i] + (double)bz[i]*bz[i];
        return 0.5 * e * g.dx * g.dy;
    };

    const int nsteps = 1000, sample = 5;
    std::vector<double> ts, lnEB;
    for (int n = 0; n < nsteps; ++n) {
        sim.step(n);
        if (n % sample == 0) {
            const double eb = mag_energy();
            ts.push_back(n * rp.dt);
            lnEB.push_back(std::log(eb > 1e-300 ? eb : 1e-300));
        }
    }

    // dynamic range + saturation
    double lnmin = lnEB[0], lnmax = lnEB[0];
    int imax = 0;
    for (size_t i = 0; i < lnEB.size(); ++i) {
        if (lnEB[i] < lnmin) lnmin = lnEB[i];
        if (lnEB[i] > lnmax) { lnmax = lnEB[i]; imax = (int)i; }
    }
    const double growth_decades = (lnmax - lnmin) / std::log(10.0);

    // least-squares slope of ln(EB) vs t over the exponential window:
    // from where EB has risen 1 e-fold above min, up to 80% of the way to the peak.
    const double lo = lnmin + 1.0, hi = lnmin + 0.8 * (lnmax - lnmin);
    double Sx=0, Sy=0, Sxx=0, Sxy=0; int m=0;
    for (size_t i = 0; i < lnEB.size(); ++i) {
        if ((int)i <= imax && lnEB[i] >= lo && lnEB[i] <= hi) {
            Sx += ts[i]; Sy += lnEB[i]; Sxx += ts[i]*ts[i]; Sxy += ts[i]*lnEB[i]; ++m;
        }
    }
    const double slope = (m >= 2) ? (m*Sxy - Sx*Sy) / (m*Sxx - Sx*Sx) : 0.0;
    const double gamma = 0.5 * slope;             // EB ~ e^{2γt}
    const double gamma_analytic = v0 / c;          // γ_max = v0 ω_pe / c

    const bool saturates = imax < (int)lnEB.size() - 1;
    std::printf("Weibel: growth range = %.1f decades of magnetic energy\n", growth_decades);
    std::printf("        fitted gamma = %.3f  (analytic gamma_max = v0/c = %.3f)\n",
                gamma, gamma_analytic);
    std::printf("        peak at t=%.1f of %.1f  (saturates: %s)\n",
                ts[imax], ts.back(), saturates ? "yes" : "no");

    bool ok = true;
    if (growth_decades < 3.0) { std::printf("FAIL: too little growth (<3 decades) — instability not operating\n"); ok = false; }
    if (gamma < 0.10 || gamma > 0.32) { std::printf("FAIL: gamma %.3f outside [0.10,0.32] band around %.2f\n", gamma, gamma_analytic); ok = false; }
    if (!saturates) { std::printf("FAIL: no saturation (still growing at end)\n"); ok = false; }

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
