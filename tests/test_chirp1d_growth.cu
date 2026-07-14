// chirp1d gate C — whistler anisotropy-instability growth rate.
//
// Periodic uniform B0, cold fluid + hot bi-Maxwellian (A = 3). Transverse
// B-energy must grow exponentially at 2*gamma with gamma near the kinetic
// linear theory value from scripts/whistler_kinetic_dispersion.py:
//   wpe/wce = 4, nh/nc = 0.02, uth_para = 0.2, uth_perp = 0.4
//   -> gamma_max = 1.635e-2 wce at k ~ 2.6 wce/c (w_r ~ 0.29 wce).
// The fit window is the two decades of W_B just below saturation. Tolerance is
// loose (35%): finite ppc noise, mild relativistic shift, multi-mode overlap.

#include "pic/hybrid1d.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace arc;

    Chirp1DParams P;
    P.nx = 1024; P.dx = 0.1; P.dt = 0.04;
    P.wpe = 4.0; P.a = 0.0;
    P.periodic = true;
    P.nh = 0.02; P.uth_para = 0.2; P.uth_perp = 0.4;
    P.ppc = 512;
    P.seed = 20260713ull;
    P.nonrel = true;      // theory value below is non-relativistic

    const double gamma_theory = 1.635e-2;

    Hybrid1D sim(P);
    std::printf("markers: %ld\n", sim.np());

    const int  stride = 25;
    const long nsteps = 40000;
    std::vector<double> ts, wb;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n % stride == 0) {
            const auto e = sim.energies();
            ts.push_back(n * P.dt);
            wb.push_back(e.wb);
        }
    }

    double wbmax = 0;
    size_t imax = 0;
    for (size_t i = 0; i < wb.size(); ++i)
        if (wb[i] > wbmax) { wbmax = wb[i]; imax = i; }
    const double wb0 = wb.front();
    std::printf("WB: first=%.3e max=%.3e (x%.1e)\n", wb0, wbmax, wbmax / wb0);
    if (wbmax < 1e3 * wb0) {
        std::printf("FAIL: less than 3 decades of growth\n");
        return 1;
    }

    // fit ln WB over the two decades below saturation (up to the max)
    const double hi = wbmax / 10.0, lo = wbmax / 1000.0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    long   N = 0;
    for (size_t i = 0; i <= imax; ++i) {
        if (wb[i] < lo || wb[i] > hi) continue;
        const double x = ts[i], y = std::log(wb[i]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
        ++N;
    }
    if (N < 5) { std::printf("FAIL: too few points in fit window (%ld)\n", N); return 1; }
    const double slope = (N * sxy - sx * sy) / (N * sxx - sx * sx);
    const double gamma_meas = 0.5 * slope;

    const double err = std::abs(gamma_meas - gamma_theory) / gamma_theory;
    std::printf("gamma_meas=%.4e  gamma_theory=%.4e  err=%.2f%%\n",
                gamma_meas, gamma_theory, 100 * err);
    const bool ok = err < 0.35;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
