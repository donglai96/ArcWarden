// chirp1d gate A — single-particle mirror bounce in the parabolic B0(h).
//
// For B0 = 1 + a (x-xc)^2 the gyro-averaged mirror force gives EXACT SHM in x:
//   du_par/dt = -(u_perp^2 / (2 gamma B0)) dB0/dx = -(2 a M / gamma) (x-xc),
// with M = u_perp^2 / (2 B0) invariant and gamma constant (no E field), so
//   omega_b = sqrt(a) * u_perp_eq / gamma.
// Asserts: bounce frequency within 1% of theory; relative mu spread < 1%;
// |u| (energy) drift at float-storage roundoff.

#include "pic/hybrid1d.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace arc;

    Chirp1DParams P;
    P.nx = 4096; P.dx = 0.5; P.dt = 0.1;
    P.a = 4e-4;
    P.wpe = 4.0;
    P.nh = 0.0; P.ppc = 0;          // no self-consistent fields
    P.periodic = false; P.nd = 16; P.numax = 0.0;

    Hybrid1D sim(P);

    const double upar0 = 0.05, uperp0 = 0.1;
    const double gam = std::sqrt(1.0 + upar0 * upar0 + uperp0 * uperp0);
    const double wb_theory = std::sqrt(P.a) * uperp0 / gam;

    sim.set_particles({sim.xc()}, {(float)upar0}, {(float)uperp0}, {0.0f});
    sim.half_step_back();

    const double T = 3.2 * 2.0 * M_PI / wb_theory;   // ~3 bounce periods
    const long nsteps = (long)(T / P.dt);

    std::vector<double> x, mu_hist, u2_hist;
    std::vector<float> ux, uy, uz;
    std::vector<double> xs(nsteps);
    double mu_min = 1e300, mu_max = -1e300, u2_0 = 0, u2_end = 0;

    for (long n = 0; n < nsteps; ++n) {
        sim.step();
        sim.get_particles(x, ux, uy, uz);
        xs[n] = x[0] - sim.xc();
        const double up2 = (double)uy[0] * uy[0] + (double)uz[0] * uz[0];
        const double b0 = 1.0 + P.a * xs[n] * xs[n];
        const double mu = up2 / (2.0 * b0);
        mu_min = std::min(mu_min, mu);
        mu_max = std::max(mu_max, mu);
        const double u2 = (double)ux[0] * ux[0] + up2;
        if (n == 0) u2_0 = u2;
        u2_end = u2;
    }

    // bounce frequency from zero crossings of x - xc
    int ncross = 0;
    double t_first = 0, t_last = 0;
    for (long n = 1; n < nsteps; ++n) {
        if ((xs[n - 1] < 0 && xs[n] >= 0) || (xs[n - 1] > 0 && xs[n] <= 0)) {
            // linear interpolation of crossing time
            const double f = xs[n - 1] / (xs[n - 1] - xs[n]);
            const double tc = (n + f) * P.dt;
            if (ncross == 0) t_first = tc;
            t_last = tc;
            ++ncross;
        }
    }
    if (ncross < 4) { std::printf("FAIL: only %d equator crossings\n", ncross); return 1; }
    const double wb_meas = M_PI * (ncross - 1) / (t_last - t_first);

    const double werr  = std::abs(wb_meas - wb_theory) / wb_theory;
    const double muerr = (mu_max - mu_min) / (0.5 * (mu_max + mu_min));
    const double eerr  = std::abs(u2_end - u2_0) / u2_0;

    std::printf("bounce: wb_meas=%.6e wb_theory=%.6e err=%.3e\n", wb_meas, wb_theory, werr);
    std::printf("mu spread=%.3e  |u|^2 drift=%.3e\n", muerr, eerr);

    const bool ok = werr < 0.01 && muerr < 0.01 && eerr < 1e-4;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
