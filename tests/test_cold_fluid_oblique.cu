// G2.2 gate — cold_full fluid OBLIQUE dispersion + zero numerical damping.
//
// Uniform B0 along x, cold_model=full fluid only (no particles), periodic
// box, small white-noise seed in the transverse B. For a set of (kx, ky)
// modes spanning theta = 0..63 deg, the measured oscillation frequency of
// the whistler branch must match the EXACT cold electron magnetized
// dispersion (Stix cold tensor, root of A n^4 - B n^2 + C below Omega_e)
// to < 2% — this is the direct proof that the symmetric staggered gather/
// scatter has no oblique phase error. Additionally the total field energy
// over the second half of the run must not drift vs the first half by more
// than a few percent (no numerical damping/growth: symmetric averages have
// a REAL transfer function).
//
// Legacy note: the ny=1 transverse-only kernel is untouched by cold_full
// (test_cold_fluid_dispersion covers it); this gate exercises the new path.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

// exact cold electron-only dispersion: for real k and angle th, find the
// whistler root w in (0, We) of  A(w) n^4 - B(w) n^2 + C(w) = 0, n = k c/w.
// Stix: R = 1 - X/(w(w - Y))... using X = wpe^2/w^2, Y = We/w (electrons,
// qm = -1 convention absorbed into the R/L assignment for the branch that
// rotates with electrons below We).
static double whistler_w_exact(double k, double th, double wpe, double We) {
    const double s = std::sin(th), c = std::cos(th);
    auto det = [&](double w) {
        const double wp2 = wpe * wpe;
        const double R = 1.0 - wp2 / (w * (w - We));
        const double L = 1.0 - wp2 / (w * (w + We));
        const double S = 0.5 * (R + L), P = 1.0 - wp2 / (w * w);
        const double n2 = (k * k) / (w * w);
        const double A = S * s * s + P * c * c;
        const double B = R * L * s * s + P * S * (1.0 + c * c);
        return A * n2 * n2 - B * n2 + R * L * P;
    };
    double lo = 1e-4 * We, hi = 0.999 * We;
    // the whistler branch is the single propagating root below We for
    // wpe >> We; bisect on the sign change
    double flo = det(lo);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (det(mid) * flo > 0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// one resolution level: physical modes fixed (mode numbers scale with ref),
// errors from the second-order discrete operators must shrink ~4x per
// refinement — convergence proof that the residual is grid dispersion, not
// an implementation error. Returns per-mode errors.
static bool run_level(int ref, double* errs, double& drift) {
    const int nx = 64 * ref, ny = 64 * ref;
    const double dx = 0.5 / ref, dt = 0.05 / ref, wce = 0.25, nc = 1.0;
    Grid g(nx, ny, nx * dx, ny * dx);

    RunParams rp;
    rp.dt = dt; rp.c = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.cold_nc = nc; rp.cold_full = 1;
    rp.dump_every = 0;

    MaxwellSimulation sim(g, rp);
    {   // white-noise transverse-B seed
        std::mt19937 gen(20260723u);
        std::uniform_real_distribution<float> un(-1e-6f, 1e-6f);
        std::vector<float> by(g.real_size()), bz(g.real_size());
        for (auto& x : by) x = un(gen);
        for (auto& x : bz) x = un(gen);
        CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), by.data(), by.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), bz.data(), bz.size() * 4, cudaMemcpyHostToDevice));
    }

    const long nsteps = 40000L * ref;       // same physical T = 2000
    const int  every  = 4 * ref;            // same physical sampling
    const int  nsamp  = (int)(nsteps / every);
    // probe modes (mx, my): theta = atan2(ky, kx)
    const int NM = 5;
    // SAME physical modes at every level (box size fixed): refinement only
    // halves k*Delta, so errors from second-order operators drop ~4x
    const int mx[NM] = {4, 4, 3, 2, 2};
    const int my[NM] = {0, 2, 3, 3, 4};
    std::vector<std::complex<double>> ts((size_t)NM * nsamp);
    std::vector<float> by(g.real_size()), bz(g.real_size());

    double e1 = 0, e2 = 0; int n1 = 0, n2 = 0;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n % every) continue;
        CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), by.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
        const int s = (int)(n / every) - 1;
        for (int m = 0; m < NM; ++m) {
            std::complex<double> acc(0, 0);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double ph = -2.0 * M_PI * (mx[m] * (double)i / nx
                                                     + my[m] * (double)j / ny);
                    const std::complex<double> e(std::cos(ph), std::sin(ph));
                    acc += e * std::complex<double>(by[(size_t)j * nx + i],
                                                    bz[(size_t)j * nx + i]);
                }
            ts[(size_t)m * nsamp + s] = acc;
        }
        double eb = 0;
        for (size_t c = 0; c < by.size(); ++c)
            eb += (double)by[c] * by[c] + (double)bz[c] * bz[c];
        if (n <= nsteps / 2) { e1 += eb; ++n1; } else { e2 += eb; ++n2; }
    }
    e1 /= n1; e2 /= n2;

    bool ok = true;
    const double dk = 2.0 * M_PI / (nx * dx);
    std::printf("ref=%d: mode  theta    k      w_meas    w_exact   err\n", ref);
    for (int m = 0; m < NM; ++m) {
        const double kx = mx[m] * dk, ky = my[m] * dk;
        const double k = std::hypot(kx, ky), th = std::atan2(ky, kx);
        // spectral peak of the mode time series (By+iBz keeps signed omega;
        // electron whistler rotation shows up at one sign — take |FFT| max)
        std::vector<std::complex<double>> f(ts.begin() + (size_t)m * nsamp,
                                            ts.begin() + (size_t)(m + 1) * nsamp);
        // DFT peak search over positive+negative freq
        auto mag_at = [&](double q) {
            // Hann window: oblique modes are elliptically polarized, so
            // (By+iBz) carries lines at BOTH +-w; rectangular-window
            // sidelobes of the mirror line bias the peak by ~1-2%
            const double w = 2.0 * M_PI * q / (nsamp * every * dt);
            std::complex<double> acc(0, 0);
            for (int s2 = 0; s2 < nsamp; ++s2) {
                const double win = 0.5 * (1.0 - std::cos(2.0 * M_PI * s2 / (nsamp - 1)));
                const double ph = w * (s2 * every * dt);
                acc += win * f[s2] * std::complex<double>(std::cos(ph), std::sin(ph));
            }
            return std::abs(acc);
        };
        int best = 0; double bmag = -1;
        const int NF = 4000;
        for (int q = -NF / 2; q < NF / 2; ++q) {
            const double w = 2.0 * M_PI * q / (nsamp * every * dt);
            if (std::fabs(w) > 0.99 * wce || std::fabs(w) < 0.02 * wce) continue;
            const double mag = mag_at(q);
            if (mag > bmag) { bmag = mag; best = q; }
        }
        // parabolic peak interpolation: kills the 2pi/T frequency-bin
        // quantization (half a bin = several % at the low-omega modes)
        const double m0 = mag_at(best), mp = mag_at(best + 1), mm2 = mag_at(best - 1);
        const double den = mm2 - 2.0 * m0 + mp;
        const double off = (std::fabs(den) > 1e-30) ? 0.5 * (mm2 - mp) / den : 0.0;
        const double wm = std::fabs(2.0 * M_PI * (best + off) / (nsamp * every * dt));
        const double wt = whistler_w_exact(k, th, 1.0, wce);
        const double err = std::fabs(wm - wt) / wt;
        std::printf("(%d,%d) %5.1f  %5.3f  %.5f  %.5f  %.3f%%\n",
                    mx[m], my[m], th * 180 / M_PI, k, wm, wt, 100 * err);
        errs[m] = err;
    }
    drift = e2 / e1 - 1.0;
    std::printf("energy drift half2/half1 - 1 = %+.4f\n", drift);
    if (std::fabs(drift) > 0.05) { std::printf("FAIL: energy drift %.1f%%\n", 100 * drift); ok = false; }
    return ok;
}

int main() {
    double eA[5], eB[5], dA, dB;
    bool ok = run_level(1, eA, dA);
    ok = run_level(2, eB, dB) && ok;
    // gates: refined level accurate (<1.5% all modes) AND ~second-order
    // convergence on the coarse level's worst modes (factor > 2.5)
    for (int m = 0; m < 5; ++m) {
        if (eB[m] > 0.015) { std::printf("FAIL: refined mode %d err %.2f%% > 1.5%%\n", m, 100 * eB[m]); ok = false; }
        if (eA[m] > 0.01 && eA[m] / eB[m] < 2.5) {
            std::printf("FAIL: mode %d converges x%.1f < 2.5 (not pure grid dispersion)\n", m, eA[m] / eB[m]);
            ok = false;
        }
        if (eA[m] > 0.01) std::printf("mode %d: convergence factor %.1f\n", m, eA[m] / eB[m]);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
