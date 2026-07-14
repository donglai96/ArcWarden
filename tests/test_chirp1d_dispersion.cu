// chirp1d gate B — cold whistler eigenmode frequency vs cold R-mode dispersion.
//
// Periodic uniform mode, cold fluid only (no hot particles). Initialize the
// R-mode eigenmode at one k (fields + cold fluid velocity from the linear
// relations, convention e^{i(kx - w t)} with e- = Ey - iEz):
//   Ey = E0 cos kx          By = (k/w) E0 sin kx        Vcy = -E0 sin(kx)/(wce-w)
//   Ez = -E0 sin kx         Bz = (k/w) E0 cos kx        Vcz = -E0 cos(kx)/(wce-w)
// Then A(t) = sum_faces (By - iBz) e^{-ikx} rotates as e^{-iwt}; the unwrapped
// phase slope gives w. Theory: w^2 + w wpe^2/(wce - w) = c^2 keff^2 on (0,wce),
// with keff = sin(k dx/2)/(dx/2) (leading Yee spatial-dispersion correction).

#include "pic/hybrid1d.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

int main() {
    using namespace arc;

    Chirp1DParams P;
    P.nx = 512; P.dx = 0.25; P.dt = 0.05;
    P.wpe = 4.0; P.a = 0.0;
    P.periodic = true;
    P.nh = 0.0; P.ppc = 0;

    const int m = 32;
    const double L = P.nx * P.dx;
    const double k = 2.0 * M_PI * m / L;
    const double keff = std::sin(0.5 * k * P.dx) / (0.5 * P.dx);

    // cold whistler root on (0, wce): f(w) = w^2 + w wpe^2/(1-w) - keff^2
    auto f = [&](double w) {
        return w * w + w * P.wpe * P.wpe / (1.0 - w) - keff * keff;
    };
    double lo = 1e-6, hi = 1.0 - 1e-6;
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) > 0) hi = mid; else lo = mid;
    }
    const double w_theory = 0.5 * (lo + hi);

    Hybrid1D sim(P);

    const double E0 = 1e-6;
    std::vector<double> ey(P.nx), ez(P.nx), by(P.nx), bz(P.nx), vy(P.nx), vz(P.nx);
    for (int i = 0; i < P.nx; ++i) {
        const double xn = i * P.dx;             // nodes
        const double xf = (i + 0.5) * P.dx;     // faces
        ey[i] =  E0 * std::cos(k * xn);
        ez[i] = -E0 * std::sin(k * xn);
        by[i] =  (k / w_theory) * E0 * std::sin(k * xf);
        bz[i] =  (k / w_theory) * E0 * std::cos(k * xf);
        vy[i] = -E0 * std::sin(k * xn) / (1.0 - w_theory);
        vz[i] = -E0 * std::cos(k * xn) / (1.0 - w_theory);
    }
    sim.set_fields(ey, ez, by, bz, vy, vz);

    const int    sample = 4;
    const double T      = 12.0 * 2.0 * M_PI / w_theory;  // ~12 periods
    const long   nsteps = (long)(T / P.dt);

    std::vector<double> phase;
    std::vector<double> hby, hbz;
    std::complex<double> prev(0, 0);
    double acc = 0;
    for (long n = 0; n < nsteps; ++n) {
        sim.step();
        if (n % sample) continue;
        sim.get_b(hby, hbz);
        std::complex<double> A(0, 0);
        for (int i = 0; i < P.nx; ++i) {
            const double xf = (i + 0.5) * P.dx;
            A += std::complex<double>(hby[i], -hbz[i]) *
                 std::exp(std::complex<double>(0, -k * xf));
        }
        if (phase.empty()) { prev = A; phase.push_back(std::arg(A)); continue; }
        const double d = std::arg(A / prev);    // increment in (-pi, pi]
        acc += d;
        phase.push_back(phase.front() + acc);
        prev = A;
    }

    // linear fit of phase(t): slope = -w
    const long N = (long)phase.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (long j = 0; j < N; ++j) {
        const double t = (double)j;
        sx += t; sy += phase[j]; sxx += t * t; sxy += t * phase[j];
    }
    const double slope = (N * sxy - sx * sy) / (N * sxx - sx * sx);
    const double w_meas = -slope / (sample * P.dt);

    const double err = std::abs(w_meas - w_theory) / w_theory;
    std::printf("k=%.4f keff=%.4f  w_theory=%.6f  w_meas=%.6f  err=%.3e\n",
                k, keff, w_theory, w_meas, err);
    const bool ok = err < 0.015 && w_meas > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
