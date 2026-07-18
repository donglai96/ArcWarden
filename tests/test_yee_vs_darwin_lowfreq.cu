// M1 gate [yee_vs_darwin_lowfreq]: the full-Maxwell (Yee) branch and the
// spectral Darwin branch must agree in the low-frequency regime where the
// Darwin approximation is valid (ω² ≪ k²c², i.e. displacement current
// negligible). Vehicle: a cold-plasma parallel WHISTLER eigenmode.
//
// Setup: uniform cold electrons in B0 = wce x̂, seeded with a right-hand
// circular transverse velocity perturbation at a single mode,
//   u_y = +v1 cos(kx),  u_z = -v1 sin(kx),
// which rotates rigidly at the whistler frequency. Cold R-mode dispersions
// (ω_pe = 1, electron-only):
//   Maxwell:  c²k² = ω² + ω/(ω_ce - ω)
//   Darwin :  c²k² =      ω/(ω_ce - ω)   →  ω = ω_ce k²c²/(1 + k²c²)
// At k c = 3, ω_ce = 0.5 these are 0.4478 vs 0.4500 — a 0.5% model gap.
//
// Measurement is IDENTICAL for both codes and involves no field staggering:
// the complex particle-current projection C(t) = Σ_p (u_y + i u_z) e^{-ikx_p}
// rotates as e^{∓iωt}; ω = |LSQ slope| of the unwrapped phase over ~5 whistler
// periods. (The Yee cold start also excites the two fast EM branches at
// |ω| ≈ 3.2, but they carry only ~7% of the velocity amplitude and average
// out of the phase fit.)
//
// PASS: each measured ω within 2% of its own analytic dispersion, Yee vs
// Darwin within 2%, and the Yee run's running conservation residuals
// (rms divB, Gauss-residual drift — MaxwellSimulation::residuals()) < 1e-5.

#include "pic/config.hpp"
#include "pic/simulation.hpp"
#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace arc;

static const double TwoPi = 6.28318530717958647692;

// add the rotating transverse velocity perturbation on the host
static void perturb(Particles& P, const Grid& g, double k, double v1) {
    const size_t n = P.n;
    std::vector<float> x(n), uy(n), uz(n);
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(), P.uy.data(), n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz.data(), P.uz.data(), n * 4, cudaMemcpyDeviceToHost));
    for (size_t t = 0; t < n; ++t) {
        const double th = k * (double)x[t] * g.dx;
        uy[t] += (float)(v1 * std::cos(th));
        uz[t] -= (float)(v1 * std::sin(th));
    }
    CUDA_CHECK(cudaMemcpy(P.uy.data(), uy.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.uz.data(), uz.data(), n * 4, cudaMemcpyHostToDevice));
}

// complex mode projection of the particle transverse velocity
static std::complex<double> project(Particles& P, const Grid& g, double k) {
    const size_t n = P.n;
    std::vector<float> x(n), uy(n), uz(n);
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(), P.uy.data(), n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz.data(), P.uz.data(), n * 4, cudaMemcpyDeviceToHost));
    std::complex<double> c = 0;
    for (size_t t = 0; t < n; ++t) {
        const double th = k * (double)x[t] * g.dx;
        // the seed writes u_y + i u_z = v1 e^{-ikx}; conjugate-match it
        c += std::complex<double>(uy[t], uz[t])
           * std::complex<double>(std::cos(th), std::sin(th));
    }
    return c;
}

// |slope| of the unwrapped phase of C_j sampled at uniform interval dt_s
static double fit_omega(const std::vector<std::complex<double>>& C, double dt_s) {
    const int m = (int)C.size();
    std::vector<double> ph(m);
    double prev = std::arg(C[0]), acc = prev;
    ph[0] = acc;
    for (int j = 1; j < m; ++j) {
        const double a = std::arg(C[j]);
        double d = a - prev;
        while (d >  M_PI) d -= TwoPi;
        while (d < -M_PI) d += TwoPi;
        acc += d; prev = a; ph[j] = acc;
    }
    double st = 0, sp = 0, stt = 0, stp = 0;              // LSQ on (t_j, ph_j)
    for (int j = 0; j < m; ++j) {
        const double t = j * dt_s;
        st += t; sp += ph[j]; stt += t * t; stp += t * ph[j];
    }
    return std::fabs((m * stp - st * sp) / (m * stt - st * st));
}

int main() {
    const int    nx = 128, ny = 1, ppc = 64, mode = 2;
    const double c = 10.0, wce = 0.5, uth = 0.01, v1 = 0.05;
    const double k  = 0.3;                     // k c = 3, ω/kc ≈ 0.15: low-freq
    const double Lx = TwoPi * mode / k;
    const double tend = 70.0;                  // ≈ 5 whistler periods
    Grid g(nx, ny, Lx, Lx / nx);

    // analytic dispersions (ω_pe = 1)
    const double kc2 = k * k * c * c;
    const double w_dar = wce * kc2 / (1.0 + kc2);
    double w_max = w_dar;                      // Newton on f = c²k² - ω² - ω/(wce-ω)
    for (int it = 0; it < 50; ++it) {
        const double f  = kc2 - w_max * w_max - w_max / (wce - w_max);
        const double df = -2.0 * w_max - wce / ((wce - w_max) * (wce - w_max));
        w_max -= f / df;
    }

    SpeciesList sp = { Species{"e", 1.0, ppc, {uth, uth, uth}, {0.0, 0.0, 0.0}} };

    // ---- Darwin branch ----
    double w_meas_dar;
    {
        RunParams rp;
        rp.dt = 0.05; rp.c = c; rp.ndc = 2; rp.qm = -1.0; rp.eps0 = 1.0;
        rp.B0[0] = (float)wce; rp.wce = wce;
        rp.noisy_load = false; rp.dump_every = 0;
        const long nsteps = (long)std::lround(tend / rp.dt);
        const int  stride = 2;
        rp.nsteps = nsteps;

        Simulation<CfgDarwin> sim(g, rp, sp);
        sim.init();
        perturb(sim.particles(), g, k, v1);
        std::vector<std::complex<double>> C;
        for (long n = 0; n < nsteps; ++n) {
            sim.step(n);
            if (n % stride == 0) {
                cudaDeviceSynchronize();
                C.push_back(project(sim.particles(), g, k));
            }
        }
        w_meas_dar = fit_omega(C, stride * rp.dt);
    }

    // ---- Yee (full Maxwell) branch ----
    double w_meas_yee, divb0, divb1, gauss1;
    {
        RunParams rp;
        rp.dt = 0.02; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
        rp.B0[0] = (float)wce; rp.wce = wce;
        rp.noisy_load = false; rp.dump_every = 0;
        rp.jfilter = 0; rp.tile_sort = 0;
        const long nsteps = (long)std::lround(tend / rp.dt);
        const int  stride = 5;
        rp.nsteps = nsteps;

        MaxwellSimulation sim(g, rp);
        sim.particles().initialize(sp, g, rp, sim.stream());
        sim.stream().synchronize();
        perturb(sim.particles(), g, k, v1);
        divb0 = sim.residuals().divb_rms;      // also captures Gauss reference
        std::vector<std::complex<double>> C;
        for (long n = 0; n < nsteps; ++n) {
            sim.step();
            if (n % stride == 0) {
                cudaDeviceSynchronize();
                C.push_back(project(sim.particles(), g, k));
            }
        }
        w_meas_yee = fit_omega(C, stride * rp.dt);
        const auto r = sim.residuals();
        divb1 = r.divb_rms; gauss1 = r.gauss_drift_rms;
    }

    const double e_dar  = std::fabs(w_meas_dar - w_dar) / w_dar;
    const double e_yee  = std::fabs(w_meas_yee - w_max) / w_max;
    const double e_x    = std::fabs(w_meas_yee - w_meas_dar) / w_meas_dar;
    bool ok = true;
    std::printf("whistler k=%.3f kc=%.1f wce=%.2f\n", k, k * c, wce);
    std::printf("  analytic: darwin %.5f  maxwell %.5f (gap %.2f%%)\n",
                w_dar, w_max, 100.0 * std::fabs(w_max - w_dar) / w_dar);
    std::printf("  measured: darwin %.5f (err %.2f%%)  yee %.5f (err %.2f%%)  "
                "cross %.2f%%\n", w_meas_dar, 100 * e_dar, w_meas_yee, 100 * e_yee,
                100 * e_x);
    std::printf("  yee residuals: divB rms %.3e -> %.3e, gauss drift %.3e\n",
                divb0, divb1, gauss1);
    if (e_dar > 0.02)  { std::printf("FAIL: darwin vs analytic\n");  ok = false; }
    if (e_yee > 0.02)  { std::printf("FAIL: yee vs analytic\n");     ok = false; }
    if (e_x   > 0.02)  { std::printf("FAIL: yee vs darwin\n");       ok = false; }
    if (divb1 > 1e-5)  { std::printf("FAIL: divB residual\n");       ok = false; }
    if (gauss1 > 1e-5) { std::printf("FAIL: gauss drift\n");         ok = false; }
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
