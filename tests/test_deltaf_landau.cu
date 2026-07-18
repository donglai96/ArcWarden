// M3 gate [deltaf_landau]: Landau damping of a Langmuir wave on the delta-f
// Yee branch vs kinetic theory.
//
// k lambda_D = 0.4: plasma dispersion function root (scipy, |D| ~ 1e-15,
// 2026-07-17):  w = 1.28506,  gamma = -6.6128e-2  (wpe = 1).
//
// Seed: linearized longitudinal current perturbation at one k,
//   wd = v1 cos(kx) u_x / Tpar
// (the delta-f image of a velocity shift u_x += v1 cos(kx): delta-rho = 0,
// E(0) = 0, Gauss-consistent). The Langmuir oscillation then rings up and
// Landau-damps; fit w from the phase rotation of the complex Ex mode pair
// (the standing wave is two counter-propagating modes; project at +k) and
// gamma from the log-amplitude slope.
//
// B0 = 0 (unmagnetized; the weight equation needs no gyrotropy then).
// Resonance at v_phi = 3.2 sigma with band width ~ gamma/(k uth) = 0.17
// sigma: like the anisotropy gate this is marker-limited; ppc = 3200 puts
// the fit inside 10%.

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace arc;

static const double TwoPi = 6.28318530717958647692;

int main() {
    const int nx = 96, ny = 1, ppc = 3200, mode = 3;
    const double uth = 0.05, c = 1.0;
    const double Lx = 3.0 * TwoPi / 8.0;            // mode 3 -> k = 8 exactly
    const double dx = Lx / nx, dt = 0.01;
    const double k = TwoPi * mode / Lx;              // 8.0; k lambda_D = 0.4
    const double w_th = 1.28506, g_th = -6.6128e-2;
    Grid g(nx, ny, Lx, dx);

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.noisy_load = false; rp.dump_every = 0;
    rp.deltaf = 1; rp.df_tpar = uth * uth; rp.df_tperp = uth * uth;

    SpeciesList sp = { Species{"e", 1.0, ppc, {uth, uth, uth}, {0, 0, 0}, true} };
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    sim.particles().enable_deltaf(sim.stream());
    sim.stream().synchronize();
    if (getenv("DFL_ADDR")) {
        YeeFields& f = sim.fields();
        std::printf("ex=%p ey=%p ez=%p bx=%p by=%p bz=%p jx=%p jy=%p jz=%p\n",
            (void*)f.ex_.data(), (void*)f.ey_.data(), (void*)f.ez_.data(),
            (void*)f.bx_.data(), (void*)f.by_.data(), (void*)f.bz_.data(),
            (void*)f.jx_.data(), (void*)f.jy_.data(), (void*)f.jz_.data());
        std::fflush(stdout);
    }

    // delta-f seed of the longitudinal current perturbation
    {
        Particles& P = sim.particles();
        const size_t n = P.n;
        const double v1 = 1e-4, T = uth * uth;
        std::vector<float> x(n), ux(n), wd(n);
        CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), P.ux.data(), n * 4, cudaMemcpyDeviceToHost));
        for (size_t t = 0; t < n; ++t)
            wd[t] = (float)(v1 * std::cos(k * (double)x[t] * dx) * ux[t] / T);
        CUDA_CHECK(cudaMemcpy(P.wd.data(), wd.data(), n * 4, cudaMemcpyHostToDevice));
    }

    // complex Ex mode amplitude at +k (Ex sites at i+1/2)
    std::vector<float> ex(g.real_size());
    const int stride = 5;
    const long nsteps = 5000;                        // t = 50 ~ 3.3 e-folds
    std::vector<double> ts, lamp, ph;
    std::complex<double> prev = 0; double acc = 0;
    for (long s = 1; s <= nsteps; ++s) {
        sim.step();
        if (s % stride) continue;
        CUDA_CHECK(cudaMemcpy(ex.data(), sim.fields().ex_.data(), ex.size() * 4,
                              cudaMemcpyDeviceToHost));
        std::complex<double> C = 0;
        for (int i = 0; i < nx; ++i) {
            const double th = k * (i + 0.5) * dx;
            C += (double)ex[i] * std::complex<double>(std::cos(th), std::sin(th));
        }
        const double t = s * dt;
        if (t > 5.0) {                               // skip the ring-up transient
            if (prev != std::complex<double>(0.0)) acc += std::arg(C / prev);
            prev = C;
            ts.push_back(t); lamp.push_back(std::log(std::abs(C) + 1e-300)); ph.push_back(acc);
        }
    }
    const int m = (int)ts.size();
    auto fit = [&](const std::vector<double>& y) {
        double st = 0, sy = 0, stt = 0, sty = 0;
        for (int i = 0; i < m; ++i) { st += ts[i]; sy += y[i]; stt += ts[i]*ts[i]; sty += ts[i]*y[i]; }
        return (m * sty - st * sy) / (m * stt - st * st);
    };
    const double w_fit = std::fabs(fit(ph));
    const double g_fit = fit(lamp);
    const double werr = std::fabs(w_fit - w_th) / w_th;
    const double gerr = std::fabs(g_fit - g_th) / std::fabs(g_th);
    std::printf("deltaf Landau (k lambda_D = 0.4): w = %.5f vs %.5f (%.2f%%), "
                "gamma = %.4e vs %.4e (%.1f%%)\n",
                w_fit, w_th, 100 * werr, g_fit, g_th, 100 * gerr);
    const bool ok = werr < 0.02 && gerr < 0.10;
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
