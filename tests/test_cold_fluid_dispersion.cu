// M4 gate C — linearized cold-electron fluid on the Yee branch: a seeded
// whistler eigenmode must oscillate at the cold R-mode frequency
//     D(w) = w^2 - c^2 k^2 - w * wpe^2/(w - wce) = 0   (whistler root w < wce)
// with wpe^2 = cold_nc (no kinetic particles: the fluid carries the whole
// plasma response, chirp1d architecture).
//
// Complex transverse convention (B0 || x, electron sense): F = Fy + i Fz,
// mode ~ e^{i(w t - k x)}. Eigenmode ratios (qm = -1):
//   v = v1 e^{-ikx},  E = -i(w - wce) v,  B = i k E / w.
// A vc-ONLY seed does NOT work: it projects onto all three R-mode branches
// (here w = {+0.076, +1.34, -1.21}) and the phase slope measures the beat —
// first version of this test failed exactly that way. Seed the full
// eigenmode (with Yee half-cell x-stagger and -dt/2 time stagger on B) and
// boxcar the projection over the fast-branch period to kill the residual.
// Measure: phase slope of P(t) = sum_x (vcy + i vcz) e^{+ikx}  ->  +w.
// Theory uses the Yee-discrete k_eff = sin(k dx/2)/(dx/2); gate 1%.

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int nx = 128, ny = 1, mode = 4;
    const double dx = 0.25, dt = 0.05, c = 1.0, wce = 0.2, nc = 0.994;
    Grid g(nx, ny, nx * dx, dx);

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.cold_nc = nc; rp.dump_every = 0;

    const double k    = 2.0 * M_PI * mode / (nx * dx);
    const double keff = std::sin(0.5 * k * dx) / (0.5 * dx);

    // whistler root of D(w) by bisection on (0, wce)
    auto D = [&](double w) { return w * w - c * c * keff * keff - w * nc / (w - wce); };
    double lo = 1e-6, hi = wce - 1e-6;
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (D(lo) * D(mid) <= 0) hi = mid; else lo = mid;
    }
    const double wth = 0.5 * (lo + hi);

    MaxwellSimulation sim(g, rp);
    const double v1 = 1e-4;
    const std::complex<double> I(0, 1);
    const std::complex<double> Ec = -I * (wth - wce) * v1;      // E amplitude
    const std::complex<double> Bc = I * k * Ec / wth
                                  * std::exp(-I * wth * 0.5 * dt);  // B at t=-dt/2
    std::vector<float> vy(g.real_size()), vz(g.real_size()),
                       ey(g.real_size()), ez(g.real_size()),
                       by(g.real_size()), bz(g.real_size());
    for (int i = 0; i < nx; ++i) {
        const std::complex<double> pn = std::exp(-I * (k * i * dx));         // nodes
        const std::complex<double> pf = std::exp(-I * (k * (i + 0.5) * dx)); // faces
        vy[i] = (float)std::real(v1 * pn); vz[i] = (float)std::imag(v1 * pn);
        ey[i] = (float)std::real(Ec * pn); ez[i] = (float)std::imag(Ec * pn);
        by[i] = (float)std::real(Bc * pf); bz[i] = (float)std::imag(Bc * pf);
    }
    CUDA_CHECK(cudaMemcpy(sim.vcy().data(),        vy.data(), vy.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.vcz().data(),        vz.data(), vz.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().ey_.data(), ey.data(), ey.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().ez_.data(), ez.data(), ez.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), by.data(), by.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), bz.data(), bz.size() * 4, cudaMemcpyHostToDevice));

    // ~10 whistler periods, projection every 2 steps, boxcar over the fast
    // branch (|w| ~ 1.2-1.3, period ~5/wpe -> 50-step window = 25 samples)
    const double T = 10.0 * 2.0 * M_PI / wth;
    const long  nsteps = (long)(T / dt);
    const int   stride = 2, bw = 25;
    std::vector<std::complex<double>> Ps; std::vector<double> Ts;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n % stride) continue;
        CUDA_CHECK(cudaMemcpy(vy.data(), sim.vcy().data(), vy.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vz.data(), sim.vcz().data(), vz.size() * 4, cudaMemcpyDeviceToHost));
        std::complex<double> P(0, 0);
        for (int i = 0; i < nx; ++i)
            P += std::complex<double>(vy[i], vz[i]) * std::exp(I * (k * i * dx));
        Ps.push_back(P); Ts.push_back(n * dt);
    }
    // boxcar + unwrapped phase + LSQ slope
    std::vector<double> ts, ph;
    std::complex<double> prev(0, 0); double acc = 0;
    for (std::size_t s = bw; s + bw < Ps.size(); s += bw / 2) {
        std::complex<double> Pm(0, 0);
        for (int j = -(bw / 2); j <= bw / 2; ++j) Pm += Ps[s + j];
        if (prev != std::complex<double>(0, 0)) {
            acc += std::arg(Pm / prev);
            ts.push_back(Ts[s]); ph.push_back(acc);
        }
        prev = Pm;
    }
    double st = 0, sp = 0, stt = 0, stp = 0; const int m = (int)ts.size();
    for (int i = 0; i < m; ++i) {
        st += ts[i]; sp += ph[i]; stt += ts[i] * ts[i]; stp += ts[i] * ph[i];
    }
    const double wm = (m * stp - st * sp) / (m * stt - st * st);   // +w (e^{+iwt})
    const double err = std::fabs(wm - wth) / wth;
    std::printf("cold-fluid whistler: w_meas = %.6f vs theory %.6f (k = %.4f, "
                "keff = %.4f, err %.3f%%)\n", wm, wth, k, keff, 100 * err);
    const bool ok = err < 0.01;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
