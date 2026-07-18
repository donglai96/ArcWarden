// M3 gate [deltaf_vs_fullf_consistency]: the SAME physical perturbation on
// the SAME Maxwellian must evolve identically under the two representations
// — the plan's designated bug-catcher (防 bug 重点).
//
// The perturbation is a linearized R-mode transverse current at one k:
//   full-f:  velocity shift  u_y += v1 cos(kx), u_z -= v1 sin(kx)
//   delta-f: weight seed     wd  = v1 (u_y cos(kx) - u_z sin(kx)) / Tperp
// (the linearization of the same shifted Maxwellian: δf = v1(uy cos - uz sin)
// f0/Tperp; δρ = 0 and E(0) = 0, so both starts are Gauss-consistent).
//
// Isotropic warm Maxwellian in B0 = wce x̂ (uth = 0.065, k = 0.8, c = 1):
// the seeded whistler is kinetic-modified (theory root w = 0.08792,
// gamma = -1.296e-3, v_res = -3.1 sigma — scipy, 2026-07-17). Both
// representations must land on the same kinetic frequency and damping.
// Measurement hygiene, each item found the hard way:
//   - project the particle CURRENT, not B (B weights the fast EM branch ~7x
//     more heavily);
//   - boxcar the complex mode series over one fast-EM period (the seed also
//     puts ~10% into the undamped EM branch at w = 1.28);
//   - seed large enough that the full-f signal stays above the thermal
//     fluctuation level the quiet start relaxes toward (sqrt(N) uth);
//   - keep the branches far apart in frequency (k = 0.8, not 1.96).
// This gate also caught a REAL integrator defect: driving the weight with
// post-push u alone (half step out of time-center with the fields)
// numerically anti-damps the fast EM branch — fixed by centering u at t^n
// in yee_advance_particle.
// PASS: |w_df - w_ff|/w < 2%, |slope_df - slope_ff| < 0.25 |gamma_th|, and
// w_df within 10% of the kinetic theory root. The decay slopes are compared
// to EACH OTHER, not to gamma_th: the seed amplitude needed to hold the
// full-f signal above its thermal-fluctuation floor sits above the O'Neil
// trapping threshold for this small gamma, so cyclotron damping saturates
// nonlinearly — identically in both representations (measured -3.3e-4 vs
// -4.5e-4 against a linear -1.30e-3). Linear damping vs theory is gated by
// deltaf_landau.

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace arc;

static const double TwoPi = 6.28318530717958647692;

struct ModeTrace { double w, slope; };

// run one representation and fit w, damping slope of the seeded mode
static ModeTrace run_rep(bool use_df, int nx, double dx, double dt, double wce,
                         double uth, double v1, int mode, int ppc, long nsteps) {
    Grid g(nx, 1, nx * dx, dx);
    const double k = TwoPi * mode / (nx * dx);
    RunParams rp;
    rp.dt = dt; rp.c = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.noisy_load = false; rp.dump_every = 0;
    if (use_df) { rp.deltaf = 1; rp.df_tpar = uth * uth; rp.df_tperp = uth * uth; }

    SpeciesList sp = { Species{"e", 1.0, ppc, {uth, uth, uth}, {0, 0, 0}, use_df} };
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    if (use_df) sim.particles().enable_deltaf(sim.stream());
    sim.stream().synchronize();

    // seed the SAME physical perturbation, representation-appropriately
    Particles& P = sim.particles();
    const size_t n = P.n;
    std::vector<float> x(n), uy(n), uz(n);
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(), P.uy.data(), n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz.data(), P.uz.data(), n * 4, cudaMemcpyDeviceToHost));
    if (use_df) {
        std::vector<float> wd(n);
        const double T = uth * uth;
        for (size_t t = 0; t < n; ++t) {
            const double th = k * (double)x[t] * dx;
            wd[t] = (float)(v1 * (uy[t] * std::cos(th) - uz[t] * std::sin(th)) / T);
        }
        CUDA_CHECK(cudaMemcpy(P.wd.data(), wd.data(), n * 4, cudaMemcpyHostToDevice));
    } else {
        for (size_t t = 0; t < n; ++t) {
            const double th = k * (double)x[t] * dx;
            uy[t] += (float)(v1 * std::cos(th));
            uz[t] -= (float)(v1 * std::sin(th));
        }
        CUDA_CHECK(cudaMemcpy(P.uy.data(), uy.data(), n * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(P.uz.data(), uz.data(), n * 4, cudaMemcpyHostToDevice));
    }

    // complex trajectory of the seeded transverse CURRENT mode — the particle
    // projection weights the whistler ~7x more strongly relative to the fast
    // EM branches than the B projection does (B/u differs per branch), which
    // is what makes the single-mode fit clean (same measurement as the
    // validated yee_vs_darwin_lowfreq gate). Each representation projects its
    // own physical perturbation current: delta-f = Sum wd (uy+iuz) e^{+ikx};
    // full-f = Sum (uy+iuz) e^{+ikx} (the f0 part phase-cancels).
    std::vector<float> wdh(use_df ? n : 0);
    const int stride = 10;                            // sample every 0.25/wpe
    std::vector<double> rts;
    std::vector<std::complex<double>> raw;
    for (long s = 1; s <= nsteps; ++s) {
        sim.step();
        if (s % stride) continue;
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), P.uy.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), P.uz.data(), n * 4, cudaMemcpyDeviceToHost));
        if (use_df)
            CUDA_CHECK(cudaMemcpy(wdh.data(), P.wd.data(), n * 4, cudaMemcpyDeviceToHost));
        std::complex<double> C = 0;
        for (size_t t2 = 0; t2 < n; ++t2) {
            const double th = k * (double)x[t2] * dx;
            const double f = use_df ? (double)wdh[t2] : 1.0;
            C += f * std::complex<double>(uy[t2], uz[t2])
               * std::complex<double>(std::cos(th), std::sin(th));
        }
        rts.push_back(s * dt); raw.push_back(C);
    }
    // boxcar over +-half the fast-EM period (w_EM = 1.28 -> T = 4.9 = 20
    // samples): nulls the fast-branch beat, leaves the whistler untouched
    const int hw = 10;
    std::vector<double> ts, lamp, ph;
    std::complex<double> prev = 0; double acc = 0;
    for (size_t i = hw; i + hw < raw.size(); ++i) {
        std::complex<double> C = 0;
        for (int j = -hw; j <= hw; ++j) C += raw[i + j];
        C /= (double)(2 * hw + 1);
        if (rts[i] <= 100.0) continue;                // ring-up transient
        if (prev != std::complex<double>(0.0)) acc += std::arg(C / prev);
        prev = C;
        ts.push_back(rts[i]); lamp.push_back(std::log(std::abs(C) + 1e-300)); ph.push_back(acc);
        if (getenv("DFC_DEBUG"))
            std::fprintf(stderr, "%d %g %.6f %.6f\n", use_df ? 1 : 0, rts[i], lamp.back(), acc);
    }
    const int m = (int)ts.size();
    auto fit = [&](const std::vector<double>& y) {
        double st = 0, sy = 0, stt = 0, sty = 0;
        for (int i = 0; i < m; ++i) { st += ts[i]; sy += y[i]; stt += ts[i]*ts[i]; sty += ts[i]*y[i]; }
        return (m * sty - st * sy) / (m * stt - st * st);
    };
    return { std::fabs(fit(ph)), fit(lamp) };
}

int main() {
    const int nx = 128, ppc = 1600, mode = 2;
    double dt = 0.025;
    if (getenv("DFC_DT")) dt = atof(getenv("DFC_DT"));
    const double dx = 15.70796 / 128, wce = 0.25, uth = 0.065, v1 = 3e-3;
    const long nsteps = (long)(800.0 / dt);   // t = 800 ~ 1.04 damping e-folds

    const ModeTrace df = run_rep(true,  nx, dx, dt, wce, uth, v1, mode, ppc, nsteps);
    const ModeTrace ff = run_rep(false, nx, dx, dt, wce, uth, v1, mode, ppc, nsteps);

    const double w_th = 0.08792, g_th = -1.296e-3;
    const double werr = std::fabs(df.w - ff.w) / ff.w;
    const double sdiff = std::fabs(df.slope - ff.slope);
    const double therr = std::fabs(df.w - w_th) / w_th;
    std::printf("deltaf vs fullf (same whistler current seed; theory w=%.5f g=%.3e):\n",
                w_th, g_th);
    std::printf("  w:     df %.5f  ff %.5f  (diff %.2f%%, df vs theory %.1f%%)\n",
                df.w, ff.w, 100 * werr, 100 * therr);
    std::printf("  slope: df %.4e  ff %.4e  (|diff| %.2e vs gate %.2e)\n",
                df.slope, ff.slope, sdiff, 0.25 * std::fabs(g_th));
    bool ok = true;
    if (werr > 0.02) { std::printf("FAIL: frequency mismatch across reps\n"); ok = false; }
    if (sdiff > 0.25 * std::fabs(g_th)) {
        std::printf("FAIL: damping mismatch across reps\n"); ok = false;
    }
    if (therr > 0.10) { std::printf("FAIL: df frequency vs kinetic theory\n"); ok = false; }
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
