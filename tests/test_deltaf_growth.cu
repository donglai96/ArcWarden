// M3 gate [deltaf_anisotropy_growth]: the delta-f Yee branch must reproduce
// the parallel whistler anisotropy-instability growth rate of kinetic linear
// theory for a single bi-Maxwellian population.
//
// Parameters = the Ma et al. case-7 velocity setup (uth_par = 0.01678,
// uth_perp = 0.03752 -> A = 4, wce = 0.25 wpe, c = 1). Kinetic theory
// (plasma dispersion function, single bi-Max population, parallel R-mode:
//   D = w^2 - c^2k^2 + wpe^2[z0 Z(z1) + A(1 + z1 Z(z1))] = 0,
//   z0 = w/(sqrt2 k a_par), z1 = (w - wce)/(sqrt2 k a_par))
// gives gamma_max = 2.89e-3 wpe at k = 1.65 wpe/c (w_r = 0.749 wce)
// [scipy scan 1.0 <= k <= 6.0, 2026-07-17].
//
// The box (Lx = 7.68 c/wpe) puts mode m = 2 at k = 1.636 (theory gamma
// there: 2.872e-3, within 1% of the peak). The m = 2 mode AMPLITUDE grows at
// gamma; fit over the last third of the run (the early record holds the
// eigenmode-selection transient). Measured convergence in ppc — the narrow
// cyclotron-resonant band (v_res = -2.3 sigma_par, width ~ gamma/k ~ 0.1
// sigma) is marker-resolution limited, error ~ 1/sqrt(ppc):
//   ppc  400 -> gamma 2.13e-3 (26% low)
//   ppc 1600 -> gamma 2.60e-3 ( 9% low)   <- gate config, tolerance 15%
//   ppc 6400 -> gamma 2.75e-3 ( 4% low)
// (dt 0.02 vs 0.01 and quiet vs noisy load: no effect; w_r measured 0.18686
// vs theory 0.1868 — the reactive response is exact.)
//
// The delta-f seed: wd(0) = 0 means NOTHING evolves from a quiet start, so
// the instability is seeded by small random By/Bz grid noise (1e-6) — this
// also demonstrates the delta-f noise floor is set by the SEED, not by ppc
// shot noise (the full-f floor at this ppc is orders of magnitude higher).

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace arc;

int main() {
    int nx = 256, ppc = 1600;
    const int ny = 1, mode_fit = 2;
    if (getenv("DF_NX"))  nx  = atoi(getenv("DF_NX"));
    if (getenv("DF_PPC")) ppc = atoi(getenv("DF_PPC"));
    const double c = 1.0, wce = 0.25, apar = 0.01678, aperp = 0.03752;
    double dx = 0.03 * 256.0 / nx, dt = 0.02;      // fixed Lx = 7.68
    if (getenv("DF_DT")) dt = atof(getenv("DF_DT"));
    const double gamma_theory = 2.872e-3;   // at k = 2*pi*2/Lx = 1.636 (header)
    Grid g(nx, ny, nx * dx, dx);

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.noisy_load = getenv("DF_NOISY") != nullptr; rp.dump_every = 0;
    rp.rng_seed = 20260717UL;
    const bool fullf = getenv("DF_FULLF") != nullptr;
    rp.deltaf = fullf ? 0 : 1;
    rp.df_tpar = apar * apar; rp.df_tperp = aperp * aperp;

    SpeciesList sp = { Species{"e", 1.0, ppc, {apar, aperp, aperp}, {0, 0, 0}, true} };
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    if (!fullf) sim.particles().enable_deltaf(sim.stream());
    sim.stream().synchronize();

    // seed: small random transverse B noise
    std::mt19937 gen(20260717u);
    std::uniform_real_distribution<float> un(-1e-6f, 1e-6f);
    std::vector<float> by(g.real_size()), bz(g.real_size());
    for (auto& v : by) v = un(gen);
    for (auto& v : bz) v = un(gen);
    CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), by.data(), by.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), bz.data(), bz.size() * 4, cudaMemcpyHostToDevice));

    // track the m = 2 transverse-B mode power |By_k|^2 + |Bz_k|^2 (host DFT of
    // one row) — cleaner than total W_B, which sums all modes.
    const double k2 = 2.0 * M_PI * mode_fit / (nx * dx);
    const long nsteps = (long)(2400.0 / dt);        // t = 2400/wpe ~ 7 e-folds
    const int  stride = 500;
    std::vector<double> ts, lp;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n % stride) continue;
        CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), by.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
        double cr = 0, ci = 0, dr = 0, di = 0;
        for (int i = 0; i < nx; ++i) {
            const double ph = k2 * (i + 0.5) * dx;   // By,Bz at i+1/2
            cr += by[i] * std::cos(ph); ci -= by[i] * std::sin(ph);
            dr += bz[i] * std::cos(ph); di -= bz[i] * std::sin(ph);
        }
        const double p = cr * cr + ci * ci + dr * dr + di * di;
        ts.push_back(n * dt); lp.push_back(0.5 * std::log(p + 1e-300));
        if (getenv("DF_DEBUG")) std::fprintf(stderr, "%g %.6f %.6f %.6f %.6f %.6f\n",
            n * dt, lp.back(), cr, ci, dr, di);
    }

    // fit gamma over the last third of the record (transient-free window)
    double st = 0, sl = 0, stt = 0, stl = 0; int m = 0;
    for (size_t i = 2 * ts.size() / 3; i < ts.size(); ++i) {
        st += ts[i]; sl += lp[i]; stt += ts[i] * ts[i]; stl += ts[i] * lp[i]; ++m;
    }
    const double gamma_fit = (m * stl - st * sl) / (m * stt - st * st);
    const double lspan = lp.back() - lp.front();
    if (lspan < 3.0) { std::printf("FAIL: only %.1f e-folds of growth\n", lspan); return 1; }
    const double err = std::fabs(gamma_fit - gamma_theory) / gamma_theory;
    std::printf("deltaf whistler anisotropy growth: gamma_fit = %.3e vs theory %.3e "
                "(err %.1f%%, %d pts, %.1f e-folds span)\n",
                gamma_fit, gamma_theory, 100 * err, m, lspan);
    const bool ok = err < 0.15;
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
