// M4 gate B — delta-f × background B0(x): the (E,mu)-mapped mirror
// equilibrium must LOAD correctly and HOLD under the delta-f weight equation
// with the local Tperp(x) drive while markers bounce in the parabolic bottle.
//
// Load checks (initialize_mirror, per 16-cell chunk vs analytic):
//   n(x)/n_eq = Tperp(x)/Tperp_eq              (marker counts)
//   <u_perp^2>/2 = Tperp(x),  <u_par^2> = Tpar  (velocity moments)
// Hold check (test_deltaf_equilibrium analog): whistler-STABLE anisotropy
// Tperp < Tpar, small transverse-B seed (1e-6), run ~2 thermal bounce periods
// (markers traverse the full mirror geometry, exercising the x-dependent
// drive): wd must stay BOUNDED — no secular growth (second half vs first
// half), max|wd| small. A wrong Tperp(x) mapping or a mirror force leaking
// into the wave force F shows up as systematic weight accumulation along
// bounce orbits.
//
// Flat path only (mirror runs are ny = 1; production tiling is for true 2D).

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;

int main() {
    const int nx = 256, ny = 1, ppc = 400;
    const double c = 1.0, wce = 0.25, dx = 0.2, dt = 0.05;
    const double upar = 0.1, uperp = 0.05;          // A = 0.25: whistler-stable
    const double a = 2e-3;                          // edge mirror ratio 2.31
    Grid g(nx, ny, nx * dx, dx);
    const double xc = 0.5 * nx * dx;

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.b0_prof = 1; rp.b0_a = a; rp.b0_xc = xc;
    rp.noisy_load = true; rp.dump_every = 0; rp.rng_seed = 20260718UL;
    rp.deltaf = 1; rp.df_tpar = upar * upar; rp.df_tperp = uperp * uperp;

    Species sp{"e", 1.0, ppc, {upar, uperp, uperp}, {0, 0, 0}, true};
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(sp, g, rp, sim.stream());
    sim.particles().enable_deltaf(sim.stream());
    sim.stream().synchronize();
    const std::size_t n = sim.particles().n;

    // ---- load verification: chunked density + velocity moments vs analytic
    {
        std::vector<float> x(n), ux(n), uy(n), uz(n);
        CUDA_CHECK(cudaMemcpy(x.data(),  sim.particles().x.data(),  n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), sim.particles().ux.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), sim.particles().uy.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), sim.particles().uz.data(), n * 4, cudaMemcpyDeviceToHost));
        const int nch = 16, cw = nx / nch;
        std::vector<double> cnt(nch, 0), sup2(nch, 0), spa2(nch, 0);
        for (std::size_t p = 0; p < n; ++p) {
            const int ch = std::min((int)(x[p] / cw), nch - 1);
            cnt[ch] += 1.0;
            sup2[ch] += (double)uy[p] * uy[p] + (double)uz[p] * uz[p];
            spa2[ch] += (double)ux[p] * ux[p];
        }
        const double Tpa = upar * upar, Tpe = uperp * uperp;
        double emax_n = 0, emax_t = 0, emax_p = 0;
        for (int chk = 0; chk < nch; ++chk) {
            const double xm = (chk + 0.5) * cw * dx - xc;
            const double b  = 1.0 + a * xm * xm;
            const double Tp = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);
            const double n_th = ppc * cw * (Tp / Tpe);
            emax_n = std::max(emax_n, std::fabs(cnt[chk] / n_th - 1.0));
            emax_t = std::max(emax_t, std::fabs(0.5 * sup2[chk] / cnt[chk] / Tp  - 1.0));
            emax_p = std::max(emax_p, std::fabs(spa2[chk] / cnt[chk] / Tpa - 1.0));
        }
        std::printf("load: n=%zu  max chunk err: n(x) %.3f%%  Tperp(x) %.2f%%  Tpar %.2f%%\n",
                    n, 100 * emax_n, 100 * emax_t, 100 * emax_p);
        if (emax_n > 0.02 || emax_t > 0.08 || emax_p > 0.08) {
            std::printf("FAIL: load moments off\n"); return 1;
        }
    }

    // small transverse-B seed: stable plasma -> bounded wd response
    std::mt19937 gen(20260718u);
    std::uniform_real_distribution<float> un(-1e-6f, 1e-6f);
    std::vector<float> by(g.real_size()), bz(g.real_size());
    for (auto& v : by) v = un(gen);
    for (auto& v : bz) v = un(gen);
    CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), by.data(), by.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), bz.data(), bz.size() * 4, cudaMemcpyHostToDevice));

    // ~2 thermal bounce periods: wb = sqrt(a) * uperp = 2.24e-3
    const double wb = std::sqrt(a) * uperp;
    const long nsteps = (long)(2.0 * 2.0 * M_PI / wb / dt);
    const int  stride = 2000;
    double rms_half1 = 0, rms_half2 = 0, mx_all = 0; int n1 = 0, n2 = 0;
    for (long s = 1; s <= nsteps; ++s) {
        sim.step();
        if (s % stride) continue;
        const auto st = sim.wd_stats();
        if (s <= nsteps / 2) { rms_half1 += st.rms; ++n1; }
        else                 { rms_half2 += st.rms; ++n2; }
        mx_all = std::max(mx_all, st.max);
    }
    rms_half1 /= n1; rms_half2 /= n2;
    const double growth = rms_half2 / rms_half1;
    std::printf("hold: %ld steps (2 bounce periods), rms(wd) half1 %.3e half2 %.3e "
                "(x%.2f), max|wd| %.3e\n", nsteps, rms_half1, rms_half2, growth, mx_all);

    bool ok = true;
    if (mx_all > 1e-2) { std::printf("FAIL: max|wd| %.3e > 1e-2\n", mx_all); ok = false; }
    if (growth > 3.0)  { std::printf("FAIL: rms(wd) grew x%.2f > 3\n", growth); ok = false; }
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
