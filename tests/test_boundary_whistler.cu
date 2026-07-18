// M2 gate [boundary_reflection_whistler]: quasi-parallel whistler reflection
// off the x-end damping layers, measured in plasma, must be below 1%.
//
// Trimmed from tools/boundary_reflection.cu (the study tool; see
// docs/BOUNDARY_STUDY.md for the R(ω, nd, ν_max) map). Config = the
// PRODUCTION recommendation: hybrid masks (fields + layer-particle
// transverse momentum), nd = 256, ν_max = 0.1 — R < 1% across the whole
// 0.1–0.5 ω_ce band (0.84% at this test's 0.4 ω_ce, 2026-07-17, with the
// honest late window opening just after the outbound pass).
//
// Method: antenna column at the domain center radiates an R-mode whistler
// packet (trapezoid envelope, 6 periods); a probe column between antenna and
// layer tracks the NARROWBAND complex envelope at the drive frequency
// (demodulation, both helicities, one-pole low-pass). R = late-window max /
// outbound-window max, windows timed from the cold whistler group velocity.

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const double w0 = 0.2, wce = 0.5, c = 10.0, numax = 0.1;
    const int nx = 4096, ny = 1, nd = 256, ppc = 100, ncyc = 6;
    const double dx = 1.0;
    Grid g(nx, ny, nx * dx, dx);

    const double kc2 = w0 * w0 + w0 / (wce - w0);
    const double k = std::sqrt(kc2) / c;
    const double vg = 2.0 * c * c * k / (2.0 * w0 + wce / ((wce - w0) * (wce - w0)));
    const double Tw = 2.0 * M_PI / w0;

    RunParams rp;
    rp.dt = 0.4 * dx / c; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.noisy_load = false; rp.dump_every = 0;
    rp.bnd_x = 2; rp.bnd_nd = nd; rp.bnd_numax = numax;
    rp.ant_amp = 1e-2; rp.ant_x0 = nx / 2.0; rp.ant_sigma = 2.0;
    rp.ant_w0 = w0; rp.ant_trmp = 2.0 * Tw; rp.ant_toff = ncyc * Tw;

    const int    xp = (nx / 2 + (nx - nd)) / 2;
    const double d_ant_probe   = xp - nx / 2.0;
    const double d_probe_layer = (nx - nd) - xp;
    const double t_out_end = rp.ant_toff + d_ant_probe / vg + 4.0 * Tw;
    const double t_ref_beg = rp.ant_toff + (d_ant_probe + 2.0 * d_probe_layer) / vg
                           + 2.0 * Tw;      // open just past the outbound pass
    const double t_end = t_ref_beg + 14.0 * Tw;
    const long   nsteps = (long)(t_end / rp.dt);

    SpeciesList sp = { Species{"e", 1.0, ppc, {0.01, 0.01, 0.01}, {0, 0, 0}} };
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    sim.stream().synchronize();

    std::vector<float> by(g.real_size()), bz(g.real_size());
    const size_t nb = (size_t)g.real_size() * sizeof(float);
    double a_inc = 0, a_ref = 0, zpr = 0, zpi = 0, zmr = 0, zmi = 0;
    for (long n = 0; n < nsteps; ++n) {
        sim.step();
        if (n % 10 != 0) continue;
        const double t = n * rp.dt;
        CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), nb, cudaMemcpyDeviceToHost));
        const double cw = std::cos(w0 * t), sw = std::sin(w0 * t);
        const double al = 10.0 * rp.dt / Tw;
        const double bY = by[xp], bZ = bz[xp];
        zpr += al * ((bY * cw - bZ * sw) - zpr);
        zpi += al * ((bY * sw + bZ * cw) - zpi);
        zmr += al * ((bY * cw + bZ * sw) - zmr);
        zmi += al * ((-bY * sw + bZ * cw) - zmi);
        const double bp = std::max(std::hypot(zpr, zpi), std::hypot(zmr, zmi));
        if (t < t_out_end)       a_inc = std::max(a_inc, bp);
        else if (t >= t_ref_beg) a_ref = std::max(a_ref, bp);
    }
    const double R = a_inc > 0 ? a_ref / a_inc : 1.0;
    const bool ok = (R < 0.01) && (a_inc > 1e-4);   // signal well above noise
    std::printf("whistler w0=%.2f wce (nd=%d numax=%g hybrid): A_inc=%.3e A_ref=%.3e R=%.4f\n",
                w0 / wce, nd, numax, a_inc, a_ref, R);
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
