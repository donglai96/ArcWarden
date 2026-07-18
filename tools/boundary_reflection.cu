// ArcWarden — M2 boundary study: whistler reflection coefficient R(ω) of the
// x-end damping layers, measured IN PLASMA (the plan's benchmark: the vacuum
// number means little for whistlers, whose wavelength and group velocity in
// the layer depend on the plasma that fills it).
//
// Method: cold magnetized plasma (B0 = wce x̂), antenna column at the domain
// center radiates R-mode whistler packets in ±x for `ncyc` drive periods,
// then turns off. The packets propagate into both damping layers. Incident
// amplitude A_inc = max over time of |B⊥| at a probe column while the packet
// passes OUTBOUND; reflected amplitude A_ref = max at the same probe over a
// LATE window (after the packet died in the layers, any returning signal is
// reflection). R = A_ref / A_inc. Group-velocity timing from the cold
// Maxwell whistler dispersion c²k² = ω² + ω ωpe²/(ωce−ω).
//
// Usage: boundary_reflection [--w0=0.2] [--wce=0.5] [--c=10] [--nx=4096]
//                            [--nd=64] [--numax=1.0] [--ppc=100] [--ncyc=6]
//                            [--mode=2] [--dump=st.bin]
// Prints one line: w0/wce, nd, numax, A_inc, A_ref, R.

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace arc;

int main(int argc, char** argv) {
    double w0 = 0.2, wce = 0.5, c = 10.0, numax = 1.0, theta = 0.0;
    double nedge = 0.25; int taper = 0;
    int nx = 4096, nd = 64, ppc = 100, ncyc = 6, mode = 2;
    std::string dump;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](size_t off) { return std::atof(a.c_str() + off); };
        if      (a.rfind("--w0=", 0) == 0)    w0 = val(5);
        else if (a.rfind("--wce=", 0) == 0)   wce = val(6);
        else if (a.rfind("--c=", 0) == 0)     c = val(4);
        else if (a.rfind("--nx=", 0) == 0)    nx = std::atoi(a.c_str() + 5);
        else if (a.rfind("--nd=", 0) == 0)    nd = std::atoi(a.c_str() + 5);
        else if (a.rfind("--numax=", 0) == 0) numax = val(8);
        else if (a.rfind("--ppc=", 0) == 0)   ppc = std::atoi(a.c_str() + 6);
        else if (a.rfind("--ncyc=", 0) == 0)  ncyc = std::atoi(a.c_str() + 7);
        else if (a.rfind("--mode=", 0) == 0)  mode = std::atoi(a.c_str() + 7);
        else if (a.rfind("--dump=", 0) == 0)  dump = a.substr(7);
        else if (a.rfind("--theta=", 0) == 0) theta = val(8);   // deg, B0 tilt in x-y
        else if (a.rfind("--taper=", 0) == 0) taper = std::atoi(a.c_str() + 8); // cells
        else if (a.rfind("--nedge=", 0) == 0) nedge = val(8);   // taper floor density
    }
    const int ny = 1;
    const double dx = 1.0;
    Grid g(nx, ny, nx * dx, dx);

    // Oblique incidence: tilt B0 by theta in the x-y plane; the antenna
    // radiates k ∥ x̂, so the wave-normal angle w.r.t. B0 is theta and the
    // packet still propagates into the x layers. Quasi-longitudinal cold
    // whistler dispersion: replace wce -> wce·cos(theta) for k, vg timing.
    const double th = theta * M_PI / 180.0;
    const double wcx = wce * std::cos(th);
    const double kc2 = w0 * w0 + w0 / (wcx - w0);
    const double k = std::sqrt(kc2) / c;
    const double vg = 2.0 * c * c * k / (2.0 * w0 + wcx / ((wcx - w0) * (wcx - w0)));
    const double Tw = 2.0 * M_PI / w0;

    RunParams rp;
    rp.dt = 0.4 * dx / c; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)(wce * std::cos(th)); rp.B0[1] = (float)(wce * std::sin(th));
    rp.wce = wce;
    rp.noisy_load = false; rp.dump_every = 0;
    rp.bnd_x = mode; rp.bnd_nd = nd; rp.bnd_numax = numax;
    rp.ant_amp = 1e-2; rp.ant_x0 = nx / 2.0; rp.ant_sigma = 2.0;
    rp.ant_w0 = w0; rp.ant_trmp = 2.0 * Tw; rp.ant_toff = ncyc * Tw;

    // timing: probe halfway between antenna and the right layer. The packet
    // is ncyc·Tw·vg long — keep it well shorter than the probe→layer leg so
    // the outbound and reflected passes are disjoint at the probe.
    const int    xp = (nx / 2 + (nx - nd)) / 2;
    const double d_ant_probe   = xp - nx / 2.0;
    const double d_probe_layer = (nx - nd) - xp;
    const double t_out_end = rp.ant_toff + d_ant_probe / vg + 4.0 * Tw;
    // reflected arrival can be EARLIER than the uniform-vg estimate when a
    // low-density taper speeds the wave up — open the late window just past
    // the outbound pass and keep it long
    const double t_ref_beg = rp.ant_toff + (d_ant_probe + 2.0 * d_probe_layer) / vg
                           + 2.0 * Tw;
    const double t_end     = t_ref_beg + 14.0 * Tw;
    const long   nsteps     = (long)(t_end / rp.dt);

    SpeciesList sp = { Species{"e", 1.0, ppc, {0.01, 0.01, 0.01}, {0, 0, 0}} };

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    sim.stream().synchronize();

    // density-gradient study: rescale per-particle weight by a cosine taper
    // n(x) from 1 down to nedge over `taper` cells in front of each layer
    // (and nedge throughout the layer). Cold static profile with E(0)=0 is
    // force-free (Gauss reference absorbs it); fields see the exact n(x).
    if (taper > 0) {
        Particles& P = sim.particles();
        const size_t n = P.n;
        std::vector<float> x(n), w(n);
        CUDA_CHECK(cudaMemcpy(x.data(), P.x.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w.data(), P.w.data(), n * 4, cudaMemcpyDeviceToHost));
        for (size_t t = 0; t < n; ++t) {
            const double e = std::min((double)x[t], nx - (double)x[t]);
            double f = 1.0;
            if (e < nd)               f = nedge;
            else if (e < nd + taper)  f = nedge + (1.0 - nedge)
                                        * 0.5 * (1.0 - std::cos(M_PI * (e - nd) / taper));
            w[t] *= (float)f;
        }
        CUDA_CHECK(cudaMemcpy(P.w.data(), w.data(), n * 4, cudaMemcpyHostToDevice));
    }

    std::printf("# whistler reflection: w0=%.4g (%.2f wce) theta=%g  k=%.4g "
                "(lambda=%.0f dx)  vg=%.3g  nd=%d numax=%g  nsteps=%ld\n",
                w0, w0 / wce, theta, k, 2 * M_PI / k / dx, vg, nd, numax, nsteps);

    std::vector<float> by(g.real_size()), bz(g.real_size()), bp_row(nx);
    const size_t nb = (size_t)g.real_size() * sizeof(float);
    std::ofstream df;
    if (!dump.empty()) df.open(dump, std::ios::binary);
    double a_inc = 0, a_ref = 0, zpr = 0, zpi = 0, zmr = 0, zmi = 0;
    for (long n = 0; n < nsteps; ++n) {
        sim.step();
        if (n % 10 != 0) continue;
        const double t = n * rp.dt;
        CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), nb, cudaMemcpyDeviceToHost));
        // narrowband envelope at the drive frequency: demodulate the R-mode
        // pair (By + iBz) by e^{+i w0 t}, one-pole low-pass with tau = Tw.
        const double cw = std::cos(w0 * t), sw = std::sin(w0 * t);
        const double al = 10.0 * rp.dt / Tw;   // sample stride 10
        const double bY = by[xp], bZ = bz[xp];
        zpr += al * ((bY * cw - bZ * sw) - zpr);   // demod e^{+i w0 t}
        zpi += al * ((bY * sw + bZ * cw) - zpi);
        zmr += al * ((bY * cw + bZ * sw) - zmr);   // demod e^{-i w0 t}
        zmi += al * ((-bY * sw + bZ * cw) - zmi);
        const double bp = std::max(std::hypot(zpr, zpi), std::hypot(zmr, zmi));
        if (t < t_out_end)       a_inc = std::max(a_inc, bp);
        else if (t >= t_ref_beg) a_ref = std::max(a_ref, bp);
        if (df.is_open() && n % 100 == 0) {
            for (int i = 0; i < nx; ++i)
                bp_row[i] = (float)std::hypot((double)by[i], (double)bz[i]);
            df.write((const char*)bp_row.data(), nx * sizeof(float));
        }
    }
    const double R = a_inc > 0 ? a_ref / a_inc : 1.0;
    std::printf("mode=%d w0/wce=%.2f theta=%g nd=%d numax=%g taper=%d nedge=%g  "
                "A_inc=%.4e A_ref=%.4e  R=%.4f\n",
                mode, w0 / wce, theta, nd, numax, taper, nedge, a_inc, a_ref, R);
    return 0;
}
