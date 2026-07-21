// M4 — Tao GRL17 rising-tone chirping through the 2D Yee code path (ny = 1):
// hot PIC (delta-f, mirror-equilibrium load) + linearized cold fluid +
// parabolic B0(x) + Umeda layers + triggering antenna. The 2D-path analog of
// chirp1d's triggered run 8 (tag v1d-chirping-tao2017), nonrelativistic push.
//
// Usage: ./chirp2d <deck.ini> [outdir] [--ppc=N] [--amp=A] [--nsteps=N]
// Dumps into outdir:
//   bline_XXXXXX.bin   float32 By[nx] then Bz[nx], every bline_every steps
//   probe.bin          float32 (By,Bz) at nprobe x-locations, every probe_every
//   energy.csv         step,time,WE,WB,wd_sum,wd_rms,wd_max
//   meta.txt           geometry + cadence for the plot script

#include "pic/deck.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation_maxwell.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace arc;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <deck.ini> [outdir] [--ppc= --amp= --nsteps=]\n", argv[0]); return 1; }
    Deck d = load_deck(argv[1]);
    std::string outdir = (argc > 2 && argv[2][0] != '-') ? argv[2] : "chirp2d_out";
    for (int i = 2; i < argc; ++i) {
        if      (!std::strncmp(argv[i], "--ppc=", 6))    d.species[0].ppc = atoi(argv[i] + 6);
        else if (!std::strncmp(argv[i], "--amp=", 6))    d.rp.ant_amp = atof(argv[i] + 6);
        else if (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);
        else if (!std::strcmp(argv[i], "--fullf"))       d.species[0].deltaf = false;
    }
    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);

    if (g.ny != 1 || d.species.size() != 1 || !rp.b0_prof || rp.cold_nc <= 0.0) {
        std::fprintf(stderr, "chirp2d: needs ny=1, one hot species, [background] "
                             "profile=parabolic|dipole and [plasma] cold_nc > 0\n");
        return 1;
    }
    const Species& q = d.species[0];
    if (q.deltaf) {
        rp.deltaf = 1;
        rp.df_tpar  = q.uth[0] * q.uth[0];
        rp.df_tperp = q.uth[1] * q.uth[1];
        rp.df_taud  = q.taud;              // drift injection (0 = off)
        rp.df_dist  = q.dist;              // f0 shape must match the load
        rp.df_rho   = q.lc_rho;
        rp.df_kappa = q.lc_kappa;
    }
    if (rp.dt >= 0.999 * g.dx / rp.c) { std::fprintf(stderr, "chirp2d: CFL violated\n"); return 1; }

    std::filesystem::create_directories(outdir);
    write_run_meta(outdir, argv[1], argc, argv);

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(q, g, rp, sim.stream());
    if (rp.deltaf) sim.particles().enable_deltaf(sim.stream());
    sim.stream().synchronize();

    if (rp.deltaf && q.wdnoise > 0.0) {
        // persistent δf sampling noise: wd(0) random, rms = wdnoise
        std::vector<float> w0(sim.particles().n);
        std::srand((unsigned)rp.rng_seed + 7);
        const float A = (float)(q.wdnoise * std::sqrt(3.0));   // uniform ±A
        for (auto& w : w0) w = A * (2.f * std::rand() / (float)RAND_MAX - 1.f);
        CUDA_CHECK(cudaMemcpy(sim.particles().wd.data(), w0.data(),
                              w0.size() * 4, cudaMemcpyHostToDevice));
    }

    if (d.bnoise > 0.0) {
        // seed BAND-LIMITED random-phase noise on the transverse B (the
        // "initial noise level" of a delta-f run; d.bnoise = target rms of
        // |δB⊥| relative to wce). Whistler-band k ∈ [0.2, 1.5] wpe/c only —
        // grid-scale white noise just disperses/absorbs within ~2000/wce
        // without ever feeding the instability (measured: in-band fraction
        // of a white seed is ~1e-2 of the total). ny = 1 keeps div B = 0.
        std::vector<float> nb(g.real_size(), 0.f), nz(g.real_size(), 0.f);
        std::srand((unsigned)rp.rng_seed);
        const double Lxp = g.nx * g.dx;
        const int m1 = std::max(1, (int)std::ceil(0.2 * Lxp / (2.0 * M_PI)));
        const int m2 = std::min(g.nx / 2 - 1, (int)(1.5 * Lxp / (2.0 * M_PI)));
        const double a = d.bnoise * rp.wce / std::sqrt((double)(m2 - m1 + 1));
        for (int m = m1; m <= m2; ++m) {
            const double k  = 2.0 * M_PI * m / Lxp;
            const double p1 = 2.0 * M_PI * std::rand() / (double)RAND_MAX;
            const double p2 = 2.0 * M_PI * std::rand() / (double)RAND_MAX;
            for (int i = 0; i < g.nx; ++i) {
                nb[i] += (float)(a * std::cos(k * (i + 0.5) * g.dx + p1));
                nz[i] += (float)(a * std::cos(k * i * g.dx + p2));
            }
        }
        CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), nb.data(), nb.size() * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), nz.data(), nz.size() * 4,
                              cudaMemcpyHostToDevice));
    }

    const int bline_every = 100;                    // 10/wpe: w-k + x-t maps
    const int probe_every = 10;                     // 1/wpe : STFT time series
    // probe x-locations: [diagnostics] probes = offsets from the equator
    // (b0_xc) in physical units; default equator +/-100, +/-200 c/wpe
    std::vector<double> poff = d.probes;
    if (poff.empty()) poff = {-200.0, -100.0, 0.0, 100.0, 200.0};
    const int nprobe = (int)poff.size();
    std::vector<int> probe_ix(nprobe);
    for (int p = 0; p < nprobe; ++p)
        probe_ix[p] = (int)((rp.b0_xc + poff[p]) / g.dx);

    std::FILE* fpb = std::fopen((outdir + "/probe.bin").c_str(), "wb");
    std::FILE* fen = std::fopen((outdir + "/energy.csv").c_str(), "w");
    std::fprintf(fen, "step,time,WE,WB,wd_sum,wd_rms,wd_max\n");
    {   std::FILE* fm = std::fopen((outdir + "/meta.txt").c_str(), "w");
        std::fprintf(fm, "nx %d\ndx %.9g\ndt %.9g\nnsteps %ld\nbline_every %d\n"
                         "probe_every %d\nnprobe %d\nwce %.9g\nb0_a %.9g\nb0_xc %.9g\n"
                         "b0_prof %d\nb0_lre %.9g\n"
                         "ant_w0 %.9g\nant_amp %.9g\nant_toff %.9g\nnh %.9g\ncold_nc %.9g\n"
                         "ppc %d\ndeltaf %d\nnmarkers %zu\n",
                     g.nx, g.dx, rp.dt, rp.nsteps, bline_every, probe_every, nprobe,
                     rp.wce, rp.b0_a, rp.b0_xc, rp.b0_prof, rp.b0_lre,
                     rp.ant_w0, rp.ant_amp, rp.ant_toff,
                     q.density, rp.cold_nc, q.ppc, rp.deltaf, sim.particles().n);
        for (int p = 0; p < nprobe; ++p) std::fprintf(fm, "probe_ix %d\n", probe_ix[p]);
        std::fclose(fm); }

    std::vector<float> by(g.real_size()), bz(g.real_size());
    const long nsteps = rp.nsteps;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        const bool want_line  = n % bline_every == 0;
        const bool want_probe = n % probe_every == 0;
        if (want_line || want_probe) {
            CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), by.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
        }
        if (want_probe) {
            std::vector<float> pb(2 * nprobe);
            for (int p = 0; p < nprobe; ++p) { pb[2 * p] = by[probe_ix[p]]; pb[2 * p + 1] = bz[probe_ix[p]]; }
            std::fwrite(pb.data(), 4, 2 * nprobe, fpb);
        }
        if (want_line) {
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, g.nx, f);
            std::fwrite(bz.data(), 4, g.nx, f);
            std::fclose(f);
        }
        if (n % 2000 == 0) {
            const auto e = sim.field_energy();
            const auto w = sim.wd_stats();
            std::fprintf(fen, "%ld,%.6g,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                         n, n * rp.dt, e.we, e.wb, w.sum, w.rms, w.max);
            std::fflush(fen);
            if (n % 20000 == 0)
                std::printf("t=%8.0f  WB=%.3e  wd_rms=%.3e\n", n * rp.dt, e.wb, w.rms);
        }
    }
    std::fclose(fpb); std::fclose(fen);
    std::printf("done: %s\n", outdir.c_str());
    return 0;
}
