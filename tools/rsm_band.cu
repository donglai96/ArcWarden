// RSM V4 — Li-type uniform-B kinetic benchmark through the RSM m = ±1 path
// (docs/RSM_MODEL_DEFINITION.md ladder V4). The liband_yee twin with the
// geometry inverted: B0 strictly along x (theta_deg = 0), the obliquity
// carried by the SPECTRAL k1 harmonic instead of tilting the box. That
// cleanly splits the two channels the RSM exists to separate:
//   m = 0: strictly parallel whistlers — cyclotron channel ONLY (E_par = 0)
//   m = 1: oblique response at fixed k_perp = k1 — the Landau channel
// No m1 seed: particle shot noise through the modal J1 deposit is the
// physical noise floor (gate V1 verified its statistics).
//
// Usage: ./rsm_band <deck.ini> [outdir] [--nsteps=N]
// Dumps into outdir:
//   bline_XXXXXX.bin   float32 By[nx], Bz[nx], Ex[nx]      (m0; Ex = E_par)
//   m1line_XXXXXX.bin  float32 (re,im)x nx for B1y, B1z, E1x, E1y
//   fhist_XXXXXX.bin   float64 w-weighted f(vpar), NB bins over [-VMAX,VMAX]
//   probe.bin, energy.csv (incl. 2*W1), meta.txt
#include "pic/deck.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace arc;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <deck.ini> [outdir] [--nsteps=N]\n", argv[0]); return 1; }
    Deck d = load_deck(argv[1]);
    std::string outdir = (argc > 2 && argv[2][0] != '-') ? argv[2] : "rsm_band_out";
    for (int i = 2; i < argc; ++i)
        if (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);

    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    if (g.ny != 1 || rp.b0_prof || d.species.empty() || !rp.rsm) {
        std::fprintf(stderr, "rsm_band: needs ny=1, uniform B0, >=1 species, "
                             "[rsm] enable = true\n");
        return 1;
    }
    if (rp.B0[1] != 0.f || rp.B0[2] != 0.f) {
        std::fprintf(stderr, "rsm_band: B0 must be along x (theta_deg = 0) — "
                             "obliquity comes from the m=1 harmonic, not a tilt\n");
        return 1;
    }
    if (rp.dt >= 0.999 * g.dx / rp.c) { std::fprintf(stderr, "rsm_band: CFL violated\n"); return 1; }

    std::filesystem::create_directories(outdir);
    write_run_meta(outdir, argv[1], argc, argv);

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(d.species, g, rp, sim.stream());
    rsm_theta_init(sim.particles(), rp, sim.stream());   // R1: phase load
    sim.stream().synchronize();

    const int bline_every = 200;
    const int probe_every = 10;
    const int fhist_every = 2000;
    constexpr int NB = 240;
    constexpr double VMAX = 0.6;

    std::vector<double> poff = d.probes;
    if (poff.empty()) poff = {0.25 * d.Lx, 0.5 * d.Lx, 0.75 * d.Lx};
    const int nprobe = (int)poff.size();
    std::vector<int> probe_ix(nprobe);
    for (int p = 0; p < nprobe; ++p)
        probe_ix[p] = ((int)(poff[p] / g.dx) % g.nx + g.nx) % g.nx;

    std::FILE* fpb = std::fopen((outdir + "/probe.bin").c_str(), "wb");
    std::FILE* fen = std::fopen((outdir + "/energy.csv").c_str(), "w");
    std::fprintf(fen, "step,time,WE,WB,W1\n");
    {   std::FILE* fm = std::fopen((outdir + "/meta.txt").c_str(), "w");
        std::fprintf(fm, "nx %d\ndx %.9g\ndt %.9g\nnsteps %ld\nbline_every %d\n"
                         "probe_every %d\nfhist_every %d\nnb %d\nvmax %.9g\n"
                         "nprobe %d\nnsp %d\nwce %.9g\nrsm_k1 %.9g\n"
                         "nmarkers %zu\n",
                     g.nx, g.dx, rp.dt, rp.nsteps, bline_every, probe_every,
                     fhist_every, NB, VMAX, nprobe, (int)d.species.size(),
                     (double)rp.B0[0], rp.rsm_k1, sim.particles().n);
        for (const auto& q : d.species)
            std::fprintf(fm, "species %s density %.9g ppc %d kappa_v %.3g\n",
                         q.name.c_str(), q.density, q.ppc, q.kappa_v);
        for (int p = 0; p < nprobe; ++p) std::fprintf(fm, "probe_ix %d\n", probe_ix[p]);
        std::fclose(fm); }

    std::vector<float>  by(g.real_size()), bz(g.real_size()), ex(g.real_size());
    std::vector<float2> m1buf(g.nx);
    std::vector<float>  hux, hw;
    const double dV = g.dx * g.dy;
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
            for (int p = 0; p < nprobe; ++p) { pb[2*p] = by[probe_ix[p]]; pb[2*p+1] = bz[probe_ix[p]]; }
            std::fwrite(pb.data(), 4, 2 * nprobe, fpb);
        }
        if (want_line) {
            CUDA_CHECK(cudaMemcpy(ex.data(), sim.fields().ex_.data(), ex.size() * 4, cudaMemcpyDeviceToHost));
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, g.nx, f);
            std::fwrite(bz.data(), 4, g.nx, f);
            std::fwrite(ex.data(), 4, g.nx, f);   // m0 E_par (B0 || x)
            std::fclose(f);
            // m1 complex lines: B1y, B1z (wave), E1x (the Landau E_par), E1y
            std::snprintf(fn, sizeof fn, "%s/m1line_%06ld.bin", outdir.c_str(), n / bline_every);
            f = std::fopen(fn, "wb");
            RsmState& r = sim.rsm();
            for (auto* arr : { &r.b1y, &r.b1z, &r.e1x, &r.e1y }) {
                CUDA_CHECK(cudaMemcpy(m1buf.data(), arr->data(), g.nx * sizeof(float2),
                                      cudaMemcpyDeviceToHost));
                std::fwrite(m1buf.data(), sizeof(float2), g.nx, f);
            }
            std::fclose(f);
        }
        if (n % fhist_every == 0) {
            // w-weighted total f(vpar); B0 || x so vpar = ux (nonrel decks)
            const size_t N = sim.particles().n;
            hux.resize(N); hw.resize(N);
            CUDA_CHECK(cudaMemcpy(hux.data(), sim.particles().ux.data(), N * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hw.data(),  sim.particles().w.data(),  N * 4, cudaMemcpyDeviceToHost));
            std::vector<double> hist(NB, 0.0);
            for (size_t t = 0; t < N; ++t) {
                const int b = (int)(((double)hux[t] + VMAX) / (2 * VMAX) * NB);
                if (b >= 0 && b < NB) hist[b] += hw[t];
            }
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/fhist_%06ld.bin", outdir.c_str(), n / fhist_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(hist.data(), 8, NB, f);
            std::fclose(f);
        }
        if (n % 2000 == 0) {
            const auto e = sim.field_energy();
            // m1 field energy 2*W1 (R6 ledger: the +-k1 pair)
            double w1 = 0;
            RsmState& r = sim.rsm();
            const double c2 = rp.c * rp.c;
            for (auto* arr : { &r.e1x, &r.e1y, &r.e1z }) {
                CUDA_CHECK(cudaMemcpy(m1buf.data(), arr->data(), g.nx * sizeof(float2), cudaMemcpyDeviceToHost));
                for (auto& z : m1buf) w1 += 0.5 * ((double)z.x * z.x + (double)z.y * z.y);
            }
            for (auto* arr : { &r.b1x, &r.b1y, &r.b1z }) {
                CUDA_CHECK(cudaMemcpy(m1buf.data(), arr->data(), g.nx * sizeof(float2), cudaMemcpyDeviceToHost));
                for (auto& z : m1buf) w1 += 0.5 * c2 * ((double)z.x * z.x + (double)z.y * z.y);
            }
            w1 *= 2.0 * dV;
            std::fprintf(fen, "%ld,%.6g,%.9e,%.9e,%.9e\n", n, n * rp.dt, e.we, e.wb, w1);
            std::fflush(fen);
            if (n % 20000 == 0)
                std::printf("t=%8.0f  WE=%.3e  WB=%.3e  2W1=%.3e\n",
                            n * rp.dt, e.we, e.wb, w1);
        }
    }
    std::fclose(fpb); std::fclose(fen);
    std::printf("done: %s\n", outdir.c_str());
    return 0;
}
