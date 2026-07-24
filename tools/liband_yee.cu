// G2.1 — Li/Bortnik/An 2019 (Nat. Commun. 10, 4672) two-band replication:
// full-PIC (no cold fluid), UNIFORM B0 tilted by [background] theta_deg in
// the x-z plane, periodic x, multi-species with bi-kappa velocities
// ([species] kappa_v). Purpose: validate the Landau/plateau channel + kappa
// loader before any 2D-mirror cost, and host the one-way-street pretest.
//
// Usage: ./liband_yee <deck.ini> [outdir] [--nsteps=N]
// Dumps into outdir:
//   bline_XXXXXX.bin  float32 By[nx], Bz[nx], Epar[nx]   every bline_every
//   probe.bin         float32 (By,Bz) at nprobe x's      every probe_every
//   fhist_XXXXXX.bin  float64 w-weighted total f(vpar), NB bins over
//                     [-VMAX, VMAX], every fhist_every (plateau monitor)
//   energy.csv, meta.txt
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
    std::string outdir = (argc > 2 && argv[2][0] != '-') ? argv[2] : "liband_out";
    for (int i = 2; i < argc; ++i)
        if (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);

    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    if (g.ny != 1 || rp.b0_prof || rp.cold_nc > 0.0 || d.species.empty()) {
        std::fprintf(stderr, "liband_yee: needs ny=1, uniform B0 (no profile), "
                             "cold_nc=0 (all-PIC), >=1 species\n");
        return 1;
    }
    if (rp.dt >= 0.999 * g.dx / rp.c) { std::fprintf(stderr, "liband_yee: CFL violated\n"); return 1; }

    // tilt: B0 = wce (cos th, 0, sin th); parallel projections use (cx, cz)
    const double th = std::atan2((double)rp.B0[2], (double)rp.B0[0]);
    const float  cx = (float)std::cos(th), cz = (float)std::sin(th);

    std::filesystem::create_directories(outdir);
    write_run_meta(outdir, argv[1], argc, argv);

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(d.species, g, rp, sim.stream());
    sim.stream().synchronize();

    const int nsp = (int)d.species.size();
    std::vector<long> sp_base(nsp + 1, 0);
    for (int s = 0; s < nsp; ++s)
        sp_base[s + 1] = sp_base[s] + (long)d.species[s].ppc * g.real_size();

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
    std::fprintf(fen, "step,time,WE,WB\n");
    {   std::FILE* fm = std::fopen((outdir + "/meta.txt").c_str(), "w");
        std::fprintf(fm, "nx %d\ndx %.9g\ndt %.9g\nnsteps %ld\nbline_every %d\n"
                         "probe_every %d\nfhist_every %d\nnb %d\nvmax %.9g\n"
                         "nprobe %d\nnsp %d\nwce %.9g\ntheta_deg %.9g\n"
                         "nmarkers %zu\n",
                     g.nx, g.dx, rp.dt, rp.nsteps, bline_every, probe_every,
                     fhist_every, NB, VMAX, nprobe, nsp,
                     std::hypot(rp.B0[0], rp.B0[2]), th * 180.0 / M_PI,
                     sim.particles().n);
        for (int s = 0; s < nsp; ++s)
            std::fprintf(fm, "species %s density %.9g ppc %d kappa_v %.3g\n",
                         d.species[s].name.c_str(), d.species[s].density,
                         d.species[s].ppc, d.species[s].kappa_v);
        for (int p = 0; p < nprobe; ++p) std::fprintf(fm, "probe_ix %d\n", probe_ix[p]);
        std::fclose(fm); }

    std::vector<float> by(g.real_size()), bz(g.real_size()),
                       ex(g.real_size()), ez(g.real_size());
    std::vector<float> hux, huz;   // fhist staging
    const long nsteps = rp.nsteps;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        const bool want_line  = n % bline_every == 0;
        const bool want_probe = n % probe_every == 0;
        const bool want_hist  = n % fhist_every == 0;
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
            CUDA_CHECK(cudaMemcpy(ez.data(), sim.fields().ez_.data(), ez.size() * 4, cudaMemcpyDeviceToHost));
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, g.nx, f);
            std::fwrite(bz.data(), 4, g.nx, f);
            std::vector<float> ep(g.nx);
            for (int i = 0; i < g.nx; ++i) ep[i] = cx * ex[i] + cz * ez[i];
            std::fwrite(ep.data(), 4, g.nx, f);
            std::fclose(f);
        }
        if (want_hist) {
            // per-species w-weighted f(vpar): PIC current is per-marker, so
            // vpar = (ux cx + uz cz) nonrelativistically (rel off in decks).
            const size_t N = sim.particles().n;
            hux.resize(N); huz.resize(N);
            std::vector<float> hw(N);
            CUDA_CHECK(cudaMemcpy(hux.data(), sim.particles().ux.data(), N * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(huz.data(), sim.particles().uz.data(), N * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hw.data(),  sim.particles().w.data(),  N * 4, cudaMemcpyDeviceToHost));
            // Tile sorting reorders particles globally (species slices are
            // load-time only), so histogram all markers together, weighted
            // by w — the physical total f(vpar), which is what the plateau
            // monitor wants.
            std::vector<double> hist(NB, 0.0);
            for (size_t t = 0; t < N; ++t) {
                const double vp = cx * (double)hux[t] + cz * (double)huz[t];
                const int b = (int)((vp + VMAX) / (2 * VMAX) * NB);
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
            std::fprintf(fen, "%ld,%.6g,%.9e,%.9e\n", n, n * rp.dt, e.we, e.wb);
            std::fflush(fen);
            if (n % 20000 == 0)
                std::printf("t=%8.0f  WE=%.3e  WB=%.3e\n", n * rp.dt, e.we, e.wb);
        }
    }
    std::fclose(fpb); std::fclose(fen);
    std::printf("done: %s\n", outdir.c_str());
    return 0;
}
