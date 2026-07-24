// G1.3/G2.2/G3 runner — 2D slab mirror ([background] profile=mirror2d) with
// the flagship diagnostics pack (pic/gap_diags.hpp). All species go through
// the multi-species (E,mu) mirror loader: anisotropic species get the mirror
// mapping, isotropic species degrade exactly to a uniform load, so
// cold-core + hot-minority decks compose in one call.
//
// Usage: ./mirror2d <deck.ini> [outdir] [--nsteps=N]
// Dumps into outdir (axis row j = ny/2 for the line diagnostics):
//   bline_XXXXXX.bin  float32 By[nx], Bz[nx], Ex[nx]     every bline_every
//                     (on-axis b-hat = x-hat exactly, so Epar = Ex)
//   probe.bin         float32 (By,Bz) at nprobe axis x's  every probe_every
//   fv_XXXXXX.bin     float64 [nreg][npar][nperp] f(vpar,vperp|region),
//                     w-weighted instantaneous, every fv_every
//   wl_XXXXXX.bin     float64 [nreg][nwb][2] resonance-tagged J.E ledger
//                     ([0]=Landau/E_par, [1]=cyclotron/E_perp), accumulated
//                     over the window then reset, every wl_every
//   f2d_XXXXXX.bin    float32 Ex,Ey,Ez,Bx,By,Bz [ny][nx] wave fields,
//                     every f2d_every (offline WNA map + E_par(x,y))
//   energy.csv, meta.txt

#include "pic/deck.hpp"
#include "pic/gap_diags.hpp"
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
    std::string outdir = (argc > 2 && argv[2][0] != '-') ? argv[2] : "mirror2d_out";
    for (int i = 2; i < argc; ++i)
        if (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);

    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    if (rp.b0_prof != 3 || d.species.empty()) {
        std::fprintf(stderr, "mirror2d: needs [background] profile=mirror2d and >=1 species\n");
        return 1;
    }
    if (rp.cold_nc > 0.0) {
        std::fprintf(stderr, "mirror2d: all-PIC only (cold_nc must be 0)\n");
        return 1;
    }
    if (rp.dt >= 0.999 * g.dx / (rp.c * std::sqrt(2.0))) {
        std::fprintf(stderr, "mirror2d: 2D CFL violated\n");
        return 1;
    }
    // design rules from test_mirror2d: every species must resolve its
    // gyroradius on dy (rule 1); warn if a species carries high beta (rule 2)
    for (const auto& q : d.species) {
        const double rho = q.uth[1] / (rp.B0[0] / std::fabs(rp.qm));
        if (rho < 2.0 * g.dy)
            std::fprintf(stderr, "mirror2d: WARNING species %s rho/dy = %.2f < 2 "
                                 "(magnetized grid heating; see test_mirror2d)\n",
                         q.name.c_str(), rho / g.dy);
        const double beta = 2.0 * q.density * q.uth[1] * q.uth[1]
                          / (rp.B0[0] * rp.B0[0]);
        if (q.uth[0] != q.uth[1] && beta > 0.05)
            std::fprintf(stderr, "mirror2d: WARNING species %s beta_perp = %.3f "
                                 "(vacuum-field (E,mu) load omits diamagnetic "
                                 "dB/B ~ beta/2)\n", q.name.c_str(), beta);
    }

    std::filesystem::create_directories(outdir);
    write_run_meta(outdir, argv[1], argc, argv);

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(d.species, g, rp, sim.stream());
    sim.stream().synchronize();

    // cadences in PHYSICAL time so dt scans keep the same output volume
    const int bline_every = std::max(1, (int)std::lround(4.0 / rp.dt));
    const int probe_every = std::max(1, (int)std::lround(0.2 / rp.dt));
    const int fv_every    = std::max(1, (int)std::lround(200.0 / rp.dt));
    const int wl_every    = std::max(1, (int)std::lround(200.0 / rp.dt));
    const int f2d_every   = std::max(1, (int)std::lround(400.0 / rp.dt));
    const int wl_acc      = 2;             // ledger sampling stride (steps)

    constexpr int   NREG = 8, NPAR = 160, NPERP = 80, NWB = 160;
    constexpr float VMAX = 0.6f;
    gapdiag::GapDiags gd(NREG, NPAR, NPERP, NWB, VMAX);

    const int jmid = g.ny / 2;             // axis row (y-hat center)
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
        std::fprintf(fm, "nx %d\nny %d\ndx %.9g\ndy %.9g\ndt %.9g\nnsteps %ld\n"
                         "bline_every %d\nprobe_every %d\nfv_every %d\n"
                         "wl_every %d\nf2d_every %d\nwl_acc %d\n"
                         "nreg %d\nnpar %d\nnperp %d\nnwb %d\nvmax %.9g\n"
                         "nprobe %d\nnsp %d\nwce %.9g\nb0_a %.9g\nb0_xc %.9g\n"
                         "b0_yc %.9g\nbnd_x %d\nnmarkers %zu\n",
                     g.nx, g.ny, g.dx, g.dy, rp.dt, rp.nsteps,
                     bline_every, probe_every, fv_every, wl_every, f2d_every,
                     wl_acc, NREG, NPAR, NPERP, NWB, (double)VMAX,
                     nprobe, (int)d.species.size(), rp.B0[0], rp.b0_a,
                     rp.b0_xc, rp.b0_yc, rp.bnd_x, sim.particles().n);
        for (const auto& q : d.species)
            std::fprintf(fm, "species %s density %.9g ppc %d uth %.9g %.9g %.9g\n",
                         q.name.c_str(), q.density, q.ppc,
                         q.uth[0], q.uth[1], q.uth[2]);
        for (int p = 0; p < nprobe; ++p) std::fprintf(fm, "probe_ix %d\n", probe_ix[p]);
        std::fclose(fm); }

    const std::size_t rowoff = (std::size_t)jmid * g.nx;
    std::vector<float> by(g.nx), bz(g.nx), ex(g.nx);
    std::vector<float> f2(g.real_size());
    char fn[512];
    const long nsteps = rp.nsteps;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step();
        if (n % wl_acc == 0)
            gd.wl_accum(sim.particles(), sim.fields(), rp,
                        (float)(rp.dt * wl_acc), sim.stream());
        const bool want_line  = n % bline_every == 0;
        const bool want_probe = n % probe_every == 0;
        if (want_line || want_probe) {
            CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
        }
        if (want_probe) {
            std::vector<float> pb(2 * nprobe);
            for (int p = 0; p < nprobe; ++p) { pb[2*p] = by[probe_ix[p]]; pb[2*p+1] = bz[probe_ix[p]]; }
            std::fwrite(pb.data(), 4, 2 * nprobe, fpb);
        }
        if (want_line) {
            CUDA_CHECK(cudaMemcpy(ex.data(), sim.fields().ex_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, g.nx, f);
            std::fwrite(bz.data(), 4, g.nx, f);
            std::fwrite(ex.data(), 4, g.nx, f);
            std::fclose(f);
        }
        if (n % fv_every == 0) {
            std::snprintf(fn, sizeof fn, "%s/fv_%06ld.bin", outdir.c_str(), n / fv_every);
            gd.fv_snapshot(sim.particles(), rp, g, sim.stream(), fn);
        }
        if (n % wl_every == 0) {
            std::snprintf(fn, sizeof fn, "%s/wl_%06ld.bin", outdir.c_str(), n / wl_every);
            gd.wl_write_reset(sim.stream(), fn);
        }
        if (n % f2d_every == 0) {
            std::snprintf(fn, sizeof fn, "%s/f2d_%06ld.bin", outdir.c_str(), n / f2d_every);
            std::FILE* f = std::fopen(fn, "wb");
            DeviceArray<float>* comps[6] = {
                &sim.fields().ex_, &sim.fields().ey_, &sim.fields().ez_,
                &sim.fields().bx_, &sim.fields().by_, &sim.fields().bz_ };
            for (auto* c : comps) {
                CUDA_CHECK(cudaMemcpy(f2.data(), c->data(), f2.size() * 4, cudaMemcpyDeviceToHost));
                std::fwrite(f2.data(), 4, f2.size(), f);
            }
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
