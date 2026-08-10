// G1.3/G2.2/G3 + PLAN_2D P0-2 runner — 2D slab mirror ([background]
// profile=mirror2d) with the flagship diagnostics pack (pic/gap_diags.hpp)
// upgraded to production: dual bline (y-mid row + y-average), per-probe
// y-columns for the continuous k_y spectrum, probe_e, checkpoint/resume.
// All species go through the multi-species (E,mu) mirror loader; the
// PLAN_2D chi_r shell split ([species] shell = v1 v2 dv) reports its
// measured shell fraction at load (stage-1 gate input).
//
// Usage: ./mirror2d <deck.ini> [outdir] [--nsteps=N] [--ckpt=N] [--ckptseq] [--resume]
// --ckpt=N   save <outdir>/ckpt.bin every N steps (atomic .tmp+rename)
// --ckptseq  write unique ckpt_<step>.bin instead of overwriting
// --resume   continue from <outdir>/ckpt.bin (same deck; diagnostics append)
//
// Dumps into outdir (probe convention = chirp2d: deck probes are OFFSETS
// from b0_xc in physical units, negative = south):
//   bline_XXXXXX.bin  float32 [5][nx]: By,Bz,Ex on the y-mid row, then
//                     y-AVERAGED By,Bz                       every bline_every
//   ycol_XXXXXX.bin   float32 [nprobe][2][ny]: By(y),Bz(y) columns at the
//                     probe x-stations (offline FFT -> B(x_st,ky,t)), same
//                     cadence as bline
//   probe.bin         float32 (By,Bz) y-mid at probe x's    every probe_every
//   probe_e.bin       float32 (Ey,Ez) y-mid at probe x's    every probe_every
//   fv_XXXXXX.bin     float64 [nreg][npar][nperp] f(vpar,vperp|region)
//   wl_XXXXXX.bin     float64 [nreg][nwb][2] resonance-tagged J.E ledger
//   f2d_XXXXXX.bin    float32 Ex,Ey,Ez,Bx,By,Bz [ny][nx] full wave fields
//   energy.csv, meta.txt

#include "pic/checkpoint_io.hpp"
#include "pic/deck.hpp"
#include "pic/gap_diags.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation_maxwell.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace arc;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <deck.ini> [outdir] [--nsteps=N] "
                             "[--ckpt=N] [--ckptseq] [--resume]\n", argv[0]);
        return 1;
    }
    Deck d = load_deck(argv[1]);
    std::string outdir = (argc > 2 && argv[2][0] != '-') ? argv[2] : "mirror2d_out";
    long ckpt_every = 0;
    bool ckpt_seq = false, resume = false;
    for (int i = 2; i < argc; ++i) {
        if      (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);
        else if (!std::strncmp(argv[i], "--ckpt=", 7))   ckpt_every = atol(argv[i] + 7);
        else if (!std::strcmp(argv[i], "--ckptseq"))     ckpt_seq = true;
        else if (!std::strcmp(argv[i], "--resume"))      resume = true;
    }

    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    if (rp.b0_prof != 3 || d.species.empty()) {
        std::fprintf(stderr, "mirror2d: needs [background] profile=mirror2d and >=1 species\n");
        return 1;
    }
    if (rp.cold_nc > 0.0 && !rp.cold_full) {
        std::fprintf(stderr, "mirror2d: cold_nc needs [field] cold_model = full "
                             "(3-component staggered-correct fluid)\n");
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
    std::string deck_text;
    {   std::ifstream df(argv[1]);
        std::stringstream ss; ss << df.rdbuf(); deck_text = ss.str(); }

    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(d.species, g, rp, sim.stream());
    sim.stream().synchronize();

    // chi_r shell report: measured kinetic fraction per species (sum w /
    // (density*dx*dy/ppc * nloaded)) — the deck's cold_nc must absorb the
    // complement density*(1-fshell) (PLAN_2D stage-1 gate).
    {
        auto& P = sim.particles();
        std::vector<float> w(P.n);
        CUDA_CHECK(cudaMemcpy(w.data(), P.w.data(), P.n * 4, cudaMemcpyDeviceToHost));
        // species blocks are contiguous in load order; recover extents by
        // walking the same per-cell counting the loader used is overkill —
        // report the GLOBAL weighted density instead (sum w / (Lx*Ly)).
        double wsum = 0.0;
        for (float v : w) wsum += v;
        std::printf("loaded %zu markers, deposited density integral = %.6g "
                    "(deck sum n*V = %.6g)\n", P.n, wsum,
                    [&]{ double s = 0; for (auto& q : d.species) s += q.density; return s * d.Lx * d.Ly; }());
    }

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
    // probes: OFFSETS from b0_xc (chirp2d convention; analysis tools share it)
    std::vector<double> poff = d.probes;
    if (poff.empty()) poff = {-0.25 * d.Lx, 0.0, 0.25 * d.Lx};
    const int nprobe = (int)poff.size();
    std::vector<int> probe_ix(nprobe);
    for (int p = 0; p < nprobe; ++p) {
        int ix = (int)((rp.b0_xc + poff[p]) / g.dx);
        if (ix < 0) ix = 0;
        if (ix >= g.nx) ix = g.nx - 1;
        probe_ix[p] = ix;
    }

    const std::string ckpt_path = outdir + "/ckpt.bin";
    long n0 = 0;
    if (resume) {
        std::string saved_deck;
        n0 = checkpoint_load(ckpt_path, sim, &saved_deck);
        if (saved_deck != deck_text)
            std::fprintf(stderr, "mirror2d: WARNING deck text differs from "
                                 "checkpoint (continuing anyway)\n");
        std::printf("resumed %s at step %ld (t=%.0f)\n", ckpt_path.c_str(),
                    n0, n0 * rp.dt);
        std::error_code ec;
        std::filesystem::resize_file(outdir + "/probe.bin",
            (uintmax_t)(n0 / probe_every) * 2 * nprobe * 4, ec);
        std::filesystem::resize_file(outdir + "/probe_e.bin",
            (uintmax_t)(n0 / probe_every) * 2 * nprobe * 4, ec);
    }

    std::FILE* fpb = std::fopen((outdir + "/probe.bin").c_str(), resume ? "ab" : "wb");
    std::FILE* fpe = std::fopen((outdir + "/probe_e.bin").c_str(), resume ? "ab" : "wb");
    std::FILE* fen = std::fopen((outdir + "/energy.csv").c_str(), resume ? "a" : "w");
    if (!resume) std::fprintf(fen, "step,time,WE,WB\n");
    {   std::FILE* fm = std::fopen((outdir + "/meta.txt").c_str(), "w");
        std::fprintf(fm, "nx %d\nny %d\ndx %.9g\ndy %.9g\ndt %.9g\nnsteps %ld\n"
                         "bline_every %d\nbline_ncomp 5\nycol 1\nprobe_every %d\n"
                         "fv_every %d\nwl_every %d\nf2d_every %d\nwl_acc %d\n"
                         "nreg %d\nnpar %d\nnperp %d\nnwb %d\nvmax %.9g\n"
                         "nprobe %d\nnsp %d\nwce %.9g\nb0_a %.9g\nb0_xc %.9g\n"
                         "cold_nc %.9g\nb0_yc %.9g\nbnd_x %d\nnmarkers %zu\n",
                     g.nx, g.ny, g.dx, g.dy, rp.dt, rp.nsteps,
                     bline_every, probe_every, fv_every, wl_every, f2d_every,
                     wl_acc, NREG, NPAR, NPERP, NWB, (double)VMAX,
                     nprobe, (int)d.species.size(), rp.B0[0], rp.b0_a,
                     rp.b0_xc, rp.cold_nc, rp.b0_yc, rp.bnd_x, sim.particles().n);
        for (const auto& q : d.species) {
            std::fprintf(fm, "species %s density %.9g ppc %d uth %.9g %.9g %.9g "
                             "dist %d shell %.9g %.9g %.9g %d\n",
                         q.name.c_str(), q.density, q.ppc,
                         q.uth[0], q.uth[1], q.uth[2], q.dist,
                         q.shell_v1, q.shell_v2, q.shell_dv, q.shell_invert);
        }
        for (int p = 0; p < nprobe; ++p)
            std::fprintf(fm, "probe_ix %d probe_off %.9g\n", probe_ix[p], poff[p]);
        std::fclose(fm); }

    const std::size_t rowoff = (std::size_t)jmid * g.nx;
    std::vector<float> by(g.nx), bz(g.nx), ex(g.nx), ey(g.nx), ez(g.nx);
    std::vector<float> fby(g.real_size()), fbz(g.real_size());
    std::vector<float> yavg(2 * g.nx), ycol(2 * g.ny);
    std::vector<float> f2(g.real_size());
    char fn[512];
    const long nsteps = rp.nsteps;
    const auto t_wall0 = std::chrono::steady_clock::now();
    long n_wall0 = n0;
    for (long n = n0 + 1; n <= nsteps; ++n) {
        sim.step();
        if (n % wl_acc == 0)
            gd.wl_accum(sim.particles(), sim.fields(), rp,
                        (float)(rp.dt * wl_acc), sim.stream());
        if (ckpt_every > 0 && n % ckpt_every == 0) {
            sim.stream().synchronize();
            const std::string cp = ckpt_seq
                ? outdir + "/ckpt_" + std::to_string(n) + ".bin" : ckpt_path;
            checkpoint_save(cp, sim, n, n * rp.dt,
                            (uint64_t)rp.rng_seed, deck_text);
            std::printf("ckpt @ step %ld -> %s\n", n, cp.c_str());
        }
        const bool want_line  = n % bline_every == 0;
        const bool want_probe = n % probe_every == 0;
        if (want_line || want_probe) {
            CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
        }
        if (want_probe) {
            CUDA_CHECK(cudaMemcpy(ey.data(), sim.fields().ey_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ez.data(), sim.fields().ez_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
            std::vector<float> pb(2 * nprobe), pe(2 * nprobe);
            for (int p = 0; p < nprobe; ++p) {
                pb[2*p] = by[probe_ix[p]]; pb[2*p+1] = bz[probe_ix[p]];
                pe[2*p] = ey[probe_ix[p]]; pe[2*p+1] = ez[probe_ix[p]];
            }
            std::fwrite(pb.data(), 4, 2 * nprobe, fpb);
            std::fwrite(pe.data(), 4, 2 * nprobe, fpe);
        }
        if (want_line) {
            CUDA_CHECK(cudaMemcpy(ex.data(), sim.fields().ex_.data() + rowoff, g.nx * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(fby.data(), sim.fields().by_.data(), fby.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(fbz.data(), sim.fields().bz_.data(), fbz.size() * 4, cudaMemcpyDeviceToHost));
            // y-averaged By,Bz (finite-WNA power survives; y-mid row alone
            // aliases oblique structure)
            std::fill(yavg.begin(), yavg.end(), 0.f);
            for (int j = 0; j < g.ny; ++j) {
                const std::size_t ro = (std::size_t)j * g.nx;
                for (int i = 0; i < g.nx; ++i) {
                    yavg[i]        += fby[ro + i];
                    yavg[g.nx + i] += fbz[ro + i];
                }
            }
            const float inv = 1.f / (float)g.ny;
            for (auto& v : yavg) v *= inv;
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, g.nx, f);
            std::fwrite(bz.data(), 4, g.nx, f);
            std::fwrite(ex.data(), 4, g.nx, f);
            std::fwrite(yavg.data(), 4, 2 * g.nx, f);
            std::fclose(f);
            // per-probe y-columns (continuous k_y spectrum offline)
            std::snprintf(fn, sizeof fn, "%s/ycol_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* fc = std::fopen(fn, "wb");
            for (int p = 0; p < nprobe; ++p) {
                const int ix = probe_ix[p];
                for (int j = 0; j < g.ny; ++j) {
                    ycol[j]        = fby[(std::size_t)j * g.nx + ix];
                    ycol[g.ny + j] = fbz[(std::size_t)j * g.nx + ix];
                }
                std::fwrite(ycol.data(), 4, 2 * g.ny, fc);
            }
            std::fclose(fc);
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
            if (n % 20000 == 0) {
                const double secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_wall0).count();
                const double rate = (double)sim.particles().n * (n - n_wall0)
                                  / std::max(secs, 1e-9);
                std::printf("t=%8.0f  WE=%.3e  WB=%.3e  %.2e p-steps/s\n",
                            n * rp.dt, e.we, e.wb, rate);
                std::fflush(stdout);
            }
        }
    }
    std::fclose(fpb); std::fclose(fpe); std::fclose(fen);
    std::printf("done: %s\n", outdir.c_str());
    return 0;
}
