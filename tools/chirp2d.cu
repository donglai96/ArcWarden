// M4 — Tao GRL17 rising-tone chirping through the 2D Yee code path (ny = 1):
// hot PIC (delta-f, mirror-equilibrium load) + linearized cold fluid +
// parabolic B0(x) + Umeda layers + triggering antenna. The 2D-path analog of
// chirp1d's triggered run 8 (tag v1d-chirping-tao2017), nonrelativistic push.
//
// Usage: ./chirp2d <deck.ini> [outdir] [--ppc=N] [--amp=A] [--nsteps=N]
//                  [--fullf] [--ckpt=N] [--resume]
// --ckpt=N   save <outdir>/ckpt.bin every N steps (atomic .tmp+rename)
// --resume   continue from <outdir>/ckpt.bin (same deck; diagnostics append)
// Dumps into outdir:
//   bline_XXXXXX.bin   float32 By[nx] then Bz[nx], every bline_every steps
//   eline_XXXXXX.bin   float32 Ex[nx],Ey[nx],Ez[nx] — with bline gives Poynting
//                      S_x = EyBz−EzBy (x-t source/direction map)
//   jline_XXXXXX.bin   float32 Jx[nx],Jy[nx],Jz[nx] — HOT deposit only (cold
//                      fluid enters Ampère separately) ⇒ J·E ledger localizes
//                      generation: J⊥·E⊥ = cyclotron channel, JxEx = Landau
//   probe.bin          float32 (By,Bz) at nprobe x-locations, every probe_every
//   probe_e.bin        float32 (Ey,Ez) same probes/cadence (probe Poynting)
//   energy.csv         step,time,WE,WB,wd_sum,wd_rms,wd_max
//   meta.txt           geometry + cadence for the plot script
// --fvdiag (A0 mechanism pack, PLAN_TWO_TRACK v2.1 §A0.2/.4; read-only):
//   fvline_XXXXXX.bin  float64 f(v_par)[fv_nv], w-weighted, equatorial window
//                      |i - i_eq| < fv_hw cells, every fv_every steps
//   fv2d_XXXXXX.bin    float64 [fv_npar][fv_nperp] coarse (v_par,v_perp) there
//   wl_XXXXXX.bin      float64 [wl_nreg][wl_nwb][2] m=0 J.E work since last
//                      dump, (x-region, v_par)-binned, [..0]=Landau [..1]=
//                      cyclotron (Yee E is pure m=0; rsm m=1 is spectral)

#include "pic/checkpoint_io.hpp"
#include "pic/deck.hpp"
#include "pic/gap_diags.hpp"
#include "pic/refresh.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation_maxwell.hpp"

#include <fstream>
#include <sstream>

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
    long ckpt_every = 0;          // --ckpt=N: save outdir/ckpt.bin every N steps
    bool resume = false;          // --resume: continue from outdir/ckpt.bin
    bool ckpt_seq = false;        // --ckptseq: write unique ckpt_<step>.bin (no overwrite,
                                  //            no watcher needed) for dense phase-space series
    bool no_eline = false;        // --noeline: skip eline_*.bin (the full-nx E line = the
                                  //            biggest "probe"; probe_e keeps probe Poynting)
    bool fvdiag = false;          // --fvdiag: A0 mechanism pack (fvline/fv2d/wl);
                                  //           mandatory on >=1 mechanism run per arm
    for (int i = 2; i < argc; ++i) {
        if      (!std::strncmp(argv[i], "--ppc=", 6))    d.species[0].ppc = atoi(argv[i] + 6);
        else if (!std::strncmp(argv[i], "--amp=", 6))    d.rp.ant_amp = atof(argv[i] + 6);
        else if (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);
        else if (!std::strcmp(argv[i], "--fullf"))       d.species[0].deltaf = false;
        else if (!std::strncmp(argv[i], "--ckpt=", 7))   ckpt_every = atol(argv[i] + 7);
        else if (!std::strcmp(argv[i], "--ckptseq"))     ckpt_seq = true;
        else if (!std::strcmp(argv[i], "--resume"))      resume = true;
        else if (!std::strcmp(argv[i], "--noeline"))     no_eline = true;
        else if (!std::strcmp(argv[i], "--fvdiag"))      fvdiag = true;
    }
    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);

    if (g.ny != 1 || d.species.empty() || d.species.size() > 2 ||
        !rp.b0_prof || rp.cold_nc <= 0.0) {
        std::fprintf(stderr, "chirp2d: needs ny=1, 1-2 hot species, [background] "
                             "profile=parabolic|dipole and [plasma] cold_nc > 0\n");
        return 1;
    }
    const Species& q = d.species[0];
    if (d.species.size() == 2 && (q.deltaf || d.species[1].deltaf)) {
        // exp G (08-01): second species = low-energy anisotropic injection
        // pancake on top of the B engine — fullf only (no multi-species δf)
        std::fprintf(stderr, "chirp2d: two species requires rep = fullf\n");
        return 1;
    }
    if (q.deltaf && q.dist == 3) {
        std::fprintf(stderr, "chirp2d: prodkappa has no delta-f dlnf0 — fullf only\n");
        return 1;
    }
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
    sim.particles().initialize_mirror(d.species, g, rp, sim.stream());
    if (rp.deltaf) sim.particles().enable_deltaf(sim.stream());
    // RSM: randomize the particle phase θ = 2π·y (R1 — the ny=1 loader pins
    // y = 0.5, a coherent fake oblique seed otherwise). Fresh starts only:
    // on --resume the θ stream is restored from the checkpoint ("py").
    if (rp.rsm && !resume) rsm_theta_init(sim.particles(), rp, sim.stream());
    // RSM m = 1 field seed ([rsm] seed, amplitude/wce): mandatory for δf runs
    // (weights start at 0 → no wd-weighted shot noise to ignite m = 1),
    // optional for full-f (which self-seeds from load noise). Fresh starts
    // only — --resume restores the m = 1 lines from the checkpoint.
    if (rp.rsm && !resume) rsm_seed_init(sim.rsm(), rp, sim.stream());
    // boundary-refresh bath (REFRESH_DESIGN.md): stateless, so fresh start
    // and --resume initialize identically (per-step-seeded RNG replays).
    RefreshState rfr;
    rfr.init(q, g, rp, sim.stream());
    sim.stream().synchronize();

    const std::string ckpt_path = outdir + "/ckpt.bin";
    std::string deck_text;
    {   std::ifstream df(argv[1]);
        std::stringstream ss; ss << df.rdbuf(); deck_text = ss.str(); }

    long n0 = 0;                   // resume point (0 = fresh start)
    if (resume) {
        std::string saved_deck;
        n0 = checkpoint_load(ckpt_path, sim, &saved_deck);
        sim.set_step_count(n0);
        if (saved_deck != deck_text)
            std::fprintf(stderr, "chirp2d: WARNING deck text differs from the "
                                 "checkpointed one — resuming anyway\n");
        std::printf("resumed %s at step %ld (t=%.0f)\n",
                    ckpt_path.c_str(), n0, n0 * rp.dt);
    }

    if (!resume && rp.deltaf && q.wdnoise > 0.0) {
        // persistent δf sampling noise: wd(0) random, rms = wdnoise
        std::vector<float> w0(sim.particles().n);
        std::srand((unsigned)rp.rng_seed + 7);
        const float A = (float)(q.wdnoise * std::sqrt(3.0));   // uniform ±A
        for (auto& w : w0) w = A * (2.f * std::rand() / (float)RAND_MAX - 1.f);
        CUDA_CHECK(cudaMemcpy(sim.particles().wd.data(), w0.data(),
                              w0.size() * 4, cudaMemcpyHostToDevice));
    }

    if (!resume && d.bnoise > 0.0) {
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

    // --fvdiag: read-only gather kernels only, sim state untouched (off =
    // zero extra launches). Window per plan hard gate: |i - i_eq| < 150
    // CELLS, i_eq from b0_xc (= nx/2 on centered decks). wl accumulates
    // every wl_acc steps (dt=0.15: >=21 samples/period up to 0.5 wce, no
    // aliasing; k_workledger costs 4x a push step — measured 6.5ms vs 1.6ms
    // at 39M markers, double-atomic bound — so 20-step sampling holds the
    // mechanism-run overhead at ~13%) and dumps+resets on the energy
    // cadence; a --resume restart loses only the partial wl interval since
    // its last dump. This ledger is diagnostic-grade (v_par structure and
    // sign flips), NOT the time-centered closure ledger — that is the rsm
    // m1ledger (97f1530).
    const int fv_hw = 150, fv_every = 2000, f2d_every = 10000;
    const int wl_acc = 20, wl_every = 2000;
    const int fv_nv = 600, fv_npar = 240, fv_nperp = 120;
    const int wl_nreg = 16, wl_nwb = 240;
    const float fv_vmax = 0.6f;
    const int i_eq = (int)(rp.b0_xc / g.dx + 0.5);
    gapdiag::EqFvDiag* fvd = nullptr;
    gapdiag::GapDiags*  wld = nullptr;   // wl ledger only (fv member unused)
    if (fvdiag) {
        fvd = new gapdiag::EqFvDiag(i_eq, fv_hw, fv_nv, fv_npar, fv_nperp, fv_vmax);
        wld = new gapdiag::GapDiags(wl_nreg, 2, 2, wl_nwb, fv_vmax);
    }
    // probe x-locations: [diagnostics] probes = offsets from the equator
    // (b0_xc) in physical units; default equator +/-100, +/-200 c/wpe
    std::vector<double> poff = d.probes;
    if (poff.empty()) poff = {-200.0, -100.0, 0.0, 100.0, 200.0};
    const int nprobe = (int)poff.size();
    std::vector<int> probe_ix(nprobe);
    for (int p = 0; p < nprobe; ++p)
        probe_ix[p] = (int)((rp.b0_xc + poff[p]) / g.dx);

    // On resume: truncate probe.bin to exactly the records written up to n0
    // (a mid-interval kill may have left a partial tail), then append.
    if (resume) {
        std::error_code ec;
        std::filesystem::resize_file(outdir + "/probe.bin",
            (uintmax_t)(n0 / probe_every) * 2 * nprobe * 4, ec);
        if (ec) std::fprintf(stderr, "chirp2d: probe.bin truncate: %s\n",
                             ec.message().c_str());
        std::error_code ec2;   // probe_e.bin may not exist in pre-eline runs
        std::filesystem::resize_file(outdir + "/probe_e.bin",
            (uintmax_t)(n0 / probe_every) * 2 * nprobe * 4, ec2);
    }
    std::FILE* fpb = std::fopen((outdir + "/probe.bin").c_str(), resume ? "ab" : "wb");
    std::FILE* fpe = std::fopen((outdir + "/probe_e.bin").c_str(), resume ? "ab" : "wb");
    std::FILE* fen = std::fopen((outdir + "/energy.csv").c_str(), resume ? "a" : "w");
    if (!resume) std::fprintf(fen, "step,time,WE,WB,wd_sum,wd_rms,wd_max\n");
    // A0 ledger (PLAN_TWO_TRACK v2.1): interval-integrated, time-centered
    // m=1 work split kinetic/fluid + W1 snapshot for the closure test
    // dW1 + dW_kin + dW_fld ≈ boundary/damp losses. Columns are ENERGIES
    // over the preceding interval (raw sums × 2·dV·dt), not sampled powers.
    std::FILE* fld = nullptr;
    if (rp.rsm && rp.rsm_ledger) {
        fld = std::fopen((outdir + "/m1ledger.csv").c_str(), resume ? "a" : "w");
        if (!resume)
            std::fprintf(fld, "step,time,W1,dW_kin,dW_kin_x,dW_fld,dW_fld_x\n");
    }
    std::FILE* frf = nullptr;
    if (rfr.on) {
        frf = std::fopen((outdir + "/refresh.csv").c_str(), resume ? "a" : "w");
        if (!resume) std::fprintf(frf, "step,time,redraws,dE_injected,draw_fails,"
                                       "precips,dE_precip\n");
    }
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
        if (d.species.size() == 2)
            std::fprintf(fm, "nh2 %.9g\nppc2 %d\nuth2 %.9g %.9g %.9g\n",
                         d.species[1].density, d.species[1].ppc,
                         d.species[1].uth[0], d.species[1].uth[1], d.species[1].uth[2]);
        std::fprintf(fm, "eline 1\njline 1\nprobe_e 1\n");
        if (fvdiag)
            std::fprintf(fm, "fvdiag 1\nfv_ieq %d\nfv_hw %d\nfv_nv %d\n"
                             "fv_npar %d\nfv_nperp %d\nfv_vmax %.9g\n"
                             "fv_every %d\nf2d_every %d\n"
                             "wl_nreg %d\nwl_nwb %d\nwl_acc %d\nwl_every %d\n",
                         i_eq, fv_hw, fv_nv, fv_npar, fv_nperp, (double)fv_vmax,
                         fv_every, f2d_every, wl_nreg, wl_nwb, wl_acc, wl_every);
        for (int p = 0; p < nprobe; ++p) std::fprintf(fm, "probe_ix %d\n", probe_ix[p]);
        std::fclose(fm); }

    std::vector<float> by(g.real_size()), bz(g.real_size());
    std::vector<float> ex(g.real_size()), ey(g.real_size()), ez(g.real_size());
    std::vector<float> jx(g.real_size()), jy(g.real_size()), jz(g.real_size());
    const long nsteps = rp.nsteps;
    for (long n = n0 + 1; n <= nsteps; ++n) {
        sim.step();
        if (rfr.on) rfr.apply(sim.particles(), g, rp, n, sim.stream());
        if (wld && n % wl_acc == 0)
            wld->wl_accum(sim.particles(), sim.fields(), rp,
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
            CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), by.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ey.data(), sim.fields().ey_.data(), ey.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ez.data(), sim.fields().ez_.data(), ez.size() * 4, cudaMemcpyDeviceToHost));
        }
        if (want_probe) {
            std::vector<float> pb(2 * nprobe), pe(2 * nprobe);
            for (int p = 0; p < nprobe; ++p) {
                pb[2 * p] = by[probe_ix[p]]; pb[2 * p + 1] = bz[probe_ix[p]];
                pe[2 * p] = ey[probe_ix[p]]; pe[2 * p + 1] = ez[probe_ix[p]];
            }
            std::fwrite(pb.data(), 4, 2 * nprobe, fpb);
            std::fwrite(pe.data(), 4, 2 * nprobe, fpe);
        }
        if (want_line) {
            CUDA_CHECK(cudaMemcpy(ex.data(), sim.fields().ex_.data(), ex.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(jx.data(), sim.fields().jx_.data(), jx.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(jy.data(), sim.fields().jy_.data(), jy.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(jz.data(), sim.fields().jz_.data(), jz.size() * 4, cudaMemcpyDeviceToHost));
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* f = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, g.nx, f);
            std::fwrite(bz.data(), 4, g.nx, f);
            std::fclose(f);
            if (!no_eline) {
                std::snprintf(fn, sizeof fn, "%s/eline_%06ld.bin", outdir.c_str(), n / bline_every);
                f = std::fopen(fn, "wb");
                std::fwrite(ex.data(), 4, g.nx, f);
                std::fwrite(ey.data(), 4, g.nx, f);
                std::fwrite(ez.data(), 4, g.nx, f);
                std::fclose(f);
            }
            std::snprintf(fn, sizeof fn, "%s/jline_%06ld.bin", outdir.c_str(), n / bline_every);
            f = std::fopen(fn, "wb");
            std::fwrite(jx.data(), 4, g.nx, f);
            std::fwrite(jy.data(), 4, g.nx, f);
            std::fwrite(jz.data(), 4, g.nx, f);
            std::fclose(f);
            if (rp.rsm) {   // m1 complex lines: B1y, B1z, E1x (E_par), E1y
                std::snprintf(fn, sizeof fn, "%s/m1line_%06ld.bin", outdir.c_str(), n / bline_every);
                f = std::fopen(fn, "wb");
                std::vector<float2> mb(g.nx);
                RsmState& r = sim.rsm();
                for (auto* arr : { &r.b1y, &r.b1z, &r.e1x, &r.e1y }) {
                    CUDA_CHECK(cudaMemcpy(mb.data(), arr->data(),
                                          g.nx * sizeof(float2), cudaMemcpyDeviceToHost));
                    std::fwrite(mb.data(), sizeof(float2), g.nx, f);
                }
                std::fclose(f);
            }
        }
        if (fvd) {
            char fn[512];
            if (n % fv_every == 0) {
                std::snprintf(fn, sizeof fn, "%s/fvline_%06ld.bin",
                              outdir.c_str(), n / fv_every);
                fvd->line_snapshot(sim.particles(), rp, g, sim.stream(), fn);
            }
            if (n % f2d_every == 0) {
                std::snprintf(fn, sizeof fn, "%s/fv2d_%06ld.bin",
                              outdir.c_str(), n / f2d_every);
                fvd->f2d_snapshot(sim.particles(), rp, g, sim.stream(), fn);
            }
            if (n % wl_every == 0) {
                std::snprintf(fn, sizeof fn, "%s/wl_%06ld.bin",
                              outdir.c_str(), n / wl_every);
                wld->wl_write_reset(sim.stream(), fn);
            }
        }
        if (n % 2000 == 0) {
            const auto e = sim.field_energy();
            const auto w = sim.wd_stats();
            std::fprintf(fen, "%ld,%.6g,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                         n, n * rp.dt, e.we, e.wb, w.sum, w.rms, w.max);
            std::fflush(fen);
            if (fld) {
                RsmState& r = sim.rsm();
                double raw[4];
                r.ledger_read(sim.stream(), raw);       // resets accumulators
                const double dV = g.dx * g.dy;
                const double sc = 2.0 * dV * rp.dt;     // ±k1 pair, work → energy
                // W1 snapshot (2·W1 of the pair, same convention as rsm_band)
                std::vector<float2> mb(g.nx);
                double w1 = 0;
                const double c2 = rp.c * rp.c;
                for (auto* arr : { &r.e1x, &r.e1y, &r.e1z }) {
                    CUDA_CHECK(cudaMemcpy(mb.data(), arr->data(),
                                          g.nx * sizeof(float2), cudaMemcpyDeviceToHost));
                    for (auto& z : mb)
                        w1 += 0.5 * ((double)z.x * z.x + (double)z.y * z.y);
                }
                for (auto* arr : { &r.b1x, &r.b1y, &r.b1z }) {
                    CUDA_CHECK(cudaMemcpy(mb.data(), arr->data(),
                                          g.nx * sizeof(float2), cudaMemcpyDeviceToHost));
                    for (auto& z : mb)
                        w1 += 0.5 * c2 * ((double)z.x * z.x + (double)z.y * z.y);
                }
                w1 *= 2.0 * dV;
                std::fprintf(fld, "%ld,%.6g,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                             n, n * rp.dt, w1, sc * raw[0], sc * raw[1],
                             sc * raw[2], sc * raw[3]);
                std::fflush(fld);
            }
            if (frf) {
                const auto r = rfr.drain(sim.stream());
                std::fprintf(frf, "%ld,%.6g,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                             n, n * rp.dt, r.redraws, r.de, r.fails,
                             r.precips, r.de_precip);
                std::fflush(frf);
            }
            if (n % 20000 == 0)
                std::printf("t=%8.0f  WB=%.3e  wd_rms=%.3e\n", n * rp.dt, e.wb, w.rms);
        }
    }
    if (frf) std::fclose(frf);
    if (fld) std::fclose(fld);
    std::fclose(fpb); std::fclose(fpe); std::fclose(fen);
    std::printf("done: %s\n", outdir.c_str());
    return 0;
}
