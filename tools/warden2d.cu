// warden2d — the reborn 2D production runner (PLAN_2D_REBORN, P3a).
//
// Flow: deck parse → finalize (derived geometry, hard-gate battery, memory
// pre-flight; FAIL refuses launch) → Sim2D build (multi-species load) →
// run with live progress → outputs into <outdir>:
//   energy.csv         t, W_EM, W_cold, per-species wd_rms
//   f2d_XXXXXXX.bin    6-component float32 field snapshot [6][nz][nx]
//   meta.txt           run card + geometry + species table
// Every experiment is a deck; this binary never changes per experiment.
//
// Usage: ./warden2d <deck.ini> [outdir] [--nsteps=N] [--preflight]

#include "pic2d/checkpoint2d.hpp"
#include "pic2d/sim2d.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace arc2d;

static void snapshot(Fields2D& F, const std::string& outdir, long n) {
    const size_t nc = size_t(F.nx) * F.nz;
    std::vector<float> h(nc);
    char path[512];
    std::snprintf(path, sizeof path, "%s/f2d_%07ld.bin", outdir.c_str(), n);
    FILE* f = std::fopen(path, "wb");
    for (auto* a : {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz}) {
        CUDA_CHECK(cudaMemcpy(h.data(), a->data(), nc * 4, cudaMemcpyDeviceToHost));
        std::fwrite(h.data(), 4, nc, f);
    }
    std::fclose(f);
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <deck.ini> [outdir] [--nsteps=N] [--preflight]\n",
                     argv[0]);
        return 1;
    }
    std::string outdir = "warden2d_out";
    long nsteps_cli = -1, ckpt_every = 0, sort_cli = -1;
    bool preflight_only = false, resume = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--nsteps=", 9)) nsteps_cli = atol(argv[i] + 9);
        else if (!std::strncmp(argv[i], "--ckpt=", 7)) ckpt_every = atol(argv[i] + 7);
        else if (!std::strcmp(argv[i], "--resume")) resume = true;
        else if (!std::strncmp(argv[i], "--sort=", 7)) sort_cli = atol(argv[i] + 7);
        else if (!std::strcmp(argv[i], "--preflight")) preflight_only = true;
        else if (argv[i][0] != '-') outdir = argv[i];
    }

    Deck2D d;
    try {
        d = load_deck2d(argv[1]);
        finalize_deck2d(d);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "warden2d: %s\n", e.what());
        return 1;
    }
    if (nsteps_cli >= 0) d.nsteps = nsteps_cli;

    std::printf("ArcWarden 2d-reborn — warden2d\ndeck: %s → %s\n\n", argv[1],
                outdir.c_str());
    print_deck2d_report(d);

    size_t mfree = 0, mtotal = 0;
    if (cudaMemGetInfo(&mfree, &mtotal) == cudaSuccess) {
        const double need = d.mem_fields_gb + d.mem_markers_gb;
        std::printf("  device   : %.1f GB free / %.1f GB total — deck needs %.1f GB [%s]\n",
                    mfree / 1e9, mtotal / 1e9, need,
                    need < mfree / 1e9 ? "fits" : "DOES NOT FIT");
    }
    if (!d.ok()) {
        std::fprintf(stderr, "\nwarden2d: hard gate FAILED — deck refused "
                             "(fix the deck, not the gate)\n");
        return 2;
    }
    if (preflight_only || d.nsteps <= 0) {
        std::printf("\npre-flight only (nsteps = %ld) — pass --nsteps=N to run\n",
                    d.nsteps);
        return 0;
    }

    std::filesystem::create_directories(outdir);
    Sim2D S;
    if (sort_cli >= 0) S.sort_every = sort_cli;
    S.build(d);
    if (resume) load_checkpoint(S, outdir + "/ckpt.bin");
    std::printf("\nloaded %zu species:\n", S.sp.size());
    for (size_t i = 0; i < S.sp.size(); ++i)
        std::printf("  %-10s %.2e markers  %s\n", S.sp[i].name.c_str(),
                    double(S.sp[i].mk->n), S.sp[i].C.deltaf ? "δf" : "full-f");

    FILE* ecsv = std::fopen((outdir + "/energy.csv").c_str(), resume ? "a" : "w");
    if (!resume) {
        std::fprintf(ecsv, "t,W_EM,W_cold");
        for (auto& s : S.sp) std::fprintf(ecsv, ",wdrms_%s", s.name.c_str());
        std::fprintf(ecsv, "\n");
    }

    // meta: probe stations with LOCAL Omega_e (dual-normalization contract)
    FILE* probes = nullptr;
    std::vector<float> pline(size_t(6) * std::max(S.diag.ns, 1));
    const long probe_every = 4, sline_every = std::max(1L, d.snap_every / 4),
               fv_every = 20000;
    if (S.diag_on) {
        FILE* meta = std::fopen((outdir + "/meta.txt").c_str(), "w");
        std::fprintf(meta, "deck %s\nL0 %.2f lam_w_deg %.1f B0eq %.4f\n"
                           "line ns %d fields Epar,E1,Ey,Bpar,B1,By\n",
                     argv[1], d.bg.L0, d.lam_w * 180 / M_PI, d.bg.B0eq, S.diag.ns);
        std::fprintf(meta, "probe_every %ld dt %.4f\nprobes (lam_deg, wce_local):\n",
                     probe_every, d.dt);
        for (size_t i = 0; i < S.diag.probe_idx.size(); ++i)
            std::fprintf(meta, "  %6.1f  %.5f\n",
                         S.diag.lam_line[S.diag.probe_idx[i]], S.diag.probe_wce[i]);
        std::fprintf(meta, "fv layout [%d][%d][%d][2] vmax %.2f uqmax %.2f\n"
                           "ledger layout [%d][%d][2] every %ld\n",
                     DIAG_NREG, S.diag.npar, S.diag.nperp, S.diag.vmax,
                     S.diag.uqmax, DIAG_NREG, S.diag.nvb, 8L);
        std::fclose(meta);
        probes = std::fopen((outdir + "/probes.bin").c_str(), resume ? "ab" : "wb");
    }

    const long n_start = S.nstep;
    const auto t0 = std::chrono::steady_clock::now();
    for (long n = n_start; n < d.nsteps; ++n) {
        S.step();
        if (ckpt_every > 0 && (n + 1) % ckpt_every == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            save_checkpoint(S, outdir + "/ckpt.bin");
            std::printf("  checkpoint at step %ld\n", n + 1);
        }
        if (S.diag_on) {
            if (n % 8 == 7) S.diag_ledger(float(8 * d.dt));
            if (n % probe_every == probe_every - 1) {
                S.diag_line();
                CUDA_CHECK(cudaMemcpy(pline.data(), S.diag.line_out.data(),
                                      size_t(6) * S.diag.ns * 4,
                                      cudaMemcpyDeviceToHost));
                for (int c = 0; c < 6; ++c)
                    for (int pidx : S.diag.probe_idx)
                        std::fwrite(&pline[size_t(c) * S.diag.ns + pidx], 4, 1,
                                    probes);
            }
            if (d.snap_every > 0 && n % sline_every == sline_every - 1) {
                char pth[512];
                std::snprintf(pth, sizeof pth, "%s/sline_%07ld.bin",
                              outdir.c_str(), n + 1);
                S.diag.dump_line(pth);
            }
            if (n % fv_every == fv_every - 1) {
                S.diag_fv();
                for (size_t i = 0; i < S.sp.size(); ++i) {
                    char lp[512], fp[512];
                    std::snprintf(lp, sizeof lp, "%s/ledger_%s_%07ld.bin",
                                  outdir.c_str(), S.sp[i].name.c_str(), n + 1);
                    std::snprintf(fp, sizeof fp, "%s/fv_%s_%07ld.bin",
                                  outdir.c_str(), S.sp[i].name.c_str(), n + 1);
                    S.diag.dump_and_reset(int(i), lp, fp);
                }
            }
        }
        if (d.snap_every > 0 && n % d.snap_every == d.snap_every - 1)
            snapshot(S.F, outdir, n + 1);
        if (n % d.energy_every == d.energy_every - 1) {
            double W[2];
            S.F.energies(W);
            std::fprintf(ecsv, "%.3f,%.6e,%.6e", S.time, W[0], W[1]);
            for (size_t i = 0; i < S.sp.size(); ++i)
                std::fprintf(ecsv, ",%.6e", S.wd_rms(int(i)));
            std::fprintf(ecsv, "\n");
            std::fflush(ecsv);
        }
        if (n % 10000 == 9999) {
            CUDA_CHECK(cudaDeviceSynchronize());
            if (!S.healthy()) {
                std::fprintf(stderr,
                             "warden2d: HEALTH CHECK FAILED at step %ld — "
                             "writing emergency checkpoint and aborting\n",
                             n + 1);
                save_checkpoint(S, outdir + "/ckpt_emergency.bin");
                return 3;
            }
            double W[2];
            S.F.energies(W);
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            const double per = el / double(n + 1 - n_start);
            std::printf("  step %8ld/%ld  t=%9.1f  W_EM %.3e  wd %.2e  runaway %llu  "
                        "%.2f ms/step  ETA %.0f min\n",
                        n + 1, d.nsteps, S.time, W[0], S.wd_rms(0),
                        S.runaway_count(), 1e3 * per,
                        per * (d.nsteps - n) / 60);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::fclose(ecsv);
    if (probes) std::fclose(probes);
    snapshot(S.F, outdir, d.nsteps);
    std::printf("done: %ld steps, outputs in %s/\n", d.nsteps, outdir.c_str());
    return 0;
}
