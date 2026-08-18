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
    long nsteps_cli = -1;
    bool preflight_only = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--nsteps=", 9)) nsteps_cli = atol(argv[i] + 9);
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
    S.build(d);
    std::printf("\nloaded %zu species:\n", S.sp.size());
    for (size_t i = 0; i < S.sp.size(); ++i)
        std::printf("  %-10s %.2e markers  %s\n", S.sp[i].name.c_str(),
                    double(S.sp[i].mk->n), S.sp[i].C.deltaf ? "δf" : "full-f");

    FILE* ecsv = std::fopen((outdir + "/energy.csv").c_str(), "w");
    std::fprintf(ecsv, "t,W_EM,W_cold");
    for (auto& s : S.sp) std::fprintf(ecsv, ",wdrms_%s", s.name.c_str());
    std::fprintf(ecsv, "\n");

    const auto t0 = std::chrono::steady_clock::now();
    for (long n = 0; n < d.nsteps; ++n) {
        S.step();
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
            double W[2];
            S.F.energies(W);
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("  step %8ld/%ld  t=%9.1f  W_EM %.3e  %.2f ms/step  "
                        "ETA %.0f min\n",
                        n + 1, d.nsteps, S.time, W[0], 1e3 * el / (n + 1),
                        el / (n + 1) * (d.nsteps - n) / 60);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::fclose(ecsv);
    snapshot(S.F, outdir, d.nsteps);
    std::printf("done: %ld steps, outputs in %s/\n", d.nsteps, outdir.c_str());
    return 0;
}
