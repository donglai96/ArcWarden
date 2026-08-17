// warden2d — the reborn 2D runner (PLAN_2D_REBORN).
//
// P0 scope: deck parse → finalize (derived geometry, gate battery, memory
// pre-flight) → run card → refuse-or-proceed. The physics loop (Yee +
// cold_full twin + multi-δf kinetic engine) lands P1/P2; this stub is the
// contract that no run can start without passing the pre-registered
// numerical gates, on any GPU size (the deck is the only thing that changes
// between the 32 GB rung and the 141–288 GB rungs).
//
// Usage: ./warden2d <deck.ini> [outdir]

#include "pic2d/deck2d.hpp"

#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <deck.ini> [outdir]\n", argv[0]);
        return 1;
    }
    arc2d::Deck2D d;
    try {
        d = arc2d::load_deck2d(argv[1]);
        arc2d::finalize_deck2d(d);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "warden2d: %s\n", e.what());
        return 1;
    }

    std::printf("ArcWarden 2d-reborn — warden2d (P0 scaffold)\n");
    std::printf("deck: %s\n\n", argv[1]);
    arc2d::print_deck2d_report(d);

    // device memory sanity against the pre-flight
    size_t mfree = 0, mtotal = 0;
    if (cudaMemGetInfo(&mfree, &mtotal) == cudaSuccess) {
        const double need = d.mem_fields_gb + d.mem_markers_gb;
        std::printf("  device   : %.1f GB free / %.1f GB total — deck needs %.1f GB [%s]\n",
                    mfree / 1e9, mtotal / 1e9, need,
                    need < mfree / 1e9 ? "fits" : "DOES NOT FIT");
    }

    if (!d.ok()) {
        std::fprintf(stderr, "\nwarden2d: hard gate FAILED — deck refused (fix the deck, "
                             "not the gate)\n");
        return 2;
    }
    std::printf("\nall hard gates pass — physics loop lands in P1/P2 "
                "(PLAN_2D_REBORN §6)\n");
    return 0;
}
