// bench2d — per-step timing harness for the pic2d stack (P3d tile-sort work).
// Usage: ./bench2d <deck.ini> [--sort=N] [--nsteps=N]
// Builds the sim, warms up, then times nsteps with cudaEvents and reports
// ms/step split into (sort amortized) vs (physics step).
#include "pic2d/sim2d.hpp"
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
    using namespace arc2d;
    setbuf(stdout, nullptr);
    long nsteps = 200, sort_cli = -1, tiled_cli = -1;
    for (int i = 2; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--sort=", 7)) sort_cli = atol(argv[i] + 7);
        if (!std::strncmp(argv[i], "--tiled=", 8)) tiled_cli = atol(argv[i] + 8);
        if (!std::strncmp(argv[i], "--nsteps=", 9)) nsteps = atol(argv[i] + 9);
    }
    Deck2D d = load_deck2d(argv[1]);
    finalize_deck2d(d);
    if (!d.ok()) { std::printf("deck refused\n"); return 2; }
    Sim2D S;
    if (sort_cli >= 0) S.sort_every = sort_cli;
    if (tiled_cli >= 0) S.deposit_tiled = int(tiled_cli);
    S.build(d);
    double nm = 0;
    for (auto& s : S.sp) nm += double(s.mk->n);
    std::printf("markers %.2e cells %.2e sort_every %ld tiled %d\n",
                nm, double(d.nx) * d.nz, S.sort_every, S.deposit_tiled);
    for (int i = 0; i < 30; ++i) S.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (long i = 0; i < nsteps; ++i) S.step();
    cudaEventRecord(b);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    std::printf("%.2f ms/step  (%.2e marker-steps/s)\n", ms / nsteps,
                nm * nsteps / (ms * 1e-3) );
    return 0;
}
