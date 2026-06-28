// ArcWarden — run a simulation from a text input deck (no recompile per setup).
//
//   ./run_deck <deck-file> [outdir]
//
// Reads the deck (grid, time, plasma, species list), runs the PIC loop, and dumps
// (x, vx) phase-space frames + a manifest.csv into outdir for the plot scripts.
// Two-stream / bump-on-tail / single-Maxwellian are all just different decks.

#include "pic/deck.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace arc;

static constexpr int DUMP_CAP = 100000;   // max points written per frame

static void dump_phase(const Particles& P, const Grid& g, const std::string& path,
                       const std::vector<int>& idx) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const int N = static_cast<int>(P.n);
    std::vector<float> x(N), vx(N);
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost));
    std::ofstream f(path);
    f << "x,vx\n";
    for (int i : idx) {
        float xd = x[i];
        if (xd < 0.0f) xd += g.nx; else if (xd >= g.nx) xd -= g.nx;
        f << xd * g.dx << ',' << vx[i] << '\n';
    }
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <deck-file> [outdir]\n", argv[0]);
        return 2;
    }
    Deck d;
    try {
        d = load_deck(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    const std::string outdir = (argc > 2) ? argv[2] : d.outdir;
    const bool do_frames = d.dump_every > 0;

    // Diagnostics cadence: compute (and CSV-log) energy at the frame cadence.
    // Without this, dump_every<=0 makes Diagnostics run a full-particle reduction
    // EVERY step (it would otherwise dominate run time and be discarded).
    if (do_frames) { fs::create_directories(outdir); d.rp.dump_every = d.dump_every; }
    else           { d.rp.dump_every = (1L << 30); }   // effectively disabled
    const std::string energy_csv = do_frames ? (outdir + "/energy.csv") : "";

    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    Simulation<> sim(g, d.rp, d.species, energy_csv);
    sim.init();
    const long long N = static_cast<long long>(sim.particles().n);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("ArcWarden deck '%s': %lld particles, %dx%d grid, %s load, on %s\n",
                argv[1], N, g.nx, g.ny, d.rp.noisy_load ? "noisy" : "quiet", prop.name);
    for (const auto& s : d.species) {
        const double w = s.density * g.dx * g.dy / s.ppc;
        std::printf("  species %-10s density=%.3g ppc=%d ufl=(%.3g,%.3g,%.3g) "
                    "uth=(%.3g,%.3g,%.3g) weight=%.3g\n",
                    s.name.c_str(), s.density, s.ppc, s.ufl[0], s.ufl[1], s.ufl[2],
                    s.uth[0], s.uth[1], s.uth[2], w);
    }

    // fixed random subsample of indices (same set every frame)
    std::vector<int> sample(N);
    std::iota(sample.begin(), sample.end(), 0);
    if (N > DUMP_CAP) {
        std::mt19937 rng(12345);
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(DUMP_CAP);
        std::sort(sample.begin(), sample.end());
    }

    int frame = 0;
    std::ofstream man;
    char name[64];
    if (do_frames) {
        man.open(outdir + "/manifest.csv");
        man << "frame,step,time,file\n";
    }
    auto write_frame = [&](long step) {
        if (!do_frames) return;
        std::snprintf(name, sizeof(name), "frame_%04d.csv", frame);
        dump_phase(sim.particles(), g, outdir + "/" + name, sample);
        man << frame << ',' << step << ',' << step * d.rp.dt << ',' << name << '\n';
        man.flush();
        ++frame;
    };

    write_frame(0);
    const auto t0 = std::chrono::steady_clock::now();
    for (long n = 0; n < d.rp.nsteps; ++n) {
        sim.step(n);
        if (do_frames && (n + 1) % d.dump_every == 0) write_frame(n + 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    if (do_frames)
        std::printf("wrote %d frames to %s/ (<= %d pts/frame)\n", frame, outdir.c_str(), DUMP_CAP);
    std::printf("ran %ld steps in %.3f s (%.3f ms/step, %.1f M particle-updates/s)\n",
                d.rp.nsteps, secs, 1e3 * secs / d.rp.nsteps,
                double(N) * d.rp.nsteps / secs / 1e6);
    return 0;
}
