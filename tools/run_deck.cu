// ArcWarden — run a simulation from a text input deck (no recompile per setup).
//
//   ./run_deck <deck-file> [outdir]
//
// Reads the deck (grid, time, plasma, species list), runs the PIC loop, and dumps
// (x, vx) phase-space frames + a manifest.csv into outdir for the plot scripts.
// Two-stream / bump-on-tail / single-Maxwellian are all just different decks.

#include "pic/deck.hpp"
#include "pic/fields.hpp"
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

// Longitudinal field profile Ex(x), averaged over y (the setup is 1D along x).
static std::vector<float> field_profile(const Fields& F, const Grid& g) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ex(g.real_size());
    CUDA_CHECK(cudaMemcpy(ex.data(), F.Ex.data(), F.Ex.bytes(), cudaMemcpyDeviceToHost));
    std::vector<float> prof(g.nx);
    for (int ix = 0; ix < g.nx; ++ix) {
        double s = 0.0;
        for (int iy = 0; iy < g.ny; ++iy) s += ex[g.idx(ix, iy)];
        prof[ix] = static_cast<float>(s / g.ny);
    }
    return prof;
}

// Write Ex(x) for one frame, so it can be plotted under the phase space.
static void dump_field(const Fields& F, const Grid& g, const std::string& path) {
    const std::vector<float> prof = field_profile(F, g);
    std::ofstream f(path);
    f << "x,Ex\n";
    for (int ix = 0; ix < g.nx; ++ix) f << (ix + 0.5) * g.dx << ',' << prof[ix] << '\n';
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
    char name[64], fname[64];
    if (do_frames) {
        man.open(outdir + "/manifest.csv");
        man << "frame,step,time,file,field\n";
    }
    auto write_frame = [&](long step) {
        if (!do_frames) return;
        std::snprintf(name,  sizeof(name),  "frame_%04d.csv", frame);
        std::snprintf(fname, sizeof(fname), "field_%04d.csv", frame);
        dump_phase(sim.particles(), g, outdir + "/" + name,  sample);
        dump_field(sim.fields(),    g, outdir + "/" + fname);
        man << frame << ',' << step << ',' << step * d.rp.dt << ',' << name << ',' << fname << '\n';
        man.flush();
        ++frame;
    };

    // Dense E(x,t) history for an omega-k spectrum (field is tiny, sampled often).
    // One row per sample = Ex(x) averaged over y; header carries dx and dt_sample.
    const bool do_hist = d.field_history_every > 0;
    std::ofstream hist;
    long hist_rows = 0;
    if (do_hist) {
        fs::create_directories(outdir);
        hist.open(outdir + "/field_xt.csv");
        hist << "# dx=" << g.dx << " dt=" << d.field_history_every * d.rp.dt
             << " nx=" << g.nx << "\n";
    }
    auto write_hist = [&] {
        if (!do_hist) return;
        const std::vector<float> prof = field_profile(sim.fields(), g);
        for (int ix = 0; ix < g.nx; ++ix) hist << (ix ? "," : "") << prof[ix];
        hist << '\n';
        ++hist_rows;
    };

    write_frame(0);
    write_hist();
    const auto t0 = std::chrono::steady_clock::now();
    for (long n = 0; n < d.rp.nsteps; ++n) {
        sim.step(n);
        if (do_frames && (n + 1) % d.dump_every == 0) write_frame(n + 1);
        if (do_hist   && (n + 1) % d.field_history_every == 0) write_hist();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    if (do_frames)
        std::printf("wrote %d frames to %s/ (<= %d pts/frame)\n", frame, outdir.c_str(), DUMP_CAP);
    if (do_hist)
        std::printf("wrote field_xt.csv (%ld samples x %d cells, dt=%.3g)\n",
                    hist_rows, g.nx, d.field_history_every * d.rp.dt);
    std::printf("ran %ld steps in %.3f s (%.3f ms/step, %.1f M particle-updates/s)\n",
                d.rp.nsteps, secs, 1e3 * secs / d.rp.nsteps,
                double(N) * d.rp.nsteps / secs / 1e6);
    return 0;
}
