// ArcWarden — run a simulation from a text input deck (no recompile per setup).
//
//   ./run_deck <deck-file> [outdir] [--em] [--c=20] [--ndc=2]
//
// Reads the deck (grid, time, plasma, species list), runs the PIC loop, and dumps
// (x, vx) phase-space frames + a manifest.csv into outdir for the plot scripts.
// Two-stream / bump-on-tail / single-Maxwellian are all just different decks.
//
// --em runs the SPECTRAL DARWIN field model instead of electrostatic (same deck).
// For an electrostatic instability (e.g. bump-on-tail) Darwin reproduces the
// longitudinal physics via E_L while B / E_T stay negligible — a check that the EM
// code reduces to ES. c (light speed) and ndc (transverse iterations) default to
// 20 / 2; override with --c= / --ndc=.

#include "pic/deck.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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

static double sum_sq(const DeviceArray<float>& a) {
    std::vector<float> h(a.size());
    CUDA_CHECK(cudaMemcpy(h.data(), a.data(), a.bytes(), cudaMemcpyDeviceToHost));
    double s = 0; for (float v : h) s += (double)v * v; return s;
}

template<class Cfg>
static int run_sim(Deck& d, const std::string& outdir, const char* deckname) {
    namespace fs = std::filesystem;
    const bool do_frames = d.dump_every > 0;
    if (do_frames) { fs::create_directories(outdir); d.rp.dump_every = d.dump_every; }
    else           { d.rp.dump_every = (1L << 30); }
    const std::string energy_csv = do_frames ? (outdir + "/energy.csv") : "";

    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    Simulation<Cfg> sim(g, d.rp, d.species, energy_csv);
    sim.init();
    const long long N = static_cast<long long>(sim.particles().n);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const bool em = (Cfg::field_model == FieldModel::Darwin);
    std::printf("ArcWarden deck '%s': %lld particles, %dx%d grid, %s load, %s, on %s\n",
                deckname, N, g.nx, g.ny, d.rp.noisy_load ? "noisy" : "quiet",
                em ? "DARWIN EM" : "electrostatic", prop.name);
    if (em) std::printf("  Darwin: c=%.3g  ndc=%d\n", d.rp.c, d.rp.ndc);
    for (const auto& s : d.species) {
        const double w = s.density * g.dx * g.dy / s.ppc;
        std::printf("  species %-10s density=%.3g ppc=%d ufl=(%.3g,%.3g,%.3g) "
                    "uth=(%.3g,%.3g,%.3g) weight=%.3g\n",
                    s.name.c_str(), s.density, s.ppc, s.ufl[0], s.ufl[1], s.ufl[2],
                    s.uth[0], s.uth[1], s.uth[2], w);
    }

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
    if (do_frames) { man.open(outdir + "/manifest.csv"); man << "frame,step,time,file\n"; }
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

    if constexpr (Cfg::field_model == FieldModel::Darwin) {
        // electrostatic instability check: B / E_T energy should stay << E_L energy.
        const Fields& f = sim.fields();
        const double cell = 0.5 * g.dx * g.dy;
        const double eE = cell * (sum_sq(f.Ex) + sum_sq(f.Ey) + sum_sq(f.Ez));   // E_total
        const double eB = cell * (sum_sq(f.Bx) + sum_sq(f.By) + sum_sq(f.Bz));
        const double eT = cell * (sum_sq(f.ETx) + sum_sq(f.ETy) + sum_sq(f.ETz));
        std::printf("  final field energy: electric(E_total)=%.4e  magnetic=%.4e (%.2e of E)"
                    "  transverse-E=%.4e (%.2e of E)\n",
                    eE, eB, eE > 0 ? eB / eE : 0.0, eT, eE > 0 ? eT / eE : 0.0);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <deck-file> [outdir] [--em] [--c=20] [--ndc=2]\n", argv[0]);
        return 2;
    }
    bool em = false;
    double c = 20.0; int ndc = 2;
    std::string deckfile, outdir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--em") em = true;
        else if (a.rfind("--c=", 0) == 0)   c = std::atof(a.c_str() + 4);
        else if (a.rfind("--ndc=", 0) == 0) ndc = std::atoi(a.c_str() + 6);
        else if (deckfile.empty())          deckfile = a;
        else if (outdir.empty())            outdir = a;
    }

    Deck d;
    try { d = load_deck(deckfile.c_str()); }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    if (outdir.empty()) outdir = d.outdir;

    if (em) {
        d.rp.c = c; d.rp.ndc = ndc;
        return run_sim<CfgDarwin>(d, outdir, deckfile.c_str());
    }
    return run_sim<arc::Cfg>(d, outdir, deckfile.c_str());
}
