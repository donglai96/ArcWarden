// ArcWarden — unified, deck-driven runner. One binary for every experiment: the physics
// comes from the deck ([grid]/[time]/[plasma]/[species] + [field]/[background]/[pump]),
// and the outputs come from a modular, deck-selected diagnostics set
// ([diagnostics] enable = spectrum kt phase_video snapshot em_energy phase_frames).
// Adding a new experiment is a new .ini — no new main(). ES vs Darwin is picked at runtime
// from [field] model. Supersedes run_deck + whistler_pump (both kept for reference).
//
//   ./arcwarden <deck.ini> [outdir] [--em] [--ppc=N] [--amp=X] [--nsteps=N]
//     outdir default "." (files land in CWD with the deck prefix, as before).

#include "pic/config.hpp"
#include "pic/deck.hpp"
#include "pic/run_meta.hpp"
#include "pic/diag/manager.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

using namespace arc;

// deck filename stem (strip dir + extension) — default output prefix.
static std::string deck_stem(const std::string& path) {
    std::string p = path;
    const auto slash = p.find_last_of("/\\"); if (slash != std::string::npos) p = p.substr(slash + 1);
    const auto dot = p.find_last_of('.');      if (dot   != std::string::npos) p = p.substr(0, dot);
    return p;
}

static std::string g_deckfile; static int g_argc = 0; static char** g_argv = nullptr;

template<class Cfg>
static int run(Deck& d, const std::string& pref, const std::string& outdir) {
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    const std::string ecsv = outdir.empty() ? "" : (outdir + "/energy.csv");
    if (!outdir.empty()) std::filesystem::create_directories(outdir);
    if (!outdir.empty()) write_run_meta(outdir, g_deckfile, g_argc, g_argv);
    d.rp.dump_every = ecsv.empty() ? (1L << 30)
                    : (d.dump_every > 0 ? d.dump_every : (d.rp.nsteps/100 > 0 ? d.rp.nsteps/100 : 1));

    Simulation<Cfg> sim(g, d.rp, d.species, ecsv);
    sim.init();
    diag::DiagManager mgr(d, d.rp, pref, outdir);

    for (long n = 0; n <= d.rp.nsteps; ++n) {
        if (n > 0) sim.step(n - 1);
        mgr.sample(sim.fields(), sim.particles(), g, d.rp, n, n * d.rp.dt);
    }
    mgr.finalize();
    return 0;
}

int main(int argc, char** argv) {
    g_argc = argc; g_argv = argv;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <deck.ini> [outdir] [--em] [--ppc=N] [--amp=X] [--nsteps=N]\n", argv[0]);
        return 2;
    }
    const std::string deckfile = argv[1];
    g_deckfile = deckfile;
    std::string outdir;
    bool em = false; int ppc = 0; double amp = 0; long nsteps = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--em")                     em = true;
        else if (a.rfind("--ppc=", 0) == 0)       ppc = std::atoi(a.c_str() + 6);
        else if (a.rfind("--amp=", 0) == 0)       amp = std::atof(a.c_str() + 6);
        else if (a.rfind("--nsteps=", 0) == 0)    nsteps = std::atol(a.c_str() + 9);
        else if (a.rfind("--", 0) != 0 && outdir.empty()) outdir = a;   // positional outdir
    }

    Deck d;
    try { d = load_deck(deckfile); }
    catch (const std::exception& e) { std::fprintf(stderr, "deck error: %s\n", e.what()); return 1; }

    if (em) d.darwin = true;
    if (ppc > 0 && !d.species.empty()) d.species[0].ppc = ppc;
    if (nsteps > 0) d.rp.nsteps = nsteps;
    const double dx = d.Lx / d.nx;
    if (d.pump_enable) {                                   // re-derive pump amps for amp override
        const double a = (amp > 0 ? amp : d.pump_amp), s = a * dx / 1e4;
        d.rp.pump_ex = d.pump_ex0 * s; d.rp.pump_ey = d.pump_ey0 * s; d.rp.pump_ez = d.pump_ez0 * s;
    }
    // default diagnostics if the deck didn't list any
    if (d.diag_enable.empty()) {
        d.diag_enable.push_back("phase_frames");
        if (d.darwin) d.diag_enable.push_back("em_energy");
    }

    const std::string base = d.prefix.empty() ? deck_stem(deckfile) : d.prefix;
    const std::string pref = (outdir.empty() ? "" : outdir + "/") + base + "_";
    const double vth = d.species.empty() ? 1.0 : d.species[0].uth[0];
    const double bmag = std::sqrt((double)d.rp.B0[0]*d.rp.B0[0] + (double)d.rp.B0[1]*d.rp.B0[1]
                                + (double)d.rp.B0[2]*d.rp.B0[2]);
    const double cost = bmag > 0 ? d.rp.B0[0]/bmag : 1.0;
    const double vr = (d.rp.pump_k0 > 0 && cost > 0) ? (d.rp.pump_w0/(d.rp.pump_k0*cost))/vth : 0.0;
    const long long N = (long long)d.nx * d.ny * (d.species.empty() ? 0 : d.species[0].ppc);
    std::printf("arcwarden [%s]  model=%s  nx=%d ny=%d N=%lld | dx=%.4f Lx=%.1f c=%.3g |B0|=%.4g | "
                "pump=%d M=%d w0=%.5g v_r/v_th=%.3f | nsteps=%ld  diag=[",
                base.c_str(), d.darwin ? "darwin" : "electrostatic", d.nx, d.ny, N,
                dx, d.Lx, d.rp.c, bmag, (int)d.rp.pump, d.pump_M, d.rp.pump_w0, vr, d.rp.nsteps);
    for (size_t i = 0; i < d.diag_enable.size(); ++i)
        std::printf("%s%s", i ? " " : "", d.diag_enable[i].c_str());
    std::printf("]  out='%s'\n", outdir.empty() ? "." : outdir.c_str());

    return d.darwin ? run<CfgDarwin>(d, pref, outdir) : run<arc::Cfg>(d, pref, outdir);
}
