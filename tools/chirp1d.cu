// ArcWarden — chirp1d: deck-driven 1D field-aligned electron-hybrid runner
// (L-shell plan M4 gate: whistler linear growth → nonlinear trapping →
// rising-tone chirping in a parabolic B0(h)).
//
// Usage:  chirp1d <deck.ini> [outdir] [--ppc=N] [--nsteps=N] [--nh=X] [--seed=N]
//
// Deck: INI, one key = value per line, # / ! comments.  Section [chirp]:
//   nx, dx, dt, nsteps            grid & time (units: Ω_e0 = c = 1)
//   wpe                           ω_pe/Ω_e0 (cold, equator)
//   a                             B0(h) = 1 + a (h-hc)²
//   nh, uth_para, uth_perp, ppc   hot population (equator values)
//   periodic, nd, numax           boundary system
//   seed_amp, seed                initialization
//   probes = h1 h2 ...            probe offsets from the equator (c/Ω_e0)
//   probe_stride, frame_stride, frame_decim, energy_stride, nphase, phase_decim
//   prefix

#include "pic/hybrid1d.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace {

std::string trim(std::string s) {
    const char* ws = " \t\r\n";
    const auto a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

arc::Chirp1DParams load_chirp_deck(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("chirp1d: cannot open deck " + path);
    arc::Chirp1DParams P;
    std::string line, section;
    while (std::getline(f, line)) {
        for (char c : {'#', '!'}) {
            const auto p = line.find(c);
            if (p != std::string::npos) line = line.substr(0, p);
        }
        line = trim(line);
        if (line.empty()) continue;
        if (line.front() == '[') {
            section = trim(line.substr(1, line.find(']') - 1));
            continue;
        }
        if (section != "chirp") continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        auto dv = [&] { return std::stod(val); };
        auto iv = [&] { return std::stol(val); };
        if      (key == "nx")            P.nx = (int)iv();
        else if (key == "dx")            P.dx = dv();
        else if (key == "dt")            P.dt = dv();
        else if (key == "nsteps")        P.nsteps = iv();
        else if (key == "wpe")           P.wpe = dv();
        else if (key == "a")             P.a = dv();
        else if (key == "nh")            P.nh = dv();
        else if (key == "uth_para")      P.uth_para = dv();
        else if (key == "uth_perp")      P.uth_perp = dv();
        else if (key == "ppc")           P.ppc = (int)iv();
        else if (key == "rep")           P.deltaf = (val == "deltaf");
        else if (key == "ant_amp")       P.ant_amp = dv();
        else if (key == "ant_w0")        P.ant_w0 = dv();
        else if (key == "ant_ton")       P.ant_ton = dv();
        else if (key == "ant_toff")      P.ant_toff = dv();
        else if (key == "ant_trise")     P.ant_trise = dv();
        else if (key == "periodic")      P.periodic = (val == "1" || val == "true");
        else if (key == "nd")            P.nd = (int)iv();
        else if (key == "numax")         P.numax = dv();
        else if (key == "seed_amp")      P.seed_amp = dv();
        else if (key == "seed")          P.seed = (unsigned long long)iv();
        else if (key == "probes") {
            std::istringstream is(val);
            double x;
            while (is >> x) P.probes.push_back(x);
        }
        else if (key == "probe_stride")  P.probe_stride = (int)iv();
        else if (key == "frame_stride")  P.frame_stride = (int)iv();
        else if (key == "frame_decim")   P.frame_decim = (int)iv();
        else if (key == "energy_stride") P.energy_stride = (int)iv();
        else if (key == "nphase")        P.nphase = (int)iv();
        else if (key == "phase_decim")   P.phase_decim = (int)iv();
        else if (key == "prefix")        P.prefix = val;
    }
    return P;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: chirp1d <deck.ini> [outdir] [--ppc=N] [--nsteps=N] [--nh=X] [--seed=N]\n");
        return 1;
    }
    try {
        arc::Chirp1DParams P = load_chirp_deck(argv[1]);
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if      (arg.rfind("--ppc=", 0) == 0)    P.ppc = std::stoi(arg.substr(6));
            else if (arg.rfind("--nsteps=", 0) == 0) P.nsteps = std::stol(arg.substr(9));
            else if (arg.rfind("--nh=", 0) == 0)     P.nh = std::stod(arg.substr(5));
            else if (arg.rfind("--seed=", 0) == 0)   P.seed = std::stoull(arg.substr(7));
            else if (arg == "--deltaf")              P.deltaf = true;
            else if (arg.rfind("--", 0) != 0)        P.outdir = arg;
        }
        ::mkdir(P.outdir.c_str(), 0755);
        if (P.prefix == "chirp1d") {           // default prefix = deck stem
            std::string stem = argv[1];
            const auto sl = stem.find_last_of('/');
            if (sl != std::string::npos) stem = stem.substr(sl + 1);
            const auto dot = stem.find_last_of('.');
            if (dot != std::string::npos) stem = stem.substr(0, dot);
            P.prefix = stem;
        }

        std::printf("chirp1d — 1D electron-hybrid (units: Omega_e0 = c = 1)\n");
        std::printf("  grid: nx=%d dx=%.4g L=%.1f  dt=%.4g nsteps=%ld (T=%.0f)\n",
                    P.nx, P.dx, P.nx * P.dx, P.dt, P.nsteps, P.nsteps * P.dt);
        std::printf("  wpe/wce=%.2f  a=%.3g  (B0 ends=%.2f)  nh/nc=%.4g\n",
                    P.wpe, P.a, 1.0 + P.a * 0.25 * P.nx * P.dx * P.nx * P.dx,
                    P.nh);
        std::printf("  hot: uth_para=%.3f uth_perp=%.3f (A=%.2f)  ppc=%d  rep=%s\n",
                    P.uth_para, P.uth_perp,
                    (P.uth_perp * P.uth_perp) / (P.uth_para * P.uth_para) - 1.0,
                    P.ppc, P.deltaf ? "deltaf" : "fullf");
        std::printf("  boundary: %s  nd=%d numax=%.2f\n",
                    P.periodic ? "periodic" : "damped+reflect", P.nd, P.numax);

        arc::Hybrid1D sim(P);
        std::printf("  hot markers: %ld (%.2f GB total device est.)\n",
                    sim.np(), sim.np() * 20e-9);
        std::fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        sim.run();
        CUDA_CHECK(cudaDeviceSynchronize());
        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("done: %.1f s  (%.3g particle-steps/s)\n", sec,
                    (double)sim.np() * P.nsteps / sec);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chirp1d: %s\n", e.what());
        return 1;
    }
}
