// ArcWarden — minimal text input deck (OSIRIS/UPIC-inspired), so physics setups
// are data, not recompiled C++. INI-style sections + `key = value`; repeated
// `[species <name>]` blocks build the species list. Comments start with # or !.
//
// Example:
//   [grid]    nx = 512   ny = 8   Lx = 50.265   Ly = 1.0
//   [time]    dt = 0.05  nsteps = 4000  dump_every = 40
//   [plasma]  qm = -1.0  noisy = true   seed = 20260627
//   [species bulk]  density = 0.92  ppc = 1024  uth = 0.2 0.2 0.2  ufl = 0 0 0
//   [species beam]  density = 0.08  ppc = 1024  uth = 0.15 .15 .15  ufl = 1 0 0
//
// Two-stream / bump-on-tail are just different species lists — no code change.

#ifndef ARC_PIC_DECK_HPP
#define ARC_PIC_DECK_HPP

#include "pic/config.hpp"
#include "pic/species.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace arc {

struct Deck {
    int    nx = 0, ny = 0;
    double Lx = 0.0, Ly = 0.0;
    RunParams   rp;
    SpeciesList species;
    long        dump_every = 0;          // runner frame cadence (0 = no frames)
    std::string outdir     = "deck_frames";

    // ---- Darwin EM / whistler-pump extensions (physical/paper units) ----
    // Raw inputs are captured during the parse and turned into code-unit RunParams
    // fields in the finalize block (needs dx = Lx/nx, so it runs after [grid]).
    bool   darwin    = false;            // [field] model = darwin
    double dx_wpe_c  = 0.0;              // [field] Δx·ω_pe/c  → rp.c = dx / dx_wpe_c
    double c_direct  = 0.0;              // [field] c = <val>  (overrides dx_wpe_c)
    double wce       = 0.0;              // [background] ω_ce
    double theta_deg = 0.0;             // [background] B0 angle in x-z plane (deg)
    bool   b0_direct = false;           // [background] B0 = bx by bz given verbatim

    bool   pump_enable = false;         // [pump] enable
    int    pump_M      = 0;             // [pump] mode  → rp.pump_k0 = 2πM/Lx
    double pump_w0     = 0.0;           // [pump] w0
    double pump_ex0    = 1.39;          // [pump] Table-I amplitudes (ey0/ez0 shared)
    double pump_ey0    = 1.81;
    double pump_ez0    = -1.81;
    double pump_amp    = 1.0;           // [pump] overall scale on top of Table I
    double pump_trmp   = 0.0;           // [pump] ramp duration
    double pump_toff   = 0.0;           // [pump] turn-off time

    // ---- whistler-tool diagnostics ----
    double tsnap     = 0.0;             // [diagnostics] paper snapshot time
    int    band_lo   = 0, band_hi = 0;  // [diagnostics] nonlinear-structure mode band
    int    kt_stride = 5;               // [diagnostics] δE_L dump cadence (steps)
    int    n_frames  = 150;            // [diagnostics] video frame count
    std::string prefix;                 // [diagnostics] output-file prefix (default: deck stem)
};

namespace detail {

inline std::string deck_trim(std::string s) {
    const char* ws = " \t\r\n";
    const auto a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

inline bool deck_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "yes" || v == "on" || v == ".true.";
}

inline std::array<double, 3> deck_vec3(const std::string& v) {
    std::istringstream is(v);
    std::array<double, 3> out{0, 0, 0};
    is >> out[0] >> out[1] >> out[2];     // missing entries stay 0
    return out;
}

} // namespace detail

inline Deck load_deck(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("deck: cannot open " + path);

    Deck d;
    d.rp.eps0 = 1.0; d.rp.n0 = 1.0; d.rp.qm = -1.0; d.rp.wpe = 1.0;

    std::string line, section;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        // strip inline comments
        for (char c : {'#', '!'}) {
            const auto p = line.find(c);
            if (p != std::string::npos) line = line.substr(0, p);
        }
        line = detail::deck_trim(line);
        if (line.empty()) continue;

        if (line.front() == '[') {
            const auto e = line.find(']');
            if (e == std::string::npos)
                throw std::runtime_error("deck: bad section at line " + std::to_string(line_no));
            section = detail::deck_trim(line.substr(1, e - 1));
            if (section.rfind("species", 0) == 0) {       // "species <name>"
                Species sp;
                const std::string nm = detail::deck_trim(section.substr(7));
                if (!nm.empty()) sp.name = nm;
                d.species.push_back(sp);
            }
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("deck: expected key=value at line " + std::to_string(line_no));
        const std::string key = detail::deck_trim(line.substr(0, eq));
        const std::string val = detail::deck_trim(line.substr(eq + 1));

        auto dv = [&] { return std::stod(val); };
        auto iv = [&] { return std::stol(val); };

        if (section == "grid") {
            if      (key == "nx") d.nx = static_cast<int>(iv());
            else if (key == "ny") d.ny = static_cast<int>(iv());
            else if (key == "Lx") d.Lx = dv();
            else if (key == "Ly") d.Ly = dv();
        } else if (section == "time") {
            if      (key == "dt")         d.rp.dt = dv();
            else if (key == "nsteps")     d.rp.nsteps = iv();
            else if (key == "dump_every") d.dump_every = iv();
        } else if (section == "plasma") {
            if      (key == "qm")    d.rp.qm = dv();
            else if (key == "eps0")  d.rp.eps0 = dv();
            else if (key == "wpe")   d.rp.wpe = dv();
            else if (key == "noisy") d.rp.noisy_load = detail::deck_bool(val);
            else if (key == "seed")  d.rp.rng_seed = static_cast<unsigned long>(iv());
            else if (key == "outdir") d.outdir = val;
        } else if (section.rfind("species", 0) == 0) {
            if (d.species.empty()) throw std::runtime_error("deck: species key outside a species block");
            Species& sp = d.species.back();
            if      (key == "density") sp.density = dv();
            else if (key == "ppc")     sp.ppc = static_cast<int>(iv());
            else if (key == "uth") { auto v = detail::deck_vec3(val); for (int i=0;i<3;++i) sp.uth[i]=v[i]; }
            else if (key == "ufl") { auto v = detail::deck_vec3(val); for (int i=0;i<3;++i) sp.ufl[i]=v[i]; }
        } else if (section == "field") {
            if      (key == "model")    d.darwin = (val == "darwin");
            else if (key == "dx_wpe_c") d.dx_wpe_c = dv();
            else if (key == "c")        d.c_direct = dv();
            else if (key == "ndc")      d.rp.ndc = static_cast<int>(iv());
        } else if (section == "background") {
            if      (key == "wce")       { d.wce = dv(); d.rp.wce = d.wce; }
            else if (key == "theta_deg") d.theta_deg = dv();
            else if (key == "B0") { auto v = detail::deck_vec3(val);
                                    for (int i=0;i<3;++i) d.rp.B0[i]=(float)v[i]; d.b0_direct = true; }
        } else if (section == "pump") {
            if      (key == "enable") d.pump_enable = detail::deck_bool(val);
            else if (key == "mode")   d.pump_M = static_cast<int>(iv());
            else if (key == "w0")     d.pump_w0 = dv();
            else if (key == "ex0")    d.pump_ex0 = dv();
            else if (key == "ey0")    d.pump_ey0 = dv();
            else if (key == "ez0")    d.pump_ez0 = dv();
            else if (key == "amp")    d.pump_amp = dv();
            else if (key == "trmp")   d.pump_trmp = dv();
            else if (key == "toff")   d.pump_toff = dv();
        } else if (section == "diagnostics") {
            if      (key == "tsnap")     d.tsnap = dv();
            else if (key == "band_lo")   d.band_lo = static_cast<int>(iv());
            else if (key == "band_hi")   d.band_hi = static_cast<int>(iv());
            else if (key == "kt_stride") d.kt_stride = static_cast<int>(iv());
            else if (key == "n_frames")  d.n_frames = static_cast<int>(iv());
            else if (key == "prefix")    d.prefix = val;
        }
        // unknown sections/keys are ignored (forward-compatible)
    }

    if (d.nx <= 0 || d.ny <= 0 || d.Lx <= 0.0 || d.Ly <= 0.0)
        throw std::runtime_error("deck: [grid] nx/ny/Lx/Ly must all be set and positive");
    if (d.species.empty())
        throw std::runtime_error("deck: at least one [species ...] block is required");

    // ---- finalize: turn physical/paper inputs into code-unit RunParams (needs dx) ----
    const double dx = d.Lx / d.nx;
    if (d.c_direct > 0.0)        d.rp.c = d.c_direct;         // direct c wins
    else if (d.dx_wpe_c > 0.0)   d.rp.c = dx / d.dx_wpe_c;    // c from Δx·ω_pe/c
    if (!d.b0_direct && d.wce != 0.0) {                        // B0 at θ in the x-z plane
        const double th = d.theta_deg * 3.14159265358979323846 / 180.0;
        d.rp.B0[0] = (float)(d.wce * std::cos(th));
        d.rp.B0[1] = 0.0f;
        d.rp.B0[2] = (float)(d.wce * std::sin(th));
    }
    if (d.pump_enable) {                                       // whistler pump (An et al. Table I)
        const double s = d.pump_amp * dx / 1e4;               // Ẽα0 = 1e4·eEα0/(me ωpe² Δx)
        d.rp.pump    = true;
        d.rp.pump_k0 = 2.0 * 3.14159265358979323846 * d.pump_M / d.Lx;
        d.rp.pump_w0 = d.pump_w0;
        d.rp.pump_ex = d.pump_ex0 * s;
        d.rp.pump_ey = d.pump_ey0 * s;
        d.rp.pump_ez = d.pump_ez0 * s;
        d.rp.pump_trmp = d.pump_trmp;
        d.rp.pump_toff = d.pump_toff;
    }
    return d;
}

} // namespace arc

#endif // ARC_PIC_DECK_HPP
