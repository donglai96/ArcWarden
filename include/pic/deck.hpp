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
        }
        // unknown sections/keys are ignored (forward-compatible)
    }

    if (d.nx <= 0 || d.ny <= 0 || d.Lx <= 0.0 || d.Ly <= 0.0)
        throw std::runtime_error("deck: [grid] nx/ny/Lx/Ly must all be set and positive");
    if (d.species.empty())
        throw std::runtime_error("deck: at least one [species ...] block is required");
    return d;
}

} // namespace arc

#endif // ARC_PIC_DECK_HPP
