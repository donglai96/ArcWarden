// pic2d — deck schema + finalize gates + memory pre-flight
// (PLAN_2D_REBORN §2.6: every numerical design rule the ribbon flagship
// discovered as a run-design defect is a deck-finalize HARD GATE here;
// FAIL refuses launch before any allocation.)
//
// Schema (INI, #/! comments):
//   [domain]      lam_w (deg), shell margin, dx, dz  → box derived from the
//                 background geometry; or explicit x0/x1/z0/z1 override
//   [background]  profile = linedipole|kemirror|uniform|tilted,
//                 B0eq, L0 (= l_re), a, theta_deg
//   [time]        dt, nsteps
//   [cold]        nc, nonlinear = 0|1   (nonlinear terms = UB-cascade arm,
//                 PLAN_2D_REBORN §2.2 — linearized default)
//   [species X]   deltaf, dist, n0, uthpar, uthperp, kappa, lc_rho, taud,
//                 shell_L0, shell_dL, edge_dL, ppc
//   [boundary]    absorber_cells, runway_lam (deg) — absorber must sit
//                 OUTSIDE the runway (v3 defect 7 → gate)
//   [diag]        target_band_max (ω/Ωe, default 0.8),
//                 target_wna_deg (default 60) → resolution gate input
//
// P0 scope: parse + finalize + gates + pre-flight report. Field/particle
// construction consumes this from P1/P2 on.

#ifndef ARC_PIC2D_DECK2D_HPP
#define ARC_PIC2D_DECK2D_HPP

#include "pic2d/background2d.hpp"
#include "pic2d/particles2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace arc2d {

struct Deck2D {
    // [domain]
    double lam_w   = 50.0 * M_PI / 180.0;  // half-latitude coverage (radians)
    double active_Lmin = 0, active_Lmax = 0;  // field-update L band (0 = full)
    double x_min   = 0;                    // user inner-x cut (0 = follow lam_w);
                                           // moves the particle wall to the
                                           // latitude where the lines exit
    double lam_eff = 0;                    // effective wall latitude (derived)
    double margin  = 60.0;                 // box margin beyond shell+absorber (c/ωpe)
    double dx = 0.35, dz = 0.35;
    double x0 = 0, x1 = 0, z0 = 0, z1 = 0; // derived (or explicit override)
    int    nx = 0, nz = 0;
    // [background]
    Background2D bg;
    // [time]
    double dt = 0.05;
    long   nsteps = 0;
    // [cold]
    double nc = 1.0;
    int    cold_nonlinear = 0;
    double cspeed = 1.0;                   // c in code units (mirror2d-family convention)
    // [boundary]
    int    absorber_cells = 240;
    double runway_lam = 45.0 * M_PI / 180.0;
    // [diag] resolution contract: the UB target must hold where UB physics
    // lives (|λ| ≤ target_lam; observations put UB at ≲15–20°), the LB
    // corridor (0.55 Ωe, 30°) must hold out to the runway edge.
    double target_band_max = 0.75;
    double target_wna = 45.0 * M_PI / 180.0;
    double target_lam = 20.0 * M_PI / 180.0;
    long   snap_every = 0;                 // full-field snapshot cadence (steps)
    long   energy_every = 1000;
    long   probe_every = 4;                // probe-station sample cadence
    long   fv_every = 20000;               // f(v) + ledger dump cadence
    int    dens_init = 1;                  // dump per-species density at t=0
    double probe_L2 = 0;                   // second probe line at this L (0=off;
                                           // V4R8 lesson: modes can select
                                           // off-L0 columns)
    // [species]
    std::vector<SpeciesCfg> species;

    // finalize products
    double shell_cells_total = 0;
    struct GateResult { std::string name; bool pass; bool hard; std::string msg; };
    std::vector<GateResult> gates;
    double mem_fields_gb = 0, mem_markers_gb = 0;
    double sparse_frac = 1.0;              // pool cells / dense cells

    bool ok() const {
        for (const auto& g : gates) if (g.hard && !g.pass) return false;
        return true;
    }
};

namespace detail {

using IniMap = std::map<std::string, std::map<std::string, std::string>>;

inline std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline IniMap parse_ini(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("deck2d: cannot open " + path);
    IniMap m;
    std::string line, sec;
    while (std::getline(f, line)) {
        const auto cpos = line.find_first_of("#!");
        if (cpos != std::string::npos) line = line.substr(0, cpos);
        line = trim(line);
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            sec = trim(line.substr(1, line.size() - 2));
            m[sec];
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos || sec.empty()) continue;
        m[sec][trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return m;
}

inline double getd(const IniMap& m, const std::string& s, const std::string& k, double dflt) {
    auto si = m.find(s);
    if (si == m.end()) return dflt;
    auto ki = si->second.find(k);
    return ki == si->second.end() ? dflt : std::stod(ki->second);
}
inline std::string gets(const IniMap& m, const std::string& s, const std::string& k,
                        const std::string& dflt) {
    auto si = m.find(s);
    if (si == m.end()) return dflt;
    auto ki = si->second.find(k);
    return ki == si->second.end() ? dflt : ki->second;
}

// cold parallel whistler wavenumber at ω = f·Ωe (resolution-gate input):
// k = (ω/c)·sqrt(1 + ωpe²/(ω(Ωe−ω))), ωpe = 1.
inline double whistler_k(double f, double wce, double c) {
    const double w = f * wce;
    return (w / c) * std::sqrt(1.0 + 1.0 / (w * (wce - w)));
}

}  // namespace detail

inline Deck2D load_deck2d(const std::string& path) {
    using namespace detail;
    const IniMap m = parse_ini(path);
    Deck2D d;

    // silent-typo protection: warn on any key this parser does not consume
    // (a misspelled key would otherwise run 10 hours on a default value)
    {
        const std::map<std::string, std::vector<std::string>> known = {
            {"domain", {"lam_w_deg", "margin", "dx", "dz", "active_Lmin",
                        "active_Lmax", "x_min"}},
            {"background", {"profile", "B0eq", "L0", "a", "theta_deg"}},
            {"time", {"dt", "nsteps"}},
            {"cold", {"nc", "nonlinear", "c"}},
            {"boundary", {"absorber_cells", "runway_lam_deg"}},
            {"diag", {"target_band_max", "target_wna_deg", "target_lam_deg",
                      "snap_every", "energy_every", "probe_every", "fv_every",
                      "dens_init", "probe_L2"}},
        };
        const std::vector<std::string> sp_keys = {
            "deltaf", "rel", "dist", "n0", "uthpar", "uthperp", "kappa",
            "lc_rho", "taud", "wdnoise", "shell_L0", "shell_dL", "edge_dL",
            "ppc"};
        for (const auto& [sec, kv] : m) {
            const bool is_sp = sec.rfind("species", 0) == 0;
            const auto* keys = is_sp ? &sp_keys
                                     : (known.count(sec) ? &known.at(sec)
                                                         : nullptr);
            if (!keys) {
                std::fprintf(stderr, "deck2d: WARNING unknown section [%s]\n",
                             sec.c_str());
                continue;
            }
            for (const auto& [k, val] : kv)
                if (std::find(keys->begin(), keys->end(), k) == keys->end())
                    std::fprintf(stderr,
                                 "deck2d: WARNING unknown key '%s' in [%s] "
                                 "(typo? value ignored)\n",
                                 k.c_str(), sec.c_str());
        }
    }

    // background
    const std::string prof = gets(m, "background", "profile", "linedipole");
    if      (prof == "linedipole") d.bg.prof = int(B0Prof::linedipole);
    else if (prof == "kemirror")   d.bg.prof = int(B0Prof::kemirror);
    else if (prof == "dipole2d")   d.bg.prof = int(B0Prof::dipole2d);
    else if (prof == "tilted")     d.bg.prof = int(B0Prof::tilted);
    else if (prof == "uniform")    d.bg.prof = int(B0Prof::uniform);
    else throw std::runtime_error("deck2d: unknown profile " + prof);
    d.bg.B0eq  = getd(m, "background", "B0eq", 0.2);
    d.bg.L0    = getd(m, "background", "L0", 1330.5);
    d.bg.a     = getd(m, "background", "a", 0.0);
    d.bg.theta = getd(m, "background", "theta_deg", 0.0) * M_PI / 180.0;
    d.bg.finalize();

    d.lam_w  = getd(m, "domain", "lam_w_deg", 50.0) * M_PI / 180.0;
    d.active_Lmin = getd(m, "domain", "active_Lmin", 0.0);
    d.active_Lmax = getd(m, "domain", "active_Lmax", 0.0);
    d.x_min = getd(m, "domain", "x_min", 0.0);
    d.margin = getd(m, "domain", "margin", 60.0);
    d.dx = getd(m, "domain", "dx", 0.35);
    d.dz = getd(m, "domain", "dz", 0.35);
    d.dt = getd(m, "time", "dt", 0.05);
    d.nsteps = long(getd(m, "time", "nsteps", 0));
    d.nc = getd(m, "cold", "nc", 1.0);
    d.cold_nonlinear = int(getd(m, "cold", "nonlinear", 0));
    d.cspeed = getd(m, "cold", "c", 1.0);
    d.absorber_cells = int(getd(m, "boundary", "absorber_cells", 240));
    d.runway_lam = getd(m, "boundary", "runway_lam_deg", 45.0) * M_PI / 180.0;
    d.target_band_max = getd(m, "diag", "target_band_max", 0.75);
    d.target_wna = getd(m, "diag", "target_wna_deg", 45.0) * M_PI / 180.0;
    d.target_lam = getd(m, "diag", "target_lam_deg", 20.0) * M_PI / 180.0;
    d.snap_every = long(getd(m, "diag", "snap_every", 0));
    d.energy_every = long(getd(m, "diag", "energy_every", 1000));
    d.probe_every = long(getd(m, "diag", "probe_every", 4));
    d.fv_every = long(getd(m, "diag", "fv_every", 20000));
    d.dens_init = int(getd(m, "diag", "dens_init", 1));
    d.probe_L2 = getd(m, "diag", "probe_L2", 0.0);

    for (const auto& [sec, kv] : m) {
        if (sec.rfind("species", 0) != 0) continue;
        SpeciesCfg s;
        s.name     = sec.size() > 8 ? sec.substr(8) : "hot";
        s.deltaf   = getd(m, sec, "deltaf", 1) != 0;
        s.rel      = int(getd(m, sec, "rel", 0));
        s.dist     = int(getd(m, sec, "dist", 0));
        s.n0       = getd(m, sec, "n0", 0.01);
        s.uthpar   = getd(m, sec, "uthpar", 0.2);
        s.uthperp  = getd(m, sec, "uthperp", 0.3);
        s.kappa    = getd(m, sec, "kappa", 0.0);
        s.lc_rho   = getd(m, sec, "lc_rho", 0.0);
        s.taud     = getd(m, sec, "taud", 0.0);
        s.wdnoise  = getd(m, sec, "wdnoise", 1e-3);
        s.shell_L0 = getd(m, sec, "shell_L0", d.bg.L0);
        s.shell_dL = getd(m, sec, "shell_dL", 40.0);
        s.edge_dL  = getd(m, sec, "edge_dL", 8.0);
        s.ppc      = int(getd(m, sec, "ppc", 300));
        d.species.push_back(s);
    }
    return d;
}

// Derived geometry + gate battery + memory pre-flight (PLAN_2D_REBORN §1.3
// promoted rules). Call once; inspect d.gates / d.ok().
inline void finalize_deck2d(Deck2D& d) {
    auto gate = [&](const std::string& n, bool pass, bool hard, const std::string& msg) {
        d.gates.push_back({n, pass, hard, msg});
    };
    char buf[256];

    // ---- box from geometry (linedipole): shell ± absorber ± margin ----
    if (has_lines(d.bg)) {
        double dLmax = 0;
        for (const auto& s : d.species) dLmax = std::max(dLmax, s.shell_dL);
        const double Lin  = d.bg.L0 - 0.5 * dLmax - d.margin;
        const double Lout = d.bg.L0 + 0.5 * dLmax + d.margin;
        double xw, zw;                       // innermost point: high-λ end of Lin
        line_point_of(d.bg, Lin, d.lam_w, xw, zw);
        d.x0 = std::max(10.0 * d.dx, xw - d.margin);
        d.lam_eff = d.lam_w;
        if (d.x_min > 0) {                   // user inner cut: wall moves to the
            d.x0 = d.x_min;                  // latitude where the lines exit
            const double xwall = d.x_min + d.absorber_cells * d.dx;
            double q = xwall / d.bg.L0;      // cos^3 (dipole2d) or cos^2 (circle)
            const double c = B0Prof(d.bg.prof) == B0Prof::dipole2d
                                 ? std::cbrt(q)
                                 : std::sqrt(q);
            if (c < 1.0) d.lam_eff = std::min(d.lam_w, std::acos(c));
        }
        d.x1 = Lout + d.margin;
        {   // z-extent: max |z| reached by the outermost line INSIDE the box
            const double le = std::min(d.lam_eff, 0.9);
            double zm = 0;
            for (double l = 0; l <= le + 1e-9; l += le / 64)
                { double xx, zz; line_point_of(d.bg, Lout, l, xx, zz);
                  zm = std::max(zm, zz); }
            d.z1 = zm + d.margin + d.absorber_cells * d.dz;
            d.z0 = -d.z1;
        }
    } else if (d.x1 <= d.x0) {               // non-dipole: explicit box required
        gate("box", false, true, "explicit x0/x1/z0/z1 required for non-dipole profiles");
        return;
    }
    d.nx = int(std::ceil((d.x1 - d.x0) / d.dx));
    d.nz = int(std::ceil((d.z1 - d.z0) / d.dz));

    // ---- CFL (Yee light wave) ----
    const double dt_cfl = 0.999 * std::min(d.dx, d.dz) / (d.cspeed * std::sqrt(2.0));
    std::snprintf(buf, sizeof buf, "dt %.3f vs CFL %.3f", d.dt, dt_cfl);
    gate("cfl", d.dt < dt_cfl, true, buf);

    // ---- resolution: ≥ 8 cells per shortest target wavelength -----------
    // Two contracts (the trap that ate Lu's upper band, PLAN_2D_REBORN
    // §1.2.2): (a) UB — band_max at target WNA, evaluated at the local Ωe
    // of the UB domain |λ| ≤ target_lam; (b) LB corridor — 0.55 Ωe at 30°,
    // out to the runway edge.
    const double wce_ub = d.bg.B0eq * mirror_ratio_of(d.bg, std::min(d.target_lam, d.runway_lam));
    const double k_ub = detail::whistler_k(d.target_band_max, wce_ub, d.cspeed) /
                        std::cos(d.target_wna);
    const double cpl_ub = 2.0 * M_PI / (k_ub * std::max(d.dx, d.dz));
    std::snprintf(buf, sizeof buf,
                  "%.1f cells/λ at ω=%.2fΩe(λ=%.0f°), WNA %.0f° (k=%.2f ωpe/c)",
                  cpl_ub, d.target_band_max, d.target_lam * 180 / M_PI,
                  d.target_wna * 180 / M_PI, k_ub);
    gate("res/UB", cpl_ub >= 8.0, true, buf);

    const double wce_rw = d.bg.B0eq * mirror_ratio_of(d.bg, d.runway_lam);
    const double k_lb = detail::whistler_k(0.55, wce_rw, d.cspeed) /
                        std::cos(30.0 * M_PI / 180.0);
    const double cpl_lb = 2.0 * M_PI / (k_lb * std::max(d.dx, d.dz));
    std::snprintf(buf, sizeof buf,
                  "%.1f cells/λ at ω=0.55Ωe(runway edge), WNA 30° (k=%.2f ωpe/c)",
                  cpl_lb, k_lb);
    gate("res/LB", cpl_lb >= 8.0, true, buf);

    // ---- per-species gyro-resolution ρ⊥/dz ≥ 2 at max shell B -----------
    // HARD for full-f (the magnetized finite-grid instability is noise-
    // driven: test_mirror2d measured T⊥ ×4.6 in 2 T_b at ρ/dy = 1); WARN
    // for δf (deposited noise suppressed by ⟨wd²⟩; the V1 quiet-hold gate
    // is the empirical check). Recorded either way — Lu 2019 runs full-f
    // hot at ρ/Δ⊥ = 0.32 with no stated mitigation (§1.2 suspect list).
    for (const auto& s : d.species) {
        const double rho_min = s.uthperp / wce_rw;    // u⊥/B, worst (highest-B) point
        const bool pass = rho_min / d.dz >= 2.0;
        std::snprintf(buf, sizeof buf, "%s: ρ⊥/dz = %.2f at runway edge%s",
                      s.name.c_str(), rho_min / d.dz,
                      (!pass && s.deltaf) ? " (δf: V1 hold gate must confirm)" : "");
        gate("gyro/" + s.name, pass, !s.deltaf, buf);
    }

    // ---- cold nonlinear: parsed but NOT implemented — refuse, don't
    // silently run the linearised fluid under a nonlinear label ---------
    if (d.cold_nonlinear)
        gate("cold/nonlinear", false, true,
             "nonlinear cold fluid is unimplemented (deck asked for it)");

    // ---- non-relativistic particle CFL: the Boris tail must not skip
    // cells (rel=1 is self-limiting at c; rel=0 tails are unbounded) ----
    for (const auto& s : d.species)
        if (!s.rel) {
            const double umax = 4.0 * std::max(s.uthpar, s.uthperp);
            const double frac = umax * d.dt / std::min(d.dx, d.dz);
            std::snprintf(buf, sizeof buf,
                          "%s: 4σ tail moves %.2f cells/step (rel=0)",
                          s.name.c_str(), frac);
            gate("pcfl/" + s.name, frac < 1.0, true, buf);
        }

    for (const auto& sp : d.species)
        if (sp.dist >= 2)
            gate("dist/" + sp.name, false, true,
                 "dist>=2 (losscone/prodkappa) has no analytic dln f0 yet");

    // ---- kappa mapping not implemented for curved backgrounds -----------
    for (const auto& sp : d.species)
        if (sp.dist == 1 && has_lines(d.bg))
            gate("kappa/" + sp.name, false, true,
                 "bi-kappa (E,mu) mapping unimplemented: dist=1 requires "
                 "uniform/tilted background");

    // ---- δf discipline: τ_D forbidden by default (H1/H2 ruling) ---------
    for (const auto& s : d.species)
        if (s.deltaf && s.taud != 0.0)
            gate("taud/" + s.name, false, false,
                 "taud≠0 is the H1/H2 discriminator arm — confirm intentional");

    // ---- absorber strictly outside the runway ---------------------------
    const double absorber_arc = d.absorber_cells * std::max(d.dx, d.dz);
    const double runway_end_s = arc_s_of(d.bg, d.bg.L0, d.runway_lam);
    const double wall_s       = arc_s_of(d.bg, d.bg.L0, d.lam_eff);
    std::snprintf(buf, sizeof buf, "runway ends s=%.0f, wall s=%.0f, absorber %.0f",
                  runway_end_s, wall_s, absorber_arc);
    gate("absorber", runway_end_s + absorber_arc <= wall_s, true, buf);

    // ---- shell-compact marker budget (|∇L| = sec²λ ⇒ exact shell area) --
    double markers_total = 0;
    for (auto& s : d.species) {
        const double lw = d.lam_eff > 0 ? d.lam_eff : d.lam_w;
        const double area = s.shell_dL * s.shell_L0 *
                            (lw + std::sin(lw) * std::cos(lw));
        const double cells = area / (d.dx * d.dz);
        s.nmax = uint64_t(cells * s.ppc);
        markers_total += double(s.nmax);
        d.shell_cells_total += cells;
    }
    constexpr int nfield_arrays = 15;  // E,B,J (9) + cold vc (3) + jtmp + masks
    // sparse tile pool: with an active band the field arrays hold only the
    // tiles within band ± BAND_MARGIN — count them exactly as build_tiles will
    size_t field_cells = size_t(d.nx) * d.nz;
    if (d.active_Lmax > d.active_Lmin && d.active_Lmin > 0 && has_lines(d.bg)) {
        const int ntx = (d.nx + 15) / 16, ntz = (d.nz + 15) / 16;
        size_t nact = 0;
        for (int tk = 0; tk < ntz; ++tk)
            for (int ti = 0; ti < ntx; ++ti) {
                bool act = false;
                for (int c = 0; c < 4 && !act; ++c) {
                    const double xx = d.x0 + (ti + (c & 1)) * 16 * d.dx;
                    const double zz = d.z0 + (tk + (c >> 1)) * 16 * d.dz;
                    const double L = lshell_of<double>(d.bg, xx, zz);
                    act = L >= d.active_Lmin - BAND_MARGIN &&
                          L <= d.active_Lmax + BAND_MARGIN;
                }
                nact += act;
            }
        field_cells = nact * 256;
        d.sparse_frac = double(field_cells) / (double(d.nx) * d.nz);
    }
    d.mem_fields_gb  = double(field_cells) * 4.0 * nfield_arrays / 1e9;
    d.mem_markers_gb = MarkerStore::bytes_for(uint64_t(markers_total)) / 1e9;
}

inline void print_deck2d_report(const Deck2D& d, std::FILE* out = stdout) {
    std::fprintf(out, "pic2d deck report\n");
    std::fprintf(out, "  geometry : L0 = %.1f (l_re), wall λ = %.1f° (deck λ_w %.1f°), mirror ratio %.2f\n",
                 d.bg.L0, d.lam_eff * 180 / M_PI, d.lam_w * 180 / M_PI,
                 mirror_ratio_of(d.bg, d.lam_eff));
    std::fprintf(out, "  box      : x [%.0f, %.0f]  z [%.0f, %.0f]  (%d × %d cells, dx %.2f dz %.2f)\n",
                 d.x0, d.x1, d.z0, d.z1, d.nx, d.nz, d.dx, d.dz);
    std::fprintf(out, "  plasma   : ωpe/Ωe(eq) = %.2f, c = %.1f, cold nc = %.3f (%s fluid)\n",
                 1.0 / d.bg.B0eq, d.cspeed, d.nc,
                 d.cold_nonlinear ? "NONLINEAR" : "linearized");
    for (const auto& s : d.species)
        std::fprintf(out, "  species  : %-10s %s dist=%d n0=%.4f shell L0=%.0f ΔL=%.0f ppc=%d → %.2e markers\n",
                     s.name.c_str(), s.deltaf ? "δf   " : "full-f", s.dist, s.n0,
                     s.shell_L0, s.shell_dL, s.ppc, double(s.nmax));
    std::fprintf(out, "  memory   : fields %.2f GB%s + markers %.2f GB (36 B/marker eff) = %.2f GB\n",
                 d.mem_fields_gb,
                 d.sparse_frac < 1.0
                     ? (" (sparse pool, " +
                        std::to_string(int(100 * d.sparse_frac + 0.5)) +
                        "% of dense)").c_str()
                     : "",
                 d.mem_markers_gb, d.mem_fields_gb + d.mem_markers_gb);
    std::fprintf(out, "  gates    :\n");
    for (const auto& g : d.gates)
        std::fprintf(out, "    [%s] %-16s %s\n",
                     g.pass ? "PASS" : (g.hard ? "FAIL" : "WARN"),
                     g.name.c_str(), g.msg.c_str());
}

}  // namespace arc2d

#endif  // ARC_PIC2D_DECK2D_HPP
