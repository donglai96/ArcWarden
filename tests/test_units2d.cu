// Host-only unit bundle (2026-08-18 code review): fast checks that need no
// GPU — deck round-trips of every shipped pic2d deck (schema regressions
// and silent-typo protection), shell profile derivative consistency,
// arc-length exactness, latitude-region binning.

#include "pic2d/deck2d.hpp"
#include "pic2d/diag2d.hpp"
#include "pic2d/kinetic2d.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-30s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

int main() {
    std::printf("test_units2d — host-only unit bundle\n");

    // 1) shipped decks parse + finalize with the expected verdicts
    struct DK { const char* path; bool expect_ok; };
    const std::vector<DK> decks = {
        {"decks/warden2d_w10.ini", true},
        {"decks/warden2d_v4_chirp.ini", true},
        {"decks/warden2d_v4a_fullf.ini", true},
        {"decks/warden2d_v4r.ini", true},
        {"decks/warden2d_v4r2.ini", true},
        {"decks/warden2d_v4r3.ini", true},
        {"decks/warden2d_v4r4.ini", true},
        {"decks/warden2d_v4r5_a5.ini", true},
    };
    int bad = 0;
    for (const auto& dk : decks) {
        try {
            Deck2D d = load_deck2d(dk.path);
            finalize_deck2d(d);
            if (d.ok() != dk.expect_ok) {
                std::printf("    verdict mismatch: %s\n", dk.path);
                ++bad;
            }
        } catch (const std::exception& e) {
            std::printf("    parse failure: %s (%s)\n", dk.path, e.what());
            ++bad;
        }
    }
    gate("shipped decks round-trip", bad == 0, bad, 0.5);

    // 2) shell profile: numeric d(ln prof)/dL vs analytic (flat + edge)
    KineticCfg c;
    c.L0 = 200.f; c.dL = 20.f; c.edge = 5.f;
    double worst = 0;
    for (double L = 186.0; L < 224.0; L += 0.37) {
        const double h = 0.01;   // float-precision function: h must sit
                                 // well above the ~2e-5 ulp at L ~ 200
        const double p0 = k2d::shell_prof(float(L - h), c);
        const double p1 = k2d::shell_prof(float(L + h), c);
        if (p0 < 1e-6 || p1 < 1e-6) continue;
        const double num = (std::log(p1) - std::log(p0)) / (2 * h);
        const double ana = k2d::shell_dlnprof(float(L), c);
        worst = std::max(worst, std::fabs(num - ana));
    }
    gate("shell dlnprof vs numeric", worst < 2e-2, worst, 2e-2);

    // 3) arc_s_of: exact s = L·λ for the circle line
    Background2D bg;
    bg.prof = int(B0Prof::linedipole);
    bg.B0eq = 0.2; bg.L0 = 300.0; bg.finalize();
    double aworst = 0;
    for (double lam = 0.1; lam < 1.0; lam += 0.17)
        aworst = std::max(aworst,
                          std::fabs(arc_s_of(bg, 300.0, lam) / (300.0 * lam) - 1.0));
    gate("arc_s (circle) = L*lam", aworst < 1e-10, aworst, 1e-10);

    // 4) dipole2d arc length vs independent fine quadrature
    bg.prof = int(B0Prof::dipole2d);
    const double lamt = 0.8;
    double ref = 0;
    const int N = 400000;
    for (int i = 0; i < N; ++i) {
        const double u = lamt * (i + 0.5) / N;
        const double cc = std::cos(u);
        ref += lamt / N * 300.0 * cc * std::sqrt(1 + 3 * (1 - cc * cc));
    }
    const double got = arc_s_of(bg, 300.0, lamt);
    gate("arc_s (dipole2d) quadrature", std::fabs(got / ref - 1.0) < 1e-6,
         std::fabs(got / ref - 1.0), 1e-6);

    // 5) latitude-region binning: boundaries land in the right bins
    struct LR { float deg; int reg; };
    const std::vector<LR> lr = {{-34.9f, 0}, {-20.1f, 0}, {-19.9f, 1},
                                {-0.1f, 3},  {0.1f, 4},   {19.9f, 6},
                                {20.1f, 7},  {34.9f, 7},  {40.f, 7},
                                {-40.f, 0}};
    int rb = 0;
    for (const auto& t : lr)
        if (d2d::lam_region(t.deg) != t.reg) ++rb;
    gate("lam_region boundaries", rb == 0, rb, 0.5);

    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
