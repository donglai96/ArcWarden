// V0 orbit gate (PLAN_2D_REBORN §5) — the linedipole background is correct
// physics before anything else is built on it.
//
// Part A (field identities, double FD at random shell points):
//   ∇·B0 = 0 and ∇×B0 = 0 (vacuum field), |B| = M/r² exact, b̂·∇L = 0 exact.
//   kemirror: ∇·B0 = 0 and curl_y = −2aB0x̃ (the known coil current).
// Part B (relativistic Boris bounce orbit, double, host — tests the MATH of
// background2d.hpp; the device Boris is validated elsewhere):
//   |u| exact to fp (B does no work);
//   instantaneous μ = u⊥²/(2B) spread < 3% — this is the PHYSICAL first-order
//     ripple (grad-B ~ ρ dlnB/ds ≈ 0.3% plus curvature κρ with the circle's
//     κ = 2/L, O(1) Northrop coefficients → measured ≈ 2%); the secular gate
//     below is the actual invariance statement;
//   secular gyro-averaged μ drift < 0.3% over ≥ 5 bounces;
//   turning point s_m vs L_gc·acos(sin α_eq) < 1%   [B/B_eq = sec²(s/L)];
//   bounce period vs quadrature 4∫ds/v∥ < 1%.
// Sizing: c = 1, B0eq = 0.2 (ωpe/Ωe = 5), L = 300, |u| = 0.18c, α_eq = 55°
// → mirror at λ ≈ 35°, ρ/L ≈ 0.25%, Ω dt = 0.05 at dt = 0.25.

#include "pic2d/background2d.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-28s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

// ---- Part A ---------------------------------------------------------------
static void field_identities() {
    std::printf("Part A: field identities\n");
    Background2D bg;
    bg.prof = int(B0Prof::linedipole);
    bg.B0eq = 0.2; bg.L0 = 300.0; bg.finalize();

    std::mt19937 rng(20260817);
    std::uniform_real_distribution<double> ulam(-0.9, 0.9), uL(200.0, 400.0);
    double div_max = 0, curl_max = 0, babs_max = 0, bdotL_max = 0;
    for (int i = 0; i < 200; ++i) {
        double x, z;
        line_point(uL(rng), ulam(rng), x, z);
        const double h = 1e-5 * std::sqrt(x * x + z * z);
        auto Bx = [&](double X, double Z) { return b0_field<double>(bg, X, Z).x; };
        auto Bz = [&](double X, double Z) { return b0_field<double>(bg, X, Z).z; };
        const double B = b0_abs<double>(bg, x, z);
        const double div = (Bx(x + h, z) - Bx(x - h, z)) / (2 * h) +
                           (Bz(x, z + h) - Bz(x, z - h)) / (2 * h);
        const double curl = (Bx(x, z + h) - Bx(x, z - h)) / (2 * h) -
                            (Bz(x + h, z) - Bz(x - h, z)) / (2 * h);
        const double r2 = x * x + z * z;
        const double babs_err = std::fabs(B - bg.M / r2) / B;
        const Vec2<double> bh = b0_bhat<double>(bg, x, z);
        const Vec2<double> gL = grad_lshell<double>(x, z);
        const double gLn = std::sqrt(gL.x * gL.x + gL.z * gL.z);
        div_max   = std::max(div_max, std::fabs(div) * std::sqrt(r2) / B);
        curl_max  = std::max(curl_max, std::fabs(curl) * std::sqrt(r2) / B);
        babs_max  = std::max(babs_max, babs_err);
        bdotL_max = std::max(bdotL_max, std::fabs(bh.x * gL.x + bh.z * gL.z) / gLn);
    }
    gate("linedipole div B", div_max < 1e-6, div_max, 1e-6);
    gate("linedipole curl B", curl_max < 1e-6, curl_max, 1e-6);
    gate("|B| = M/r^2", babs_max < 1e-12, babs_max, 1e-12);
    gate("bhat . grad L = 0", bdotL_max < 1e-12, bdotL_max, 1e-12);

    // kemirror: solenoidal, curl_y = −2 a B0 x̃ (coil current, known)
    Background2D km;
    km.prof = int(B0Prof::kemirror);
    km.B0eq = 0.2; km.a = 1e-4; km.finalize();
    double kdiv = 0, kcurl_err = 0;
    std::uniform_real_distribution<double> ux(-80.0, 80.0), uz(-120.0, 120.0);
    for (int i = 0; i < 200; ++i) {
        const double x = ux(rng), z = uz(rng), h = 1e-4;
        auto Bx = [&](double X, double Z) { return b0_field<double>(km, X, Z).x; };
        auto Bz = [&](double X, double Z) { return b0_field<double>(km, X, Z).z; };
        const double div = (Bx(x + h, z) - Bx(x - h, z)) / (2 * h) +
                           (Bz(x, z + h) - Bz(x, z - h)) / (2 * h);
        const double curl = (Bx(x, z + h) - Bx(x, z - h)) / (2 * h) -
                            (Bz(x + h, z) - Bz(x - h, z)) / (2 * h);
        kdiv = std::max(kdiv, std::fabs(div) / km.B0eq);
        kcurl_err = std::max(kcurl_err,
                             std::fabs(curl - (-2.0 * km.a * km.B0eq * x)) / km.B0eq);
    }
    gate("kemirror div B", kdiv < 1e-9, kdiv, 1e-9);
    gate("kemirror curl = coil", kcurl_err < 1e-9, kcurl_err, 1e-9);
}

// ---- Part B ---------------------------------------------------------------
struct State { double x, z, ux, uy, uz; };

static void boris_step(const Background2D& bg, State& s, double dt, double qm) {
    const Vec2<double> B = b0_field<double>(bg, s.x, s.z);
    const double gam = std::sqrt(1.0 + s.ux * s.ux + s.uy * s.uy + s.uz * s.uz);
    const double f = qm * dt / (2.0 * gam);
    const double tx = f * B.x, tz = f * B.z;              // t_y = 0 (B_y = 0)
    const double t2 = tx * tx + tz * tz;
    const double sx = 2.0 * tx / (1.0 + t2), sz = 2.0 * tz / (1.0 + t2);
    // u' = u + u×t ; u×(tx,0,tz) = (uy·tz, uz·tx − ux·tz, −uy·tx)
    const double px = s.ux + s.uy * tz;
    const double py = s.uy + s.uz * tx - s.ux * tz;
    const double pz = s.uz - s.uy * tx;
    // u+ = u + u'×s
    s.ux += py * sz;
    s.uy += pz * sx - px * sz;
    s.uz += -py * sx;
    const double g2 = std::sqrt(1.0 + s.ux * s.ux + s.uy * s.uy + s.uz * s.uz);
    s.x += s.ux / g2 * dt;
    s.z += s.uz / g2 * dt;
}

static double mu_of(const Background2D& bg, const State& s) {
    const Vec2<double> bh = b0_bhat<double>(bg, s.x, s.z);
    const double upar = s.ux * bh.x + s.uz * bh.z;
    const double u2 = s.ux * s.ux + s.uy * s.uy + s.uz * s.uz;
    return (u2 - upar * upar) / (2.0 * b0_abs<double>(bg, s.x, s.z));
}

static void bounce_orbit() {
    std::printf("Part B: bounce orbit (rel-Boris, double)\n");
    Background2D bg;
    bg.prof = int(B0Prof::linedipole);
    bg.B0eq = 0.2; bg.L0 = 300.0; bg.finalize();

    const double alpha = 55.0 * M_PI / 180.0, umag = 0.18, qm = -1.0, dt = 0.25;
    State s{ bg.L0, 0.0, umag * std::sin(alpha), 0.0, umag * std::cos(alpha) };

    // gyro-average the first period → μ̄₀, guiding-center L and B
    const double Tg = 2.0 * M_PI * std::sqrt(1.0 + umag * umag) / bg.B0eq;
    const int nga = int(std::round(Tg / dt));
    double mu0 = 0, xgc = 0, zgc = 0;
    State s0 = s;
    for (int i = 0; i < nga; ++i) {
        mu0 += mu_of(bg, s0); xgc += s0.x; zgc += s0.z;
        boris_step(bg, s0, dt, qm);
    }
    mu0 /= nga; xgc /= nga; zgc /= nga;
    const double Lgc = lshell(xgc, zgc);
    const double Bgc = b0_abs<double>(bg, xgc, zgc);
    const double sin2a = 2.0 * mu0 * Bgc / (umag * umag);
    const double sm_pred = Lgc * std::acos(std::sqrt(sin2a));

    // quadrature bounce period: T = 4 ∫₀^sm ds / v∥,  v∥ = v√(1−sin²α sec²(s/L))
    const double v = umag / std::sqrt(1.0 + umag * umag);
    const int nq = 200000;
    double Tq = 0;
    for (int i = 0; i < nq; ++i) {                    // midpoint in θ, s = sm sinθ
        const double th = (i + 0.5) * (M_PI / 2) / nq;
        const double sq = sm_pred * std::sin(th);
        const double c = std::cos(sq / Lgc);
        const double rad = 1.0 - sin2a / (c * c);
        Tq += sm_pred * std::cos(th) * (M_PI / 2) / nq / (v * std::sqrt(rad));
    }
    Tq *= 4.0;

    // integrate ~6 bounces, gyro-averaged μ, turning points, flip times
    const long nsteps = long(6.5 * Tq / dt);
    const double u2_0 = s.ux * s.ux + s.uy * s.uy + s.uz * s.uz;
    std::vector<double> flips, smax_list, mu_ga;
    double mu_acc = 0, mu_inst_min = 1e300, mu_inst_max = 0;
    int ga = 0;
    double smax_cur = 0, last_flip = -1e9, upar_prev = 1;
    for (long n = 0; n < nsteps; ++n) {
        const Vec2<double> bh = b0_bhat<double>(bg, s.x, s.z);
        const double upar = s.ux * bh.x + s.uz * bh.z;
        const double sarc = std::fabs(arc_s(s.x, s.z));
        smax_cur = std::max(smax_cur, sarc);
        const double mu = mu_of(bg, s);
        mu_inst_min = std::min(mu_inst_min, mu);
        mu_inst_max = std::max(mu_inst_max, mu);
        mu_acc += mu;
        if (++ga == nga) { mu_ga.push_back(mu_acc / nga); mu_acc = 0; ga = 0; }
        const double t = n * dt;
        if (upar * upar_prev < 0 && t - last_flip > 8 * Tg) {
            flips.push_back(t);
            smax_list.push_back(smax_cur);
            smax_cur = 0;
            last_flip = t;
        }
        upar_prev = upar;
        boris_step(bg, s, dt, qm);
    }
    const double u2_1 = s.ux * s.ux + s.uy * s.uy + s.uz * s.uz;

    // gates
    const double u_err = std::fabs(std::sqrt(u2_1 / u2_0) - 1.0);
    gate("|u| conservation", u_err < 1e-12, u_err, 1e-12);

    const double mu_spread = (mu_inst_max - mu_inst_min) / mu0;
    gate("mu instantaneous spread", mu_spread < 3e-2, mu_spread, 3e-2);

    double mu_late = 0;
    const int ntail = int(mu_ga.size()) / 5;
    for (int i = int(mu_ga.size()) - ntail; i < int(mu_ga.size()); ++i) mu_late += mu_ga[i];
    mu_late /= ntail;
    const double mu_sec = std::fabs(mu_late / mu0 - 1.0);
    gate("mu secular drift", mu_sec < 3e-3, mu_sec, 3e-3);

    double sm_meas = 0;
    for (size_t i = 1; i < smax_list.size(); ++i) sm_meas += smax_list[i];
    sm_meas /= double(smax_list.size() - 1);
    const double sm_err = std::fabs(sm_meas / sm_pred - 1.0);
    std::printf("    turning point: measured %.2f, predicted %.2f (lam_m %.1f deg)\n",
                sm_meas, sm_pred, sm_pred / Lgc * 180 / M_PI);
    gate("turning point vs acos(sin a)", sm_err < 1e-2, sm_err, 1e-2);

    // mean half-period over all flip intervals after the first (settling) one
    const double Tb_meas = (flips.back() - flips[1]) / double(flips.size() - 2) * 2.0;
    std::printf("    bounce period: measured %.1f, quadrature %.1f (%zu half-bounces)\n",
                Tb_meas, Tq, flips.size());
    const double Tb_err = std::fabs(Tb_meas / Tq - 1.0);
    gate("bounce period vs quadrature", Tb_err < 1e-2, Tb_err, 1e-2);
}

int main() {
    std::printf("test_dipole2d_orbit — V0 gate for pic2d/background2d.hpp\n");
    field_identities();
    bounce_orbit();
    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
