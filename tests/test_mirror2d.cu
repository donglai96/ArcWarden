// G1.2 gates — 2D slab mirror background (b0_prof = 3, background_b0.hpp):
//   Bx = B0eq(1 + a x̃²),  By = −2 a B0eq x̃ ỹ   (∇·B = 0 by construction).
// Unlike the 1D b0_prof=1 path there is NO effective-field trick: the mirror
// force comes from the RESOLVED y-gyration sampling the real By(x,y), so
// these gates exercise exactly the physics the flagship 2D runs rely on.
//
// Gate A (single particle, zero weight — orbit integrator only): the
// gyro-averaged motion is the same SHM as the 1D parabolic bottle,
//   omega_b = sqrt(a) u_perp_eq / gamma,
// asserted to 1%; mu = u_perp^2/(2|B|) (u_perp taken w.r.t. the LOCAL tilted
// b̂) spread < 1%. The residual mu oscillation is the physical first-order
// adiabatic ripple O(2 a rho x̃_max), so cases keep that below the gate.
// |u|^2 drift < 1e-4 (no E: Boris is a pure rotation).
//
// Gate B (loaded plasma, full-f): initialize_mirror with ny > 1 must load
// n(x) ∝ Tperp(x)/Tperp_eq uniform in y (chunked moments vs analytic), and
// the plasma must sit QUIET for >= 2 thermal bounce periods: no secular
// field growth (second-half <dB2> / first-half < 2), density profile still
// on the analytic curve at the end (< 5%), INTERIOR Tperp/Tpar drift < 5%.
// Whistler-stable anisotropy Tperp < Tpar so any growth would be a
// numerical or equilibrium artifact. Uses the PRODUCTION closed-x boundary
// (bnd_x = 1; also keeps passing particles off the x-wrap where By flips
// sign) and production J smoothing (jfilter = 3).
//
// FLAGSHIP DESIGN RULES found by earlier versions of this gate (all with
// n(x) still perfectly on the analytic curve — pure velocity-space effects):
//  1. rho_thermal/dy >= ~2 is MANDATORY once the gyration is resolved on
//     the grid: at rho/dy = 1 the magnetized finite-grid instability heated
//     Tperp x4.6 in 2 bounce periods (to rho ~ 2 dy, Birdsall-Langdon
//     saturation). The 1D mirror path never sees this.
//  2. The (E,mu) load is a VACUUM-FIELD equilibrium: it omits the
//     self-consistent diamagnetic dB/B ~ -beta_perp/2. Loading the FULL
//     density as the mirror species (beta_perp = 0.32) relaxed visibly
//     (T drifts 18-35% at ppc 400, ~1/ppc shot-noise part on top).
//     Production decks must keep the mirror-loaded species a MINORITY
//     (beta_hot << 1) — the cold majority is isotropic and carries no
//     mirror mapping. This gate therefore runs at density 0.1.
//  3. The periodic y-wrap By sign flip pitch-scatters the gyro-orbits
//     that straddle it (2*rho/Ly of markers): measured Tpar -10.6% /
//     Tperp +5.7% per 2 T_b in the layer — ISOTROPIZATION at constant
//     |u|, not heating — with a -4%/+1% echo mixed into the interior by
//     passing-particle gyro-center migration. Scales as (wrap jump
//     2 a B0 x~ Ly)^2 x (layer fraction rho/Ly). If a flagship geometry
//     ever needs better, the escalation is the cos-modulated fully
//     periodic variant Bx = B0(1 + a x~^2 cos(2 pi y~/Ly)) (smooth
//     everywhere, anti-mirror strip at the wrap) or specular y-walls.

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

// double-precision Boris rotation about B over signed interval dts (q/m = qm)
static void rotate_host(double& ux, double& uy, double& uz,
                        double Bx, double By, double Bz,
                        double qm, double dts, double gam) {
    const double h = 0.5 * qm * dts / gam;
    const double tx = h * Bx, ty = h * By, tz = h * Bz;
    const double t2 = tx * tx + ty * ty + tz * tz;
    const double sf = 2.0 / (1.0 + t2);
    const double upx = ux + (uy * tz - uz * ty);
    const double upy = uy + (uz * tx - ux * tz);
    const double upz = uz + (ux * ty - uy * tx);
    ux += sf * (upy * tz - upz * ty);
    uy += sf * (upz * tx - upx * tz);
    uz += sf * (upx * ty - upy * tx);
}

static bool bounce_case(int rel, double a, double upar0, double uperp0) {
    const int nx = 512, ny = 64;
    const double dx = 0.2, dt = 0.1, c = 1.0, b0eq = 1.0;
    Grid g(nx, ny, nx * dx, ny * dx);
    const double xc = 0.5 * nx * dx, yc = 0.5 * ny * dx;

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)b0eq; rp.wce = b0eq;
    rp.b0_prof = 3; rp.b0_a = a; rp.b0_xc = xc; rp.b0_yc = yc;
    rp.rel = rel; rp.dump_every = 0;

    const double gam = rel ? std::sqrt(1.0 + upar0 * upar0 + uperp0 * uperp0) : 1.0;
    const double wb_theory = std::sqrt(a) * uperp0 / gam;

    MaxwellSimulation sim(g, rp);
    Particles& parts = sim.particles();
    parts.allocate_n(1);
    {
        // start at the axis (x=xc, y=yc) with uz = 0 so the gyro-center sits
        // AT ỹ = 0 (the y-offset of the gyro-center is ∝ uz for B ∥ x̂)
        double ux = upar0, uy = uperp0, uz = 0.0;
        rotate_host(ux, uy, uz, b0eq, 0.0, 0.0, rp.qm, -0.5 * dt, gam);
        const float xf = (float)(xc / dx), yf = (float)(yc / dx);
        const float uxf = (float)ux, uyf = (float)uy, uzf = (float)uz;
        const float wf = 0.0f;                        // zero weight: no deposit
        const int   ci = g.idx((int)xf, (int)yf);
        CUDA_CHECK(cudaMemcpy(parts.x.data(),   &xf,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.y.data(),   &yf,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.ux.data(),  &uxf, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.uy.data(),  &uyf, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.uz.data(),  &uzf, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.w.data(),   &wf,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.cell.data(), &ci, 4, cudaMemcpyHostToDevice));
    }

    const double T = 3.2 * 2.0 * M_PI / wb_theory;    // ~3 bounce periods
    const long nsteps = (long)(T / dt);

    std::vector<double> xs(nsteps);
    double mu_min = 1e300, mu_max = -1e300, u2_0 = 0, u2_end = 0;
    for (long n = 0; n < nsteps; ++n) {
        sim.step();
        float xf, yf, uxf, uyf, uzf;
        CUDA_CHECK(cudaMemcpy(&xf,  parts.x.data(),  4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&yf,  parts.y.data(),  4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uxf, parts.ux.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uyf, parts.uy.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uzf, parts.uz.data(), 4, cudaMemcpyDeviceToHost));
        const double xt = (double)xf * dx - xc, yt = (double)yf * dx - yc;
        xs[n] = xt;
        // mu w.r.t. the LOCAL field: B = (Bx, By, 0), u_perp^2 = |u|^2 - (u·b̂)^2
        const double Bx = b0eq * (1.0 + a * xt * xt);
        const double By = -2.0 * a * b0eq * xt * yt;
        const double Bm = std::sqrt(Bx * Bx + By * By);
        const double u2 = (double)uxf * uxf + (double)uyf * uyf + (double)uzf * uzf;
        const double upar = ((double)uxf * Bx + (double)uyf * By) / Bm;
        const double mu = (u2 - upar * upar) / (2.0 * Bm);
        mu_min = std::min(mu_min, mu);
        mu_max = std::max(mu_max, mu);
        if (n == 0) u2_0 = u2;
        u2_end = u2;
    }

    int ncross = 0;
    double t_first = 0, t_last = 0;
    for (long n = 1; n < nsteps; ++n) {
        if ((xs[n - 1] < 0 && xs[n] >= 0) || (xs[n - 1] > 0 && xs[n] <= 0)) {
            const double f = xs[n - 1] / (xs[n - 1] - xs[n]);
            const double tc = (n + f) * dt;
            if (ncross == 0) t_first = tc;
            t_last = tc;
            ++ncross;
        }
    }
    if (ncross < 4) { std::printf("FAIL: only %d equator crossings\n", ncross); return false; }
    const double wb_meas = M_PI * (ncross - 1) / (t_last - t_first);

    const double werr  = std::abs(wb_meas - wb_theory) / wb_theory;
    const double muerr = (mu_max - mu_min) / (0.5 * (mu_max + mu_min));
    const double eerr  = std::abs(u2_end - u2_0) / u2_0;

    std::printf("[%s a=%.1e] wb_meas=%.6e wb_theory=%.6e err=%.3e  mu spread=%.3e  "
                "|u|^2 drift=%.3e\n", rel ? "rel" : "nonrel", a,
                wb_meas, wb_theory, werr, muerr, eerr);
    return werr < 0.01 && muerr < 0.01 && eerr < 1e-4;
}

static bool quiescence_case() {
    const int nx = 1024, ny = 32, ppc = 400;
    const double dx = 0.2, dt = 0.1, c = 1.0, wce = 0.25;  // rho_perp = 2 dy
    const double upar = 0.125, uperp = 0.1;         // A = 0.64: whistler-stable
    Grid g(nx, ny, nx * dx, ny * dx);
    const double xc = 0.5 * nx * dx, yc = 0.5 * ny * dx;
    const double a = 1.0 / (xc * xc);               // wall mirror ratio 2

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.b0_prof = 3; rp.b0_a = a; rp.b0_xc = xc; rp.b0_yc = yc;
    rp.bnd_x = 1;                                   // production closed box
    rp.jfilter = 3;                                 // production J smoothing —
    // without it, residual shot-noise grid heating at ppc 400 is 7%/14%
    // (Tperp/Tpar) per 2 T_b even at beta_perp = 0.032
    rp.noisy_load = true; rp.dump_every = 0; rp.rng_seed = 20260723UL;

    // density 0.1: beta_perp = 2 n Tperp / B0^2 = 0.032, so the omitted
    // diamagnetic correction is ~1.6% — production-like (minority species)
    Species sp{"e", 0.1, ppc, {upar, uperp, uperp}, {0, 0, 0}, true};
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize_mirror(sp, g, rp, sim.stream());
    sim.stream().synchronize();
    const std::size_t n = sim.particles().n;

    const double Tpa = upar * upar, Tpe = uperp * uperp;
    const int nch = 16, cw = nx / nch;
    auto profile_err = [&](double& emax_n, double& emax_t, double& emax_p) {
        std::vector<float> x(n), uy(n), uz(n), ux(n);
        CUDA_CHECK(cudaMemcpy(x.data(),  sim.particles().x.data(),  n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), sim.particles().ux.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), sim.particles().uy.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), sim.particles().uz.data(), n * 4, cudaMemcpyDeviceToHost));
        std::vector<double> cnt(nch, 0), sup2(nch, 0), spa2(nch, 0);
        for (std::size_t p = 0; p < n; ++p) {
            const int ch = std::min((int)(x[p] / cw), nch - 1);
            cnt[ch] += 1.0;
            sup2[ch] += (double)uy[p] * uy[p] + (double)uz[p] * uz[p];
            spa2[ch] += (double)ux[p] * ux[p];
        }
        emax_n = emax_t = emax_p = 0.0;
        for (int chk = 0; chk < nch; ++chk) {
            const double xm = (chk + 0.5) * cw * dx - xc;
            const double b  = 1.0 + a * xm * xm;
            const double Tp = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);
            const double n_th = (double)ppc * cw * ny * (Tp / Tpe);
            emax_n = std::max(emax_n, std::fabs(cnt[chk] / n_th - 1.0));
            emax_t = std::max(emax_t, std::fabs(0.5 * sup2[chk] / cnt[chk] / Tp  - 1.0));
            emax_p = std::max(emax_p, std::fabs(spa2[chk] / cnt[chk] / Tpa - 1.0));
        }
    };

    // y-region moment split: markers whose gyro-orbit can straddle the
    // periodic y-wrap (within ~rho of it) see the By sign flip and pitch-
    // scatter (measured: they isotropize the Tperp<Tpar load, Tpar -10%
    // per 2 T_b, |u|-conserving — NOT heating). Gate on the INTERIOR;
    // the layer numbers are printed as the wrap-artifact record.
    struct Split { double tpa[2], tpe[2], frac; };
    auto ysplit = [&]() {
        std::vector<float> y(n), ux(n), uy(n), uz(n);
        CUDA_CHECK(cudaMemcpy(y.data(),  sim.particles().y.data(),  n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), sim.particles().ux.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), sim.particles().uy.data(), n * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), sim.particles().uz.data(), n * 4, cudaMemcpyDeviceToHost));
        const double rho_c = uperp / wce / dx;       // thermal gyroradius, cells
        double sp[2] = {0, 0}, st[2] = {0, 0}; long cnt2[2] = {0, 0};
        for (std::size_t p = 0; p < n; ++p) {
            const double dwall = std::min((double)y[p], (double)ny - y[p]);
            const int r = dwall < 2.0 * rho_c ? 0 : 1;   // 0 = straddler layer
            sp[r] += (double)ux[p] * ux[p];
            st[r] += (double)uy[p] * uy[p] + (double)uz[p] * uz[p];
            ++cnt2[r];
        }
        Split s;
        for (int r = 0; r < 2; ++r) {
            s.tpa[r] = sp[r] / cnt2[r] / Tpa;
            s.tpe[r] = 0.5 * st[r] / cnt2[r] / Tpe;
        }
        s.frac = (double)cnt2[0] / n;
        return s;
    };

    double en, et, ep;
    profile_err(en, et, ep);
    std::printf("load: n=%zu  max chunk err: n(x) %.3f%%  Tperp(x) %.2f%%  Tpar %.2f%%\n",
                n, 100 * en, 100 * et, 100 * ep);
    if (en > 0.02 || et > 0.08 || ep > 0.08) {
        std::printf("FAIL: load moments off\n"); return false;
    }
    const Split s0 = ysplit();

    // 2 thermal bounce periods, tracking <dB2> (transverse wave field)
    const double wb = std::sqrt(a) * uperp;
    const long nsteps = (long)(2.0 * 2.0 * M_PI / wb / dt);
    const int  stride = 500;
    std::vector<float> by(g.real_size()), bz(g.real_size());
    double db2_h1 = 0, db2_h2 = 0; int n1 = 0, n2 = 0;
    for (long s = 1; s <= nsteps; ++s) {
        sim.step();
        if (s % stride) continue;
        CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), by.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
        double db2 = 0.0;
        for (std::size_t i = 0; i < by.size(); ++i)
            db2 += (double)by[i] * by[i] + (double)bz[i] * bz[i];
        if (s <= nsteps / 2) { db2_h1 += db2; ++n1; } else { db2_h2 += db2; ++n2; }
    }
    db2_h1 /= n1; db2_h2 /= n2;
    const double growth = db2_h2 / db2_h1;

    profile_err(en, et, ep);
    const Split s1 = ysplit();
    const double d_tpa_int = s1.tpa[1] / s0.tpa[1] - 1.0;
    const double d_tpe_int = s1.tpe[1] / s0.tpe[1] - 1.0;
    std::printf("quiet: %ld steps (2 bounce periods), <dB2> half2/half1 = %.2f, "
                "n(x) err %.2f%%\n", nsteps, growth, 100 * en);
    std::printf("  interior (gated): dTpar %+.2f%%  dTperp %+.2f%%\n",
                100 * d_tpa_int, 100 * d_tpe_int);
    std::printf("  wrap layer (%.0f%% of markers, recorded): dTpar %+.2f%%  "
                "dTperp %+.2f%%\n", 100 * s1.frac,
                100 * (s1.tpa[0] / s0.tpa[0] - 1.0),
                100 * (s1.tpe[0] / s0.tpe[0] - 1.0));

    bool ok = true;
    if (growth > 2.0) { std::printf("FAIL: field energy grew x%.2f > 2\n", growth); ok = false; }
    if (en > 0.05)    { std::printf("FAIL: density profile drifted %.2f%% > 5%%\n", 100 * en); ok = false; }
    if (std::fabs(d_tpa_int) > 0.05) { std::printf("FAIL: interior Tpar drifted %+.2f%% (|.|>5%%)\n", 100 * d_tpa_int); ok = false; }
    if (std::fabs(d_tpe_int) > 0.05) { std::printf("FAIL: interior Tperp drifted %+.2f%% (|.|>5%%)\n", 100 * d_tpe_int); ok = false; }
    return ok;
}

int main() {
    bool ok = bounce_case(0, 1e-3,   0.05, 0.1);   // Newtonian, thermal-scale
    // rel case at smaller a: keeps the physical adiabatic ripple
    // O(2 a rho x̃_max) below the 1% mu gate at rho = u_perp = 0.53
    ok = bounce_case(1, 2.5e-4, 0.2, 0.53) && ok;
    ok = quiescence_case() && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
