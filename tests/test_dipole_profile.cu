// M5a gate — dipole B0(s) profile (background_b0.hpp, b0_prof = 2).
//
// Gate 1 (fit): bg::fit_dipole least-squares polynomial vs the EXACT dipole
//   B(λ(s))/Beq (bisection inversion of the closed-form arc length), through
//   the float device-side evaluation path bg::b0x. rel err < 5e-4.
// Gate 2 (orbit): zero-weight particle bounce in MaxwellSimulation with the
//   dipole mirror force vs an independent host RK4 of the gyro-averaged
//   parallel motion  ds/dt = u∥/γ,  du∥/dt = −(μ/γ) dB/ds  using the EXACT
//   dipole B(s) (NOT the fit — so the gate also bounds the fit's dynamical
//   effect). Asserts (nonrel + rel): bounce period within 1%, relative μ
//   spread < 1%, |u|² drift < 1e-4.
//
// Unlike the parabolic bounce gate (exact SHM), the dipole bounce is
// anharmonic — the RK4 reference is the theory line.

#include "pic/background_b0.hpp"
#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

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

// exact-dipole db/ds (b = B/Beq), central difference
static double db_ds(double s, double lre) {
    const double h = 1e-3;
    return (bg::dipole_b_of_s(s + h, lre) - bg::dipole_b_of_s(s - h, lre)) / (2.0 * h);
}

// RK4 bounce period of the gyro-averaged parallel motion in the EXACT dipole
static double rk4_period(double upar0, double uperp0, double lre, double gam) {
    const double mu = uperp0 * uperp0 / 2.0;   // b(0) = 1
    auto acc = [&](double s) { return -(mu / gam) * db_ds(s, lre); };
    double s = 0.0, v = upar0;                 // v = u∥ (momentum)
    const double dt = 0.02;
    double t = 0.0, t_first = -1.0, t_last = -1.0;
    int ncross = 0;
    double sprev = s;
    for (long n = 0; n < 4000000 && ncross < 9; ++n) {
        const double k1s = v / gam,            k1v = acc(s);
        const double k2s = (v + 0.5 * dt * k1v) / gam, k2v = acc(s + 0.5 * dt * k1s);
        const double k3s = (v + 0.5 * dt * k2v) / gam, k3v = acc(s + 0.5 * dt * k2s);
        const double k4s = (v + dt * k3v) / gam,       k4v = acc(s + dt * k3s);
        sprev = s;
        s += dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s);
        v += dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v);
        t += dt;
        if ((sprev < 0 && s >= 0) || (sprev > 0 && s <= 0)) {
            const double f = sprev / (sprev - s);
            const double tc = t - dt + f * dt;
            if (ncross == 0) t_first = tc;
            t_last = tc;
            ++ncross;
        }
    }
    return 2.0 * (t_last - t_first) / (ncross - 1);   // full bounce period
}

static bool run_case(int rel, double upar0, double uperp0) {
    const int nx = 500, ny = 1;
    const double dx = 0.26, dt = 0.1, c = 1.0;
    const double lre = 266.0;                 // compressed field line
    Grid g(nx, ny, nx * dx, dx);
    const double xc = 0.5 * nx * dx;

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = 1.f; rp.wce = 1.0;
    rp.b0_prof = 2; rp.b0_lre = lre; rp.b0_xc = xc;
    rp.rel = rel; rp.dump_every = 0;
    bg::fit_dipole(rp, 1.01 * xc);

    // gate 1: float polynomial vs exact dipole
    double fmax_err = 0.0;
    for (int k = 0; k <= 200; ++k) {
        const double s = xc * k / 200.0;
        const double ref = bg::dipole_b_of_s(s, lre);
        const double got = (double)bg::b0x(rp, (float)(xc + s));
        fmax_err = std::max(fmax_err, std::fabs(got - ref) / ref);
    }
    std::printf("[%s] fit max rel err = %.3e\n", rel ? "rel" : "nonrel", fmax_err);
    if (fmax_err > 5e-4) { std::printf("FAIL: fit error\n"); return false; }

    const double gam = rel ? std::sqrt(1.0 + upar0 * upar0 + uperp0 * uperp0) : 1.0;
    const double T_ref = rk4_period(upar0, uperp0, lre, gam);

    MaxwellSimulation sim(g, rp);
    Particles& parts = sim.particles();
    parts.allocate_n(1);
    {
        double ux = upar0, uy = uperp0, uz = 0.0;
        rotate_host(ux, uy, uz, 1.0, 0.0, 0.0, rp.qm, -0.5 * dt, gam);
        const float xf = (float)(xc / dx), yf = 0.5f;
        const float uxf = (float)ux, uyf = (float)uy, uzf = (float)uz;
        const float wf = 0.0f;
        const int   ci = g.idx((int)xf, (int)yf);
        CUDA_CHECK(cudaMemcpy(parts.x.data(),   &xf,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.y.data(),   &yf,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.ux.data(),  &uxf, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.uy.data(),  &uyf, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.uz.data(),  &uzf, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.w.data(),   &wf,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parts.cell.data(), &ci, 4, cudaMemcpyHostToDevice));
    }

    const long nsteps = (long)(3.2 * T_ref / dt);
    std::vector<double> xs(nsteps);
    double mu_min = 1e300, mu_max = -1e300, u2_0 = 0, u2_end = 0;
    for (long n = 0; n < nsteps; ++n) {
        sim.step();
        float xf, uxf, uyf, uzf;
        CUDA_CHECK(cudaMemcpy(&xf,  parts.x.data(),  4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uxf, parts.ux.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uyf, parts.uy.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uzf, parts.uz.data(), 4, cudaMemcpyDeviceToHost));
        xs[n] = (double)xf * dx - xc;
        const double up2 = (double)uyf * uyf + (double)uzf * uzf;
        const double b0 = bg::dipole_b_of_s(xs[n], lre);
        const double mu = up2 / (2.0 * b0);
        mu_min = std::min(mu_min, mu);
        mu_max = std::max(mu_max, mu);
        const double u2 = (double)uxf * uxf + up2;
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
    const double T_meas = 2.0 * (t_last - t_first) / (ncross - 1);

    const double werr  = std::abs(T_meas - T_ref) / T_ref;
    const double muerr = (mu_max - mu_min) / (0.5 * (mu_max + mu_min));
    const double eerr  = std::abs(u2_end - u2_0) / u2_0;

    std::printf("[%s] T_meas=%.6e T_rk4=%.6e err=%.3e  mu spread=%.3e  "
                "|u|^2 drift=%.3e\n", rel ? "rel" : "nonrel",
                T_meas, T_ref, werr, muerr, eerr);
    return werr < 0.01 && muerr < 0.01 && eerr < 1e-4;
}

int main() {
    // mirror points must sit inside the ±65 half-domain (wall b = 1.294):
    // nonrel b_m = 1 + u∥²/u⊥² = 1.16 (s_m ≈ 50); rel b_m = 1.14 (s_m ≈ 47)
    bool ok = run_case(0, 0.08, 0.2);          // Newtonian
    ok = run_case(1, 0.2, 0.53) && ok;         // relativistic, Tao momenta
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
