// M4 gate A (2D code path) — single-particle mirror bounce in the parabolic
// background B0(x) = B0eq(1 + a (x-xc)^2) x̂, through MaxwellSimulation with the
// b0_prof effective mirror field (background_b0.hpp). 2D analog of
// test_chirp1d_mirror: the gyro-averaged mirror force gives EXACT SHM in x,
//   du_par/dt = -(u_perp^2 / 2 B0) dB0/dx = -2 a M (x - xc),
// with M = u_perp^2 / 2 B0 invariant, so omega_b = sqrt(a) * u_perp_eq
// (the Yee push is nonrelativistic, gamma = 1).
//
// The particle carries ZERO macro weight, so it deposits nothing and the wave
// fields stay identically zero: this isolates the orbit integrator inside the
// full simulation loop. u is rotated backward dt/2 about B0 + B_mir on the
// host (leapfrog init, chirp1d k_half_back analog).
//
// Asserts: bounce frequency within 1% of theory; relative mu spread < 1%;
// |u|^2 drift < 1e-4 (mirror rotation is a pure rotation).

#include "pic/simulation_maxwell.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

// double-precision Boris rotation about B over signed interval dts (q/m = qm)
static void rotate_host(double& ux, double& uy, double& uz,
                        double Bx, double By, double Bz, double qm, double dts) {
    const double h = 0.5 * qm * dts;
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

int main() {
    const int nx = 512, ny = 1;
    const double dx = 0.2, dt = 0.1, c = 1.0;
    const double b0eq = 1.0, a = 1e-3;
    Grid g(nx, ny, nx * dx, dx);
    const double xc = 0.5 * nx * dx;

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)b0eq; rp.wce = b0eq;
    rp.b0_prof = 1; rp.b0_a = a; rp.b0_xc = xc;
    rp.dump_every = 0;

    const double upar0 = 0.05, uperp0 = 0.1;
    const double wb_theory = std::sqrt(a) * uperp0;   // gamma = 1 (nonrel push)

    MaxwellSimulation sim(g, rp);
    Particles& parts = sim.particles();
    parts.allocate_n(1);
    {
        // start at the equator (x = xc), u = (upar0, uperp0, 0) at t = 0;
        // leapfrog init: rotate backward dt/2 about B0(xc) + B_mir (db0 = 0
        // at the equator, so B_mir = 0 here — but keep the general form).
        double ux = upar0, uy = uperp0, uz = 0.0;
        const double b0 = b0eq * (1.0 + a * 0.0);
        const double mc = 0.0 / (2.0 * b0 * rp.qm);   // db0/dx = 0 at xc
        rotate_host(ux, uy, uz, b0, mc * uz, -mc * uy, rp.qm, -0.5 * dt);
        const float xf = (float)(xc / dx), yf = 0.5f;
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
        float xf, uxf, uyf, uzf;
        CUDA_CHECK(cudaMemcpy(&xf,  parts.x.data(),  4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uxf, parts.ux.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uyf, parts.uy.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&uzf, parts.uz.data(), 4, cudaMemcpyDeviceToHost));
        xs[n] = (double)xf * dx - xc;
        const double up2 = (double)uyf * uyf + (double)uzf * uzf;
        const double b0 = b0eq * (1.0 + a * xs[n] * xs[n]);
        const double mu = up2 / (2.0 * b0);
        mu_min = std::min(mu_min, mu);
        mu_max = std::max(mu_max, mu);
        const double u2 = (double)uxf * uxf + up2;
        if (n == 0) u2_0 = u2;
        u2_end = u2;
    }

    // bounce frequency from zero crossings of x - xc
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
    if (ncross < 4) { std::printf("FAIL: only %d equator crossings\n", ncross); return 1; }
    const double wb_meas = M_PI * (ncross - 1) / (t_last - t_first);

    const double werr  = std::abs(wb_meas - wb_theory) / wb_theory;
    const double muerr = (mu_max - mu_min) / (0.5 * (mu_max + mu_min));
    const double eerr  = std::abs(u2_end - u2_0) / u2_0;

    std::printf("yee mirror bounce: wb_meas=%.6e wb_theory=%.6e err=%.3e\n",
                wb_meas, wb_theory, werr);
    std::printf("mu spread=%.3e  |u|^2 drift=%.3e\n", muerr, eerr);

    const bool ok = werr < 0.01 && muerr < 0.01 && eerr < 1e-4;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
