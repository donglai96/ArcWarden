// ArcWarden — L-shell plan M0 gate: magnetized Boris on the 2D spectral path.
//
// test_single_particle already asserts gyro period/radius/|u| in uniform Bz.
// This gate adds the remaining magnetized-electrostatic physics on the
// Cfg::has_b0 path (uniform rp.B0, self-consistent or prescribed E):
//
//   A) E×B drift: uniform Ex, B0 = Bz. Gyro-averaged velocity over an integer
//      number of gyro periods equals v_d = E×B/B² = (0, -Ex/Bz, 0), charge-
//      independent.
//   B) mu conservation (uniform-B form): in the E×B case the drift-frame
//      perpendicular speed |u - v_d| — i.e. mu = |u-v_d|²/(2B) — must show no
//      secular drift over 50 gyro periods (Boris is a pure rotation + kicks;
//      any monotonic growth flags a scheme/wiring bug).
//   C) oblique B0 (the whistler-run configuration): B0 = B(cosθ,0,sinθ),
//      E = 0. u∥ = u·b̂ and |u| are exact invariants of the discrete rotation.
//
// Exits non-zero on failure.

#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/pusher.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;
using Cfg = arc::Cfg;   // has_b0 = true

struct OneParticle {
    DeviceArray<float> x, y, ux, uy, uz, w;
    DeviceArray<int>   cell;
    OneParticle() : x(1), y(1), ux(1), uy(1), uz(1), w(1), cell(1) {}
    ParticleViews views() { return ParticleViews{ x.view(), y.view(), ux.view(),
                                                  uy.view(), uz.view(), w.view(), cell.view(), 1 }; }
    void set(float X, float Y, float Ux, float Uy, float Uz) {
        CUDA_CHECK(cudaMemcpy(x.data(),  &X,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(y.data(),  &Y,  4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ux.data(), &Ux, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(uy.data(), &Uy, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(uz.data(), &Uz, 4, cudaMemcpyHostToDevice));
    }
    void get(float& X, float& Y, float& Ux, float& Uy, float& Uz) {
        CUDA_CHECK(cudaMemcpy(&X,  x.data(),  4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&Y,  y.data(),  4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&Ux, ux.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&Uy, uy.data(), 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&Uz, uz.data(), 4, cudaMemcpyDeviceToHost));
    }
};

int main() {
    // Big periodic box so the trajectory (incl. drift) never wraps mid-test.
    const int nx = 256, ny = 256;
    Grid g(nx, ny, 256.0, 256.0);       // dx = dy = 1
    CudaStream stream;
    bool ok = true;

    // =============== A + B: E×B drift and drift-frame mu ====================
    {
        RunParams rp; rp.qm = -1.0; rp.dt = 0.02;
        rp.B0[0] = 0.0f; rp.B0[1] = 0.0f; rp.B0[2] = 2.0f;
        const double wc = std::fabs(rp.qm) * rp.B0[2];          // 2.0
        const double E0 = 0.02;                                 // weak: r_d << box
        const double vdy = -E0 / rp.B0[2];                      // E×B/B² = -Ex/Bz ŷ

        Fields fld(g);
        std::vector<float> ex(g.real_size(), (float)E0), zero(g.real_size(), 0.0f);
        CUDA_CHECK(cudaMemcpy(fld.Ex.data(), ex.data(),  fld.Ex.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fld.Ey.data(), zero.data(), fld.Ey.bytes(), cudaMemcpyHostToDevice));
        FieldViews fv = fld.views();

        // start at rest: pure drift + gyration about the drifting center
        OneParticle p; p.set(128.0f, 128.0f, 0.0f, 0.0f, 0.0f);

        const int steps_per_period = (int)std::round(2.0 * M_PI / (wc * rp.dt)); // 157
        const int nper = 50;
        float X, Y, Ux, Uy, Uz;
        double mu0 = -1, mu_min = 1e300, mu_max = -1e300;
        for (int k = 0; k < nper; ++k) {
            for (int s = 0; s < steps_per_period; ++s)
                detail::boris_push_kernel<Cfg><<<1, 1, 0, stream>>>(p.views(), fv, g, rp);
            CUDA_CHECK(cudaPeekAtLastError());
            stream.synchronize();
            p.get(X, Y, Ux, Uy, Uz);
            const double upx = Ux, upy = Uy - vdy;               // drift frame
            const double mu = 0.5 * (upx * upx + upy * upy) / rp.B0[2];
            if (mu0 < 0) mu0 = mu;
            mu_min = std::fmin(mu_min, mu); mu_max = std::fmax(mu_max, mu);
        }
        const double T = nper * steps_per_period * rp.dt;
        const double vy_meas = (Y - 128.0) / T;                  // dy = 1
        const double vx_meas = (X - 128.0) / T;
        const double vd_err = std::fabs(vy_meas - vdy) / std::fabs(vdy);
        // drift-frame mu: relative spread around its mean (gyro-phase ripple is
        // O(wc dt); a secular trend would blow past this over 50 periods)
        const double mu_spread = (mu_max - mu_min) / (0.5 * (mu_max + mu_min));

        std::printf("A: v_drift y=%.6e (expect %.6e, err %.2e), |vx|=%.2e\n",
                    vy_meas, vdy, vd_err, std::fabs(vx_meas));
        std::printf("B: mu0=%.6e spread=%.3e over %d periods\n", mu0, mu_spread, nper);
        if (vd_err > 2e-3)                { std::printf("FAIL A: ExB drift\n"); ok = false; }
        if (std::fabs(vx_meas) > 5e-5)    { std::printf("FAIL A: spurious vx\n"); ok = false; }
        if (mu_spread > 2e-2)             { std::printf("FAIL B: mu not conserved\n"); ok = false; }
    }

    // =============== C: oblique B0 — u∥ and |u| invariants ===================
    {
        RunParams rp; rp.qm = -1.0; rp.dt = 0.02;
        const double th = 30.0 * M_PI / 180.0, B = 1.5;
        rp.B0[0] = (float)(B * std::cos(th)); rp.B0[1] = 0.0f;
        rp.B0[2] = (float)(B * std::sin(th));
        const double bx = std::cos(th), bz = std::sin(th);

        Fields fld(g); fld.zero(stream); stream.synchronize();
        FieldViews fv = fld.views();

        OneParticle p; p.set(128.0f, 128.0f, 0.3f, 0.4f, -0.2f);
        const double upar0 = 0.3 * bx + (-0.2) * bz;
        const double u20   = 0.3 * 0.3 + 0.4 * 0.4 + 0.2 * 0.2;

        float X, Y, Ux, Uy, Uz;
        double upar_dev = 0, u2_dev = 0;
        for (int k = 0; k < 40; ++k) {
            for (int s = 0; s < 250; ++s)
                detail::boris_push_kernel<Cfg><<<1, 1, 0, stream>>>(p.views(), fv, g, rp);
            CUDA_CHECK(cudaPeekAtLastError());
            stream.synchronize();
            p.get(X, Y, Ux, Uy, Uz);
            const double upar = Ux * bx + Uz * bz;
            const double u2 = (double)Ux * Ux + (double)Uy * Uy + (double)Uz * Uz;
            upar_dev = std::fmax(upar_dev, std::fabs(upar - upar0));
            u2_dev   = std::fmax(u2_dev, std::fabs(u2 - u20) / u20);
        }
        std::printf("C: max |u_par - u_par0| = %.2e ; max |u^2| rel dev = %.2e\n",
                    upar_dev, u2_dev);
        if (upar_dev > 1e-4) { std::printf("FAIL C: u_par drift (oblique B)\n"); ok = false; }
        if (u2_dev > 1e-4)   { std::printf("FAIL C: |u| drift (oblique B)\n"); ok = false; }
    }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
