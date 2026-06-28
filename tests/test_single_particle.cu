// ArcWarden — Step 11 test [S6]: single-particle Boris trajectories.
//
// Case A (B0=0, uniform E): a particle starting at rest gains velocity
//   u(t) = (q/m) E t   exactly (leapfrog adds (q/m)E dt per step). Check ux, uy.
// Case B (B0≠0, E=0): a particle gyrates at the cyclotron frequency
//   ω_c = |(q/m) Bz|, radius r = v_perp/ω_c, with |u| conserved (Boris is
//   energy-exact). Check per-step rotation = 2 atan(ω_c dt/2) (Boris-exact),
//   ω_eff ≈ ω_c, speed conserved, and gyroradius ≈ v_perp/ω_c.
//
// Exits non-zero on failure.

#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/pusher.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;
using Cfg = arc::Cfg;   // has_b0 = true (B0=0 makes the rotation the identity)

// minimal single-particle storage + one-step push
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
    const int nx = 16, ny = 16;
    Grid g(nx, ny, 16.0, 16.0);   // dx = dy = 1 (cell units == length units)
    CudaStream stream;
    bool ok = true;

    // =================== Case A: uniform E, B0 = 0 ===================
    {
        RunParams rp; rp.qm = -1.0; rp.dt = 0.01;
        rp.B0[0] = rp.B0[1] = rp.B0[2] = 0.0f;
        const float E0 = 0.5f;

        Fields fld(g);
        std::vector<float> ex(g.real_size(), E0), ey(g.real_size(), 0.0f);
        CUDA_CHECK(cudaMemcpy(fld.Ex.data(), ex.data(), fld.Ex.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fld.Ey.data(), ey.data(), fld.Ey.bytes(), cudaMemcpyHostToDevice));
        FieldViews fv = fld.views();

        OneParticle p; p.set(8.0f, 8.0f, 0.0f, 0.0f, 0.0f);
        const int N = 100;
        for (int s = 0; s < N; ++s)
            detail::boris_push_kernel<Cfg><<<1,1,0,stream>>>(p.views(), fv, g, rp);
        CUDA_CHECK(cudaPeekAtLastError());
        stream.synchronize();

        float X,Y,Ux,Uy,Uz; p.get(X,Y,Ux,Uy,Uz);
        const double ux_expect = rp.qm * E0 * (N * rp.dt);   // -0.5
        std::printf("A: ux=%.6f expect %.6f ; uy=%.2e\n", Ux, ux_expect, Uy);
        if (std::fabs(Ux - ux_expect) > 1e-5) { std::printf("FAIL A: ux\n"); ok = false; }
        if (std::fabs(Uy) > 1e-6)             { std::printf("FAIL A: uy\n"); ok = false; }
    }

    // =================== Case B: gyration, E = 0, B0 = Bz ===================
    {
        RunParams rp; rp.qm = -1.0; rp.dt = 0.01;
        rp.B0[0] = 0.0f; rp.B0[1] = 0.0f; rp.B0[2] = 1.0f;
        const double wc = std::fabs(rp.qm * rp.B0[2]);   // 1.0
        const double vperp = 1.0;

        Fields fld(g); fld.zero(stream); stream.synchronize();   // E = 0
        FieldViews fv = fld.views();

        OneParticle p; p.set(8.0f, 8.0f, float(vperp), 0.0f, 0.0f);

        const int N = static_cast<int>(std::round(2.0 * M_PI / (wc * rp.dt)));  // ~one period
        double prev_ang = 0.0, total_ang = 0.0, speed_dev = 0.0;
        double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30;
        float X,Y,Ux,Uy,Uz;
        for (int s = 0; s < N; ++s) {
            detail::boris_push_kernel<Cfg><<<1,1,0,stream>>>(p.views(), fv, g, rp);
            CUDA_CHECK(cudaPeekAtLastError());
            stream.synchronize();
            p.get(X,Y,Ux,Uy,Uz);
            const double ang = std::atan2(Uy, Ux);
            double d = ang - prev_ang;
            while (d >  M_PI) d -= 2*M_PI;     // unwrap
            while (d < -M_PI) d += 2*M_PI;
            if (s > 0) total_ang += d;
            prev_ang = ang;
            const double speed = std::sqrt(double(Ux)*Ux + double(Uy)*Uy + double(Uz)*Uz);
            speed_dev = std::fmax(speed_dev, std::fabs(speed - vperp));
            xmin = std::fmin(xmin, X); xmax = std::fmax(xmax, X);
            ymin = std::fmin(ymin, Y); ymax = std::fmax(ymax, Y);
        }
        const double ang_per_step  = std::fabs(total_ang) / (N - 1);
        const double ang_boris     = 2.0 * std::atan(wc * rp.dt / 2.0);   // Boris-exact
        const double w_eff         = ang_per_step / rp.dt;
        const double r_meas        = 0.5 * std::fmax(xmax - xmin, ymax - ymin);
        const double r_expect      = vperp / wc;   // 1.0

        std::printf("B: angle/step=%.6e (Boris-exact %.6e) ; w_eff=%.5f (wc=%.1f)\n",
                    ang_per_step, ang_boris, w_eff, wc);
        std::printf("   |u| max dev=%.2e ; gyroradius=%.4f (expect %.4f)\n",
                    speed_dev, r_meas, r_expect);
        if (std::fabs(ang_per_step - ang_boris) > 1e-6) { std::printf("FAIL B: rotation angle\n"); ok = false; }
        if (std::fabs(w_eff - wc) > 1e-3)               { std::printf("FAIL B: cyclotron freq\n"); ok = false; }
        if (speed_dev > 1e-5)                           { std::printf("FAIL B: energy not conserved\n"); ok = false; }
        if (std::fabs(r_meas - r_expect) > 2e-2)        { std::printf("FAIL B: gyroradius\n"); ok = false; }
    }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
