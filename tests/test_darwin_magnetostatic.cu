// ArcWarden — Darwin GATE 1: magnetostatic B-solve + current deposit.
//
// Part 1 (B solve, the new EM physics): set a static transverse current
//   J = (0, 0, Jz),  Jz(x) = J0 cos(k1 x),  k1 = 2π/Lx,  ρ = 0.
// Darwin/Ampère (Coulomb gauge):  ∇×B = μ₀ J,  with μ₀ = 1/(ε₀ c²).
//   (∇×B)_z = ∂x By = μ₀ Jz  ⇒  By(x) = (μ₀ J0 / k1) sin(k1 x),  Bx = Bz = 0.
// With Lx = 2π (k1=1), J0=1, ε₀=c=1 (μ₀=1):  By = sin(x).
//
// Part 2 (current deposit, locks Phase B): a uniform drifting electron beam
//   ufl=(vd,0,0) deposits a uniform current; grid-mean Jx ≈ q·n0·vd, Jy,Jz ≈ 0.
//
// Exits non-zero on failure.

#include "pic/config.hpp"
#include "pic/depositor.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/solver_darwin.hpp"
#include "pic/sources.hpp"
#include "pic/spectral.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int    nx = 64, ny = 32;
    const double TwoPi = 6.28318530717958647692;
    const double Lx = TwoPi, Ly = TwoPi;
    Grid g(nx, ny, Lx, Ly);
    const double k1 = TwoPi / Lx;           // 1.0
    const int    N  = g.real_size();

    Sources src(g); src.allocate_em(g);
    Fields  fld(g); fld.allocate_em(g);
    SpectralEngine engine(g); engine.enable_em();
    DarwinSpectralSolver solver;
    RunParams rp;                            // eps0=1, c=1 -> mu0=1
    CudaStream stream;

    bool ok = true;

    // ---- Part 1: B from a prescribed transverse current Jz = cos(k1 x) ----
    src.zero_rho_j(stream);
    std::vector<float> h_jz(N);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            h_jz[g.idx(i, j)] = static_cast<float>(std::cos(k1 * i * g.dx));
    CUDA_CHECK(cudaMemcpy(src.Jz.data(), h_jz.data(), src.Jz.bytes(), cudaMemcpyHostToDevice));
    stream.synchronize();

    solver.solve(src, fld, engine, rp, stream);
    stream.synchronize();

    std::vector<float> h_Bx(N), h_By(N), h_Bz(N);
    CUDA_CHECK(cudaMemcpy(h_Bx.data(), fld.Bx.data(), fld.Bx.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_By.data(), fld.By.data(), fld.By.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bz.data(), fld.Bz.data(), fld.Bz.bytes(), cudaMemcpyDeviceToHost));

    const double mu0 = 1.0 / (rp.eps0 * rp.c * rp.c);
    double max_by_err = 0, max_bx = 0, max_bz = 0, peak_by = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double by_an = (mu0 * 1.0 / k1) * std::sin(k1 * i * g.dx);
            const int c = g.idx(i, j);
            max_by_err = std::fmax(max_by_err, std::fabs(h_By[c] - by_an));
            max_bx     = std::fmax(max_bx, std::fabs(h_Bx[c]));
            max_bz     = std::fmax(max_bz, std::fabs(h_Bz[c]));
            peak_by    = std::fmax(peak_by, std::fabs(h_By[c]));
        }
    std::printf("Darwin B-solve: max|By-analytic|=%.3e (peak|By|=%.4f, expect 1)\n", max_by_err, peak_by);
    std::printf("                max|Bx|=%.3e  max|Bz|=%.3e  (expect 0)\n", max_bx, max_bz);
    const double tol = 1e-4;
    if (max_by_err > tol) { std::printf("FAIL: By error > %.1e\n", tol); ok = false; }
    if (max_bx > tol || max_bz > tol) { std::printf("FAIL: Bx/Bz not ~0\n"); ok = false; }

    // ---- Part 2: current deposit from a uniform drifting beam ----
    const double vd = 0.3;
    SpeciesList sp = { Species{ "beam", 1.0, 64, {0.05, 0.05, 0.05}, {vd, 0.0, 0.0} } };
    rp.noisy_load = true; rp.rng_seed = 7u;
    Particles P; P.initialize(sp, g, rp, stream); stream.synchronize();

    src.zero_rho_j(stream);
    Depositor<CfgDarwin>::charge_current(P, src, g, rp, stream);
    stream.synchronize();

    std::vector<float> h_Jx(N), h_Jy(N), h_Jz2(N);
    CUDA_CHECK(cudaMemcpy(h_Jx.data(), src.Jx.data(), src.Jx.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Jy.data(), src.Jy.data(), src.Jy.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Jz2.data(), src.Jz.data(), src.Jz.bytes(), cudaMemcpyDeviceToHost));
    double mx = 0, my = 0, mz = 0;
    for (int c = 0; c < N; ++c) { mx += h_Jx[c]; my += h_Jy[c]; mz += h_Jz2[c]; }
    mx /= N; my /= N; mz /= N;
    const double jx_expect = rp.qm * 1.0 * vd;   // q n0 vd = -0.3
    std::printf("Current deposit: mean Jx=%.4f (expect %.4f)  mean Jy=%.4f Jz=%.4f (expect 0)\n",
                mx, jx_expect, my, mz);
    if (std::fabs(mx - jx_expect) > 0.02) { std::printf("FAIL: mean Jx off\n"); ok = false; }
    if (std::fabs(my) > 0.02 || std::fabs(mz) > 0.02) { std::printf("FAIL: mean Jy/Jz not ~0\n"); ok = false; }

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
