// ArcWarden — Step 8 test [S5]: spectral Poisson on a single sine mode.
//
// Set rho(x,y) = sin(k1 x), k1 = 2π/Lx (one full wave along x, uniform in y).
// Analytic electrostatic solution (eps0 = 1):
//   ∇²φ = -ρ  =>  φ =  sin(k1 x) / k1²
//   E   = -∇φ =>  Ex = -cos(k1 x) / k1 ,  Ey = 0
// With Lx = 2π we have k1 = 1, so Ex(x) = -cos(x), Ey = 0.
//
// Checks Ex amplitude+phase against analytic and Ey ≈ 0. Exits non-zero on fail.

#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/solver_es.hpp"
#include "pic/sources.hpp"
#include "pic/spectral.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int    nx = 64, ny = 32;
    const double TwoPi = 2.0 * 3.14159265358979323846;
    const double Lx = TwoPi, Ly = TwoPi;     // => k1 = 2π/Lx = 1
    Grid g(nx, ny, Lx, Ly);
    const double k1 = TwoPi / Lx;            // 1.0

    Sources src(g);
    Fields  fld(g);
    SpectralEngine engine(g);
    ElectrostaticSpectralSolver solver;
    RunParams rp;                            // defaults: eps0 = 1
    CudaStream stream;

    const int N = g.real_size();

    // ---- load rho = sin(k1 x) ----
    std::vector<float> h_rho(N);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            h_rho[g.idx(i, j)] = static_cast<float>(std::sin(k1 * i * g.dx));
    CUDA_CHECK(cudaMemcpy(src.rho.data(), h_rho.data(), src.rho.bytes(),
                          cudaMemcpyHostToDevice));

    // ---- solve rho -> E ----
    solver.solve(src, fld, engine, rp, stream);
    stream.synchronize();

    std::vector<float> h_Ex(N), h_Ey(N);
    CUDA_CHECK(cudaMemcpy(h_Ex.data(), fld.Ex.data(), fld.Ex.bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ey.data(), fld.Ey.data(), fld.Ey.bytes(),
                          cudaMemcpyDeviceToHost));

    // ---- compare to analytic: Ex = -cos(k1 x)/k1, Ey = 0 ----
    double max_ex_err = 0.0, max_ey = 0.0, max_ex_amp = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x  = i * g.dx;
            const double ex_analytic = -std::cos(k1 * x) / k1;
            const double ex = h_Ex[g.idx(i, j)];
            const double ey = h_Ey[g.idx(i, j)];
            max_ex_err = std::fmax(max_ex_err, std::fabs(ex - ex_analytic));
            max_ey     = std::fmax(max_ey, std::fabs(ey));
            max_ex_amp = std::fmax(max_ex_amp, std::fabs(ex));
        }
    }

    std::printf("Poisson sine:  max|Ex - Ex_analytic| = %.3e  (peak|Ex|=%.4f, expect 1)\n",
                max_ex_err, max_ex_amp);
    std::printf("               max|Ey|               = %.3e  (expect 0)\n", max_ey);

    const double tol = 1e-4;
    bool ok = true;
    if (max_ex_err > tol) { std::printf("FAIL: Ex error exceeds %.1e\n", tol); ok = false; }
    if (max_ey     > tol) { std::printf("FAIL: Ey not ~0 (>%.1e)\n", tol);     ok = false; }
    if (std::fabs(max_ex_amp - 1.0 / k1) > 1e-3) {
        std::printf("FAIL: Ex amplitude %.4f != %.4f\n", max_ex_amp, 1.0 / k1); ok = false;
    }
    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
