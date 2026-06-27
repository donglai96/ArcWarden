// ArcWarden — Step 6 test [S1]: FFT round trip + Parseval.
//
// Checks two properties of the SpectralEngine R2C/C2R wrappers:
//   1. Round trip: c2r(r2c(f)) ≈ f, i.e. the 1/(nx*ny) normalization is right.
//   2. Parseval:   Σ_x f(x)²  ==  (1/N) Σ_k |F(k)|²  over the FULL spectrum,
//      reconstructed from the stored R2C half by counting non-DC/non-Nyquist
//      x-modes twice (they stand in for their dropped conjugate).
//
// Exits non-zero on failure so CTest reports it.

#include "pic/grid.hpp"
#include "pic/spectral.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int nx = 64, ny = 32;
    const double Lx = 2.0 * 3.14159265358979323846;  // 2π
    const double Ly = 2.0 * 3.14159265358979323846;
    Grid  g(nx, ny, Lx, Ly);
    KGrid kg(g);

    SpectralEngine engine(g);
    CudaStream stream;

    const int N    = g.real_size();        // nx*ny
    const int Nk   = kg.complex_size();     // nkx*ny

    // ---- build a smooth, non-trivial real field f(x,y) ----
    std::vector<float> h_f(N);
    double sum_sq = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = i * g.dx, y = j * g.dy;
            const double v = 0.7 * std::sin(x) + 0.5 * std::cos(2.0 * x + y)
                           - 0.3 * std::sin(3.0 * y) + 1.0;  // includes a DC offset
            h_f[g.idx(i, j)] = static_cast<float>(v);
            sum_sq += v * v;
        }
    }

    // ---- device buffers ----
    DeviceArray<float>        d_f(N);
    DeviceArray<cufftComplex> d_F(Nk);
    CUDA_CHECK(cudaMemcpy(d_f.data(), h_f.data(), d_f.bytes(),
                          cudaMemcpyHostToDevice));

    // ---- forward transform: f -> F ----
    engine.r2c(d_f.data(), d_F.data(), stream);
    stream.synchronize();

    // ---- Parseval check (before C2R consumes F) ----
    std::vector<cufftComplex> h_F(Nk);
    CUDA_CHECK(cudaMemcpy(h_F.data(), d_F.data(), d_F.bytes(),
                          cudaMemcpyDeviceToHost));
    double e_k = 0.0;
    for (int j = 0; j < kg.nky; ++j) {
        for (int i = 0; i < kg.nkx; ++i) {
            const cufftComplex c = h_F[kg.kidx(i, j)];
            const double mag2 = double(c.x) * c.x + double(c.y) * c.y;
            // x is the reduced axis: every i except DC(0) and Nyquist(nx/2)
            // represents two full-spectrum modes (±k_x).
            const double w = (i == 0 || i == nx / 2) ? 1.0 : 2.0;
            e_k += w * mag2;
        }
    }
    const double e_x        = sum_sq;            // Σ_x f²
    const double e_k_scaled = e_k / double(N);   // (1/N) Σ_full |F|²
    const double parseval_rel = std::fabs(e_k_scaled - e_x) / e_x;

    // ---- inverse transform: F -> f' (normalized); F is consumed ----
    engine.c2r(d_F.data(), d_f.data(), stream);
    stream.synchronize();

    std::vector<float> h_f2(N);
    CUDA_CHECK(cudaMemcpy(h_f2.data(), d_f.data(), d_f.bytes(),
                          cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    for (int k = 0; k < N; ++k) {
        max_abs_err = std::fmax(max_abs_err,
                                std::fabs(double(h_f2[k]) - double(h_f[k])));
    }

    std::printf("FFT round trip:  max|c2r(r2c(f)) - f| = %.3e\n", max_abs_err);
    std::printf("Parseval:        |E_k/N - E_x| / E_x  = %.3e  (E_x=%.4f)\n",
                parseval_rel, e_x);

    const double roundtrip_tol = 1e-4;   // single precision, N=2048
    const double parseval_tol  = 1e-4;
    bool ok = true;
    if (max_abs_err > roundtrip_tol) {
        std::printf("FAIL: round-trip error %.3e exceeds tol %.1e\n",
                    max_abs_err, roundtrip_tol);
        ok = false;
    }
    if (parseval_rel > parseval_tol) {
        std::printf("FAIL: Parseval mismatch %.3e exceeds tol %.1e\n",
                    parseval_rel, parseval_tol);
        ok = false;
    }
    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
