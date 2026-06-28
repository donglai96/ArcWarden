// ArcWarden — Step 10 test [S4]: charge conservation.
//
// CIC conserves total charge exactly: sum(rho)*dx*dy == N * q * weight, for any
// particle positions (the 4 CIC weights sum to 1, periodic wrap loses nothing).
// Checks:
//   1. Single particle: deposits to 4 cells, total charge = q*weight.
//   2. Boundary particle (straddling the periodic edge): wraps, still conserved.
//   3. Uniform quiet load (ppc/cell): total charge = N*q*weight AND rho ≈ q*n0.
//
// Exits non-zero on failure.

#include "pic/depositor.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/sources.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;
using Cfg = arc::Cfg;

// Deposit a handful of hand-placed particles and return total charge sum(rho)*dx*dy.
static double deposit_points(const Grid& g, const RunParams& rp,
                             const std::vector<float>& xs, const std::vector<float>& ys,
                             std::vector<float>& rho_out) {
    const int np = static_cast<int>(xs.size());
    DeviceArray<float> x(np), y(np), ux(np), uy(np), uz(np), w(np);
    DeviceArray<int>   cell(np);
    CUDA_CHECK(cudaMemcpy(x.data(), xs.data(), x.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y.data(), ys.data(), y.bytes(), cudaMemcpyHostToDevice));
    std::vector<float> ws(np, static_cast<float>(rp.weight));   // uniform weight
    CUDA_CHECK(cudaMemcpy(w.data(), ws.data(), w.bytes(), cudaMemcpyHostToDevice));

    ParticleViews pv{ x.view(), y.view(), ux.view(), uy.view(), uz.view(),
                      w.view(), cell.view(), np };
    Sources src(g);
    CudaStream stream;
    src.zero(stream);
    deposit_charge_kernel<Cfg><<<1, 256, 0, stream>>>(pv, src.views(), g, rp);
    CUDA_CHECK(cudaPeekAtLastError());
    stream.synchronize();

    rho_out.resize(g.real_size());
    CUDA_CHECK(cudaMemcpy(rho_out.data(), src.rho.data(), src.rho.bytes(),
                          cudaMemcpyDeviceToHost));
    double s = 0;
    for (float r : rho_out) s += r;
    return s * g.dx * g.dy;
}

int main() {
    const int nx = 8, ny = 8;
    Grid g(nx, ny, 1.0, 1.0);     // dx=dy=0.125
    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.ppc = 32;
    rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;
    const double q = rp.qm;
    const double tol = 1e-5;
    bool ok = true;

    // ---- 1. single particle ----
    {
        std::vector<float> rho;
        const double Q = deposit_points(g, rp, {2.3f}, {3.7f}, rho);
        const double expect = 1.0 * q * rp.weight;
        // exactly 4 nonzero cells
        int nz = 0; for (float r : rho) if (std::fabs(r) > 0) ++nz;
        std::printf("single: Q=%.6e expect %.6e (nonzero cells=%d)\n", Q, expect, nz);
        if (std::fabs(Q - expect) > tol * std::fabs(expect)) { std::printf("FAIL single Q\n"); ok = false; }
        if (nz != 4) { std::printf("FAIL single: expected 4 cells\n"); ok = false; }
    }

    // ---- 2. boundary particle (straddles periodic edge at i=7|0, j=7|0) ----
    {
        std::vector<float> rho;
        const double Q = deposit_points(g, rp, {7.6f}, {7.6f}, rho);
        const double expect = 1.0 * q * rp.weight;
        // wrapped neighbor cells (0,7),(7,0),(0,0) must be populated
        const bool wrapped = std::fabs(rho[g.idx(0,0)]) > 0 && std::fabs(rho[g.idx(0,7)]) > 0
                          && std::fabs(rho[g.idx(7,0)]) > 0;
        std::printf("boundary: Q=%.6e expect %.6e (wrapped cells set=%d)\n", Q, expect, wrapped);
        if (std::fabs(Q - expect) > tol * std::fabs(expect)) { std::printf("FAIL boundary Q\n"); ok = false; }
        if (!wrapped) { std::printf("FAIL boundary: wrap cells empty\n"); ok = false; }
    }

    // ---- 3. uniform quiet load ----
    {
        Particles parts;
        CudaStream stream;
        parts.initialize(rp, g, stream);
        Sources src(g);
        src.zero(stream);
        Depositor<Cfg>::charge(parts, src, g, rp, stream);
        stream.synchronize();

        std::vector<float> rho(g.real_size());
        CUDA_CHECK(cudaMemcpy(rho.data(), src.rho.data(), src.rho.bytes(),
                              cudaMemcpyDeviceToHost));
        double s = 0, smin = 1e30, smax = -1e30;
        for (float r : rho) { s += r; smin = std::fmin(smin, r); smax = std::fmax(smax, r); }
        const double Q = s * g.dx * g.dy;
        const double N = static_cast<double>(parts.n);
        const double expect = N * q * rp.weight;
        const double rho_expect = q * rp.n0;     // = -1
        const double mean_rho = s / rho.size();
        std::printf("uniform: Q=%.6e expect %.6e ; rho mean=%.5f [min %.5f, max %.5f] expect %.5f\n",
                    Q, expect, mean_rho, smin, smax, rho_expect);
        if (std::fabs(Q - expect) > tol * std::fabs(expect)) { std::printf("FAIL uniform Q\n"); ok = false; }
        if (std::fabs(mean_rho - rho_expect) > 1e-5) { std::printf("FAIL uniform rho mean\n"); ok = false; }
        // quiet start => near-flat; allow modest CIC ripple
        if (std::fabs(smax - smin) > 0.05 * std::fabs(rho_expect)) {
            std::printf("FAIL uniform: rho not flat (spread %.4f)\n", smax - smin); ok = false;
        }
    }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
