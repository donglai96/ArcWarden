// ArcWarden — Step 9 test: particle loading (quiet Maxwellian) + migrate.
//
// Checks:
//   1. Positions in [0,nx) x [0,ny); cell == idx(floor x, floor y); each cell has
//      exactly ppc particles (stratified quiet load).
//   2. Velocity statistics ≈ Maxwellian: per-component mean ≈ 0 (vd=0), std ≈ vth,
//      fraction within ±vth ≈ 0.6827 (Gaussian).
//   3. Momentum ~ 0 (quiet start); kinetic energy/particle ≈ 1.5 vth² (3 dims).
//   4. migrate() keeps positions in range and cell consistent after a shove.
//
// Exits non-zero on failure.

#include "pic/grid.hpp"
#include "pic/particles.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc;

int main() {
    const int nx = 16, ny = 16;
    Grid g(nx, ny, 2.0 * 3.14159265358979, 2.0 * 3.14159265358979);
    RunParams rp;
    rp.ppc = 64;
    rp.vth = 1.0;
    rp.vd  = 0.0;

    Particles parts;
    CudaStream stream;
    parts.initialize(rp, g, stream);
    stream.synchronize();

    const int N = static_cast<int>(parts.n);
    if (N != rp.ppc * nx * ny) { std::printf("FAIL: N=%d\n", N); return 1; }

    std::vector<float> x(N), y(N), ux(N), uy(N), uz(N);
    std::vector<int>   cell(N);
    auto dl = [&](void* h, const void* d, size_t b) {
        CUDA_CHECK(cudaMemcpy(h, d, b, cudaMemcpyDeviceToHost)); };
    dl(x.data(), parts.x.data(), parts.x.bytes());
    dl(y.data(), parts.y.data(), parts.y.bytes());
    dl(ux.data(), parts.ux.data(), parts.ux.bytes());
    dl(uy.data(), parts.uy.data(), parts.uy.bytes());
    dl(uz.data(), parts.uz.data(), parts.uz.bytes());
    dl(cell.data(), parts.cell.data(), parts.cell.bytes());

    bool ok = true;

    // ---- 1. positions + cell + stratification ----
    std::vector<int> count(nx * ny, 0);
    for (int p = 0; p < N; ++p) {
        if (!(x[p] >= 0.0f && x[p] < nx && y[p] >= 0.0f && y[p] < ny)) {
            std::printf("FAIL: particle %d out of range (%.3f,%.3f)\n", p, x[p], y[p]);
            ok = false; break;
        }
        const int ci = int(std::floor(x[p])), cj = int(std::floor(y[p]));
        if (cell[p] != g.idx(ci, cj)) {
            std::printf("FAIL: cell mismatch p=%d\n", p); ok = false; break;
        }
        count[cell[p]]++;
    }
    int cmin = N, cmax = 0;
    for (int c = 0; c < nx * ny; ++c) { cmin = std::min(cmin, count[c]); cmax = std::max(cmax, count[c]); }
    if (cmin != rp.ppc || cmax != rp.ppc) {
        std::printf("FAIL: stratification, cell counts [%d,%d] != ppc=%d\n", cmin, cmax, rp.ppc);
        ok = false;
    }

    // ---- 2/3. velocity statistics ----
    auto stats = [&](const std::vector<float>& v, double& mean, double& std_, double& frac1) {
        double s = 0, s2 = 0; int in1 = 0;
        for (float a : v) { s += a; s2 += double(a) * a; if (std::fabs(a) <= rp.vth) ++in1; }
        mean = s / v.size();
        std_ = std::sqrt(s2 / v.size() - mean * mean);
        frac1 = double(in1) / v.size();
    };
    double mx, sx, fx, my, sy, fy, mz, sz, fz;
    stats(ux, mx, sx, fx); stats(uy, my, sy, fy); stats(uz, mz, sz, fz);
    std::printf("vx: mean=%+.4f std=%.4f frac|v|<vth=%.4f\n", mx, sx, fx);
    std::printf("vy: mean=%+.4f std=%.4f frac|v|<vth=%.4f\n", my, sy, fy);
    std::printf("vz: mean=%+.4f std=%.4f frac|v|<vth=%.4f\n", mz, sz, fz);

    auto check = [&](double m, double sd, double fr, const char* nm) {
        if (std::fabs(m) > 0.03)               { std::printf("FAIL: %s mean\n", nm); ok = false; }
        if (std::fabs(sd - rp.vth) > 0.03)     { std::printf("FAIL: %s std\n", nm); ok = false; }
        if (std::fabs(fr - 0.6827) > 0.02)     { std::printf("FAIL: %s frac\n", nm); ok = false; }
    };
    check(mx, sx, fx, "vx"); check(my, sy, fy, "vy"); check(mz, sz, fz, "vz");

    double ke = 0;
    for (int p = 0; p < N; ++p) ke += 0.5 * (double(ux[p]) * ux[p] + double(uy[p]) * uy[p] + double(uz[p]) * uz[p]);
    const double ke_pp = ke / N, ke_expect = 1.5 * rp.vth * rp.vth;
    std::printf("KE/particle = %.4f  (expect 1.5 vth^2 = %.4f)\n", ke_pp, ke_expect);
    if (std::fabs(ke_pp - ke_expect) > 0.03) { std::printf("FAIL: KE/particle\n"); ok = false; }

    // ---- 4. migrate after a shove ----
    // push every particle by +nx in x and -ny-0.5 in y on the host, upload, migrate.
    for (int p = 0; p < N; ++p) { x[p] += nx; y[p] -= (ny + 0.5f); }
    CUDA_CHECK(cudaMemcpy(parts.x.data(), x.data(), parts.x.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parts.y.data(), y.data(), parts.y.bytes(), cudaMemcpyHostToDevice));
    parts.migrate(g, stream);
    stream.synchronize();
    dl(x.data(), parts.x.data(), parts.x.bytes());
    dl(y.data(), parts.y.data(), parts.y.bytes());
    dl(cell.data(), parts.cell.data(), parts.cell.bytes());
    for (int p = 0; p < N; ++p) {
        if (!(x[p] >= 0.0f && x[p] < nx && y[p] >= 0.0f && y[p] < ny)) {
            std::printf("FAIL: post-migrate out of range p=%d (%.3f,%.3f)\n", p, x[p], y[p]);
            ok = false; break;
        }
        if (cell[p] != g.idx(int(std::floor(x[p])), int(std::floor(y[p])))) {
            std::printf("FAIL: post-migrate cell p=%d\n", p); ok = false; break;
        }
    }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
