// RSM gate V2 — spectral m = 1 cold oblique dispersion + polarization + E∥
// (docs/RSM_MODEL_DEFINITION.md ladder V2, risks R2/R5).
//
// Uniform B0 along x, cold fluid only, periodic x, ONE spectral harmonic
// k1 = 0.4. Seed: white noise on B1z ONLY — exactly divergence-free for
// in-plane k (k·B1 = kx·B1x + k1·B1y = 0 when B1x = B1y = 0), satisfying R2
// without a projection. For kx modes mx = 1..5 (theta = 64..22 deg):
//   (1) omega(kx, k1) must match the EXACT Stix cold dispersion to < 1.5%
//       at the refined level, with ~2nd-order convergence from the coarse
//       level (same protocol/tolerances as test_cold_fluid_oblique, which
//       gates the REAL-SPACE cold_full path against the SAME solver — that
//       is the R5 cross-check, transitively);
//   (2) field energy must not drift (no numerical damping/growth);
//   (3) polarization ratios |E1x/E1y| and |E1z/E1y| at the spectral peak
//       must match the Stix eigenvector — |E1x/E1y| is the LONGITUDINAL
//       (E∥, B0 ∥ x) response, the physics the RSM exists to add.
//
// The m = 1 arrays are complex, so a single spatial DFT coefficient at +kx
// is already the full signed-frequency time series (no By+iBz pairing
// needed); modes at +-omega are the two oblique propagation directions.

#include "pic/rsm_oblique.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc;
using C = std::complex<double>;

// exact cold electron-only dispersion (same solver as test_cold_fluid_oblique;
// R5 demands the two paths gate against the SAME theory): whistler root
// w in (0, We) of A n^4 - B n^2 + C = 0.
static double whistler_w_exact(double k, double th, double wpe, double We) {
    const double s = std::sin(th), c = std::cos(th);
    auto det = [&](double w) {
        const double wp2 = wpe * wpe;
        const double R = 1.0 - wp2 / (w * (w - We));
        const double L = 1.0 - wp2 / (w * (w + We));
        const double S = 0.5 * (R + L), P = 1.0 - wp2 / (w * w);
        const double n2 = (k * k) / (w * w);
        const double A = S * s * s + P * c * c;
        const double B = R * L * s * s + P * S * (1.0 + c * c);
        return A * n2 * n2 - B * n2 + R * L * P;
    };
    double lo = 1e-4 * We, hi = 0.999 * We;
    double flo = det(lo);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (det(mid) * flo > 0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Stix eigenvector at (w, k, th): null vector of the cold wave matrix.
// Stix frame has B0 along z_s, k in the x_s-z_s plane; ours has B0 along x
// with k = (kx, k1, 0), so (x_s, y_s, z_s) = (y, z, x) ours. Returns the
// magnitude ratios |E1x/E1y|, |E1z/E1y| (sign-agnostic: k -> -k flips only
// the cross-term sign, magnitudes invariant — the measured peak may sit at
// either signed frequency).
static void stix_ratios(double w, double k, double th, double wpe, double We,
                        double& rxy, double& rzy) {
    const double wp2 = wpe * wpe;
    const double R = 1.0 - wp2 / (w * (w - We));
    const double L = 1.0 - wp2 / (w * (w + We));
    const double S = 0.5 * (R + L), D = 0.5 * (R - L), P = 1.0 - wp2 / (w * w);
    const double n2 = (k * k) / (w * w);
    const double s = std::sin(th), c = std::cos(th);
    const C M[3][3] = {
        { C(S - n2 * c * c), C(0, -D),   C(n2 * s * c)     },
        { C(0, D),           C(S - n2),  C(0)              },
        { C(n2 * s * c),     C(0),       C(P - n2 * s * s) } };
    // null vector = the largest pairwise cross product of rows
    C best[3]; double bn = -1.0;
    const int pairs[3][2] = { {0, 1}, {0, 2}, {1, 2} };
    for (auto& pr : pairs) {
        const C* a = M[pr[0]]; const C* b = M[pr[1]];
        const C v[3] = { a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0] };
        const double nn = std::norm(v[0]) + std::norm(v[1]) + std::norm(v[2]);
        if (nn > bn) { bn = nn; best[0] = v[0]; best[1] = v[1]; best[2] = v[2]; }
    }
    // ours: E1y = x_s comp, E1z = y_s comp, E1x = z_s comp
    rxy = std::abs(best[2]) / std::abs(best[0]);
    rzy = std::abs(best[1]) / std::abs(best[0]);
}

struct ModeResult { double werr, rxy_m, rxy_t, rzy_m, rzy_t; };

static bool run_level(int ref, ModeResult* res, double& drift) {
    const int    nx = 64 * ref;
    const double dx = 0.5 / ref, dt = 0.05 / ref, wce = 0.25, nc = 1.0;
    const double k1 = 0.4;
    Grid g(nx, 1, nx * dx, 2.0 * M_PI / k1);

    RunParams rp;
    rp.dt = dt; rp.c = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.cold_nc = nc;
    rp.rsm = 1; rp.rsm_k1 = k1;

    RsmState r;
    r.init(g, rp, 0);
    {   // div-free seed: white noise on B1z only (k · B1 = 0 exactly)
        std::mt19937 gen(20260724u);
        std::uniform_real_distribution<float> un(-1e-6f, 1e-6f);
        std::vector<float2> bz(nx);
        for (auto& z : bz) z = float2{un(gen), un(gen)};
        CUDA_CHECK(cudaMemcpy(r.b1z.data(), bz.data(), nx * sizeof(float2),
                              cudaMemcpyHostToDevice));
    }

    const long nsteps = 40000L * ref;               // physical T = 2000
    const int  every  = 4 * ref;
    const int  nsamp  = (int)(nsteps / every);
    const int  NM     = 5;
    const int  mx[NM] = {1, 2, 3, 4, 5};

    // time series of the spatial DFT coefficient at +kx for 4 fields
    std::vector<C> tsB((size_t)NM * nsamp), tsEx((size_t)NM * nsamp),
                   tsEy((size_t)NM * nsamp), tsEz((size_t)NM * nsamp);
    std::vector<float2> hbz(nx), hex(nx), hey(nx), hez(nx);

    double e1 = 0, e2 = 0; int n1 = 0, n2 = 0;
    std::vector<float2> hb1(nx), hb2(nx), hb3(nx);
    for (long n = 1; n <= nsteps; ++n) {
        rsm_cold_step(r, rp, 0);
        if (n % every) continue;
        CUDA_CHECK(cudaMemcpy(hbz.data(), r.b1z.data(), nx * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hex.data(), r.e1x.data(), nx * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hey.data(), r.e1y.data(), nx * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hez.data(), r.e1z.data(), nx * 8, cudaMemcpyDeviceToHost));
        const int s = (int)(n / every) - 1;
        for (int m = 0; m < NM; ++m) {
            C aB(0), aEx(0), aEy(0), aEz(0);
            for (int i = 0; i < nx; ++i) {
                const double phn = -2.0 * M_PI * mx[m] * (double)i / nx;
                const double phs = -2.0 * M_PI * mx[m] * (i + 0.5) / nx;  // i+1/2 sites
                const C en(std::cos(phn), std::sin(phn));
                const C es(std::cos(phs), std::sin(phs));
                aB  += es * C(hbz[i].x, hbz[i].y);
                aEx += es * C(hex[i].x, hex[i].y);
                aEy += en * C(hey[i].x, hey[i].y);
                aEz += en * C(hez[i].x, hez[i].y);
            }
            tsB [(size_t)m * nsamp + s] = aB;
            tsEx[(size_t)m * nsamp + s] = aEx;
            tsEy[(size_t)m * nsamp + s] = aEy;
            tsEz[(size_t)m * nsamp + s] = aEz;
        }
        CUDA_CHECK(cudaMemcpy(hb1.data(), r.b1x.data(), nx * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb2.data(), r.b1y.data(), nx * 8, cudaMemcpyDeviceToHost));
        double eb = 0;
        for (int i = 0; i < nx; ++i)
            eb += (double)hb1[i].x * hb1[i].x + (double)hb1[i].y * hb1[i].y
                + (double)hb2[i].x * hb2[i].x + (double)hb2[i].y * hb2[i].y
                + (double)hbz[i].x * hbz[i].x + (double)hbz[i].y * hbz[i].y;
        if (n <= nsteps / 2) { e1 += eb; ++n1; } else { e2 += eb; ++n2; }
    }
    e1 /= n1; e2 /= n2;

    bool ok = true;
    const double dk = 2.0 * M_PI / (nx * dx);
    std::printf("ref=%d: mx theta    k      w_meas   w_exact  err    "
                "|Ex/Ey| m/t      |Ez/Ey| m/t\n", ref);
    for (int m = 0; m < NM; ++m) {
        const double kx = mx[m] * dk;
        const double k = std::hypot(kx, k1), th = std::atan2(k1, kx);
        const C* f = &tsB[(size_t)m * nsamp];
        // Hann-windowed DFT magnitude at frequency-bin q (may be negative)
        auto amp_at = [&](const C* ser, double q) {
            const double w = 2.0 * M_PI * q / (nsamp * every * dt);
            C acc(0);
            for (int s2 = 0; s2 < nsamp; ++s2) {
                const double win = 0.5 * (1.0 - std::cos(2.0 * M_PI * s2 / (nsamp - 1)));
                const double ph = w * (s2 * every * dt);
                acc += win * ser[s2] * C(std::cos(ph), std::sin(ph));
            }
            return acc;
        };
        int best = 0; double bmag = -1;
        const int NF = 4000;
        for (int q = -NF / 2; q < NF / 2; ++q) {
            const double w = 2.0 * M_PI * q / (nsamp * every * dt);
            if (std::fabs(w) > 0.99 * wce || std::fabs(w) < 0.02 * wce) continue;
            const double mag = std::abs(amp_at(f, q));
            if (mag > bmag) { bmag = mag; best = q; }
        }
        const double m0 = std::abs(amp_at(f, best));
        const double mp = std::abs(amp_at(f, best + 1));
        const double mm2 = std::abs(amp_at(f, best - 1));
        const double den = mm2 - 2.0 * m0 + mp;
        const double off = (std::fabs(den) > 1e-30) ? 0.5 * (mm2 - mp) / den : 0.0;
        const double qpk = best + off;
        const double wm = std::fabs(2.0 * M_PI * qpk / (nsamp * every * dt));
        const double wt = whistler_w_exact(k, th, 1.0, wce);
        res[m].werr = std::fabs(wm - wt) / wt;
        // polarization at the peak: complex E amplitudes at the same bin
        const double aEx = std::abs(amp_at(&tsEx[(size_t)m * nsamp], qpk));
        const double aEy = std::abs(amp_at(&tsEy[(size_t)m * nsamp], qpk));
        const double aEz = std::abs(amp_at(&tsEz[(size_t)m * nsamp], qpk));
        res[m].rxy_m = aEx / aEy; res[m].rzy_m = aEz / aEy;
        stix_ratios(wt, k, th, 1.0, wce, res[m].rxy_t, res[m].rzy_t);
        std::printf("%3d  %5.1f  %5.3f  %.5f  %.5f  %.3f%%  %.4f/%.4f  %.4f/%.4f\n",
                    mx[m], th * 180 / M_PI, k, wm, wt, 100 * res[m].werr,
                    res[m].rxy_m, res[m].rxy_t, res[m].rzy_m, res[m].rzy_t);
    }
    drift = e2 / e1 - 1.0;
    std::printf("energy drift half2/half1 - 1 = %+.4f\n", drift);
    if (std::fabs(drift) > 0.05) { std::printf("FAIL: energy drift %.1f%%\n", 100 * drift); ok = false; }
    return ok;
}

int main() {
    ModeResult A[5], B[5];
    double dA, dB;
    bool ok = run_level(1, A, dA);
    ok = run_level(2, B, dB) && ok;
    for (int m = 0; m < 5; ++m) {
        if (B[m].werr > 0.015) {
            std::printf("FAIL: refined mode %d w err %.2f%% > 1.5%%\n", m, 100 * B[m].werr); ok = false;
        }
        if (A[m].werr > 0.01 && A[m].werr / B[m].werr < 2.5) {
            std::printf("FAIL: mode %d converges x%.1f < 2.5\n", m, A[m].werr / B[m].werr); ok = false;
        }
        const double exy = std::fabs(B[m].rxy_m - B[m].rxy_t)
                         / std::max(B[m].rxy_t, 0.05);
        const double ezy = std::fabs(B[m].rzy_m - B[m].rzy_t)
                         / std::max(B[m].rzy_t, 0.05);
        if (exy > 0.05) { std::printf("FAIL: mode %d |Ex/Ey| off %.1f%% (E_par!)\n", m, 100 * exy); ok = false; }
        if (ezy > 0.05) { std::printf("FAIL: mode %d |Ez/Ey| off %.1f%%\n", m, 100 * ezy); ok = false; }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
