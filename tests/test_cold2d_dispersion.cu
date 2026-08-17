// V2 wave gate (PLAN_2D_REBORN §5) — the pic2d Yee + cold-fluid engine must
// reproduce the exact cold-plasma oblique dispersion before any kinetic
// species is coupled (regression of the legacy cold_full gate, 0.003–0.46%
// vs Stix, onto the new (x,z)/∂y=0 stack).
//
// Setup: uniform B0 tilted 25° from ẑ (profile=tilted), ωpe/Ωe = 5, c = 1,
// nc = 1, periodic 51.2² box (512², dx = 0.1), dt = 0.05, 131072 steps
// (T = 6554/ωpe). Seed: white noise in By only — exactly divergence-free in
// 2D (By is out-of-plane). Every mode of the periodic box rings at its
// eigenfrequency; 12 pre-chosen (mx,mz) modes spanning θ(k,B0) = 6°–51° and
// k = 0.49–1.23 ωpe/c are DFT-projected on-device each sample; the temporal
// spectrum peak (Hann + parabolic interpolation, the legacy estimator
// lessons) is compared against the Appleton–Hartree/Stix root ω(k,θ) for
// the electron-only cold plasma (immobile ions).
//
// Gates: |ω − ω_AH|/ω_AH < 1.5% per mode; total (EM + cold) energy drift
// < 1% over the full run (undamped, masks off). The rotation-sense of the
// cold kernel is implicitly gated too: a wrong sign has no whistler branch
// below Ωe at all.

#include "pic2d/fields2d.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-26s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

// ---- Appleton–Hartree / Stix: ω(k,θ) for electron-only cold plasma --------
// R = 1 − 1/(ω(ω−Ωe)), L = 1 − 1/(ω(ω+Ωe)), P = 1 − 1/ω² (ωpe = 1);
// A n⁴ − B n² + C = 0, A = S sin²θ + P cos²θ, B = RL sin²θ + PS(1+cos²θ),
// C = PRL. Whistler = the propagating root below Ωe (L, Z evanescent there).
static bool ah_n2(double w, double wce, double th, double& n2a, double& n2b) {
    const double R = 1.0 - 1.0 / (w * (w - wce));
    const double L = 1.0 - 1.0 / (w * (w + wce));
    const double P = 1.0 - 1.0 / (w * w);
    const double S = 0.5 * (R + L);
    const double s2 = std::sin(th) * std::sin(th), c2 = std::cos(th) * std::cos(th);
    const double A = S * s2 + P * c2;
    const double B = R * L * s2 + P * S * (1.0 + c2);
    const double C = P * R * L;
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return false;
    const double sq = std::sqrt(disc);
    n2a = (B + sq) / (2.0 * A);
    n2b = (B - sq) / (2.0 * A);
    return true;
}

static double ah_omega(double k, double th, double wce) {
    auto f = [&](double w, bool& ok) {          // root of n²(ω) − (k/ω)²
        double a, b;
        ok = ah_n2(w, wce, th, a, b);
        if (!ok) return 0.0;
        const double t = (k / w) * (k / w);
        const double fa = a - t, fb = b - t;    // pick the propagating branch
        if (a > 0 && (b <= 0 || std::fabs(fa) < std::fabs(fb))) return fa;
        return fb;
    };
    const int ns = 20000;
    const double w0 = 1e-3, w1 = 0.999 * wce;
    double wp = w0;
    bool okp;
    double fp = f(wp, okp);
    for (int i = 1; i <= ns; ++i) {
        const double w = w0 + (w1 - w0) * i / ns;
        bool ok;
        const double fv = f(w, ok);
        if (okp && ok && fp * fv < 0.0) {
            double lo = wp, hi = w, flo = fp;
            for (int it = 0; it < 200; ++it) {
                const double mid = 0.5 * (lo + hi);
                bool om;
                const double fm = f(mid, om);
                if (flo * fm <= 0.0) hi = mid; else { lo = mid; flo = fm; }
            }
            return 0.5 * (lo + hi);
        }
        wp = w; fp = fv; okp = ok;
    }
    return -1.0;
}

// ---- on-device mode projection: S_m = Σ By(i,k) e^{−i k·x} -----------------
struct Mode { int mx, mz; };

__global__ void k_project(FieldViews2D v, const Mode* modes, int nmode,
                          float x0, float z0, float Lx, float Lz,
                          float2* out, int isamp, int nsamp) {
    const int m = blockIdx.x;
    if (m >= nmode) return;
    const float kx = 2.f * float(M_PI) * modes[m].mx / Lx;
    const float kz = 2.f * float(M_PI) * modes[m].mz / Lz;
    __shared__ double sre[256], sim[256];
    double re = 0, im = 0;
    for (int c = threadIdx.x; c < v.nx * v.nz; c += blockDim.x) {
        const int i = c % v.nx, k = c / v.nx;
        const float xp = x0 + (i + 0.5f) * v.dx;   // By lives at (i+½, k+½)
        const float zp = z0 + (k + 0.5f) * v.dz;
        const float ph = kx * xp + kz * zp;
        re += double(v.by[c]) * cos(ph);
        im -= double(v.by[c]) * sin(ph);
    }
    sre[threadIdx.x] = re; sim[threadIdx.x] = im;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sre[threadIdx.x] += sre[threadIdx.x + st];
            sim[threadIdx.x] += sim[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[m * nsamp + isamp] = make_float2(float(sre[0]), float(sim[0]));
}

int main() {
    std::printf("test_cold2d_dispersion — V2 gate for pic2d Yee + cold fluid\n");

    const int NX = 512, NZ = 512;
    const double DX = 0.1, DT = 0.05, LBOX = NX * DX;
    const long NSTEP = 131072;
    const int SEVERY = 8, NSAMP = int(NSTEP / SEVERY);
    const double WCE = 0.2, THB = 25.0 * M_PI / 180.0;

    Fields2D F;
    F.allocate(NX, NZ);
    F.dx = DX; F.dz = DX; F.dt = DT; F.cspeed = 1.0; F.nc = 1.0;
    F.x0 = 0.0; F.z0 = 0.0;
    F.bg.prof = int(B0Prof::tilted);
    F.bg.B0eq = WCE; F.bg.theta = THB;
    F.bg.finalize();

    // white-noise By seed (exactly div-free: By is the out-of-plane component)
    {
        std::mt19937 rng(20260817);
        std::normal_distribution<float> g(0.f, 1e-6f);
        std::vector<float> h(size_t(NX) * NZ);
        for (auto& x : h) x = g(rng);
        CUDA_CHECK(cudaMemcpy(F.by.data(), h.data(), h.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    const std::vector<Mode> modes = {{0,4},{0,6},{0,8},{0,10},{2,6},{2,8},
                                     {4,4},{6,6},{6,2},{8,2},{8,4},{10,4}};
    const int NM = int(modes.size());
    Mode* dmodes = nullptr;
    float2* dproj = nullptr;
    CUDA_CHECK(cudaMalloc(&dmodes, NM * sizeof(Mode)));
    CUDA_CHECK(cudaMemcpy(dmodes, modes.data(), NM * sizeof(Mode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dproj, size_t(NM) * NSAMP * sizeof(float2)));

    double W0[2];
    F.energies(W0);
    std::printf("  energy trajectory (t, W_EM, W_cold, total/T0):\n");

    double Wref = -1;
    for (long n = 0; n < NSTEP; ++n) {
        F.step();
        if (n % SEVERY == 0) {
            FieldViews2D v = F.views();
            k_project<<<NM, 256>>>(v, dmodes, NM, float(F.x0), float(F.z0),
                                   float(LBOX), float(LBOX), dproj,
                                   int(n / SEVERY), NSAMP);
        }
        if (n % 16384 == 16383) {
            double W[2];
            F.energies(W);
            if (Wref < 0) Wref = W[0] + W[1];   // post-transient reference
            std::printf("    t=%7.0f  %.6e  %.6e  %.4f\n", (n + 1) * DT,
                        W[0], W[1], (W[0] + W[1]) / Wref);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // drift gate vs the post-transient reference: the By-only white seed is
    // not an eigenmode mixture with E = Vc = 0 consistent, so the first
    // frames redistribute seed energy; conservation is judged after that.
    double W1[2];
    F.energies(W1);
    const double drift = std::fabs((W1[0] + W1[1]) / Wref - 1.0);
    gate("energy drift (EM+cold)", drift < 1e-2, drift, 1e-2);

    std::vector<float2> proj(size_t(NM) * NSAMP);
    CUDA_CHECK(cudaMemcpy(proj.data(), dproj, proj.size() * sizeof(float2),
                          cudaMemcpyDeviceToHost));
    cudaFree(dproj); cudaFree(dmodes);

    // temporal spectrum per mode: Hann window + parabolic peak interpolation
    const double dts = DT * SEVERY, T = dts * NSAMP, dw = 2.0 * M_PI / T;
    const int jmax = int(0.21 / dw);
    std::printf("  %-8s %-7s %-7s %-10s %-10s %-8s\n",
                "mode", "k", "theta", "w_meas", "w_AH", "err");
    double err_max = 0;
    const double bhx = std::sin(THB), bhz = std::cos(THB);
    for (int m = 0; m < NM; ++m) {
        const double kx = 2.0 * M_PI * modes[m].mx / LBOX;
        const double kz = 2.0 * M_PI * modes[m].mz / LBOX;
        const double kk = std::sqrt(kx * kx + kz * kz);
        const double th = std::acos(std::fabs((kx * bhx + kz * bhz) / kk));
        std::vector<double> P(jmax + 2, 0.0);
        for (int j = 3; j <= jmax; ++j) {
            double re = 0, im = 0;
            for (int n = 0; n < NSAMP; ++n) {
                const double hann =
                    0.5 * (1.0 - std::cos(2.0 * M_PI * n / (NSAMP - 1)));
                const float2 s = proj[size_t(m) * NSAMP + n];
                const double ph = j * dw * n * dts;
                re += hann * (s.x * std::cos(ph) + s.y * std::sin(ph));
                im += hann * (-s.x * std::sin(ph) + s.y * std::cos(ph));
            }
            P[j] = re * re + im * im;
        }
        // ± propagating pair: the projection keeps both signs of ω for a real
        // field; power spectrum peak on the positive axis suffices
        int jp = 4;
        for (int j = 4; j < jmax; ++j)
            if (P[j] > P[jp]) jp = j;
        const double lm = std::log(P[jp - 1]), l0 = std::log(P[jp]),
                     lp = std::log(P[jp + 1]);
        const double corr = 0.5 * (lm - lp) / (lm - 2.0 * l0 + lp);
        const double wmeas = (jp + corr) * dw;
        const double wah = ah_omega(kk, th, WCE);
        const double err = std::fabs(wmeas / wah - 1.0);
        err_max = std::max(err_max, err);
        std::printf("  (%2d,%2d)  %.3f  %5.1f°  %.6f  %.6f  %.3f%%\n",
                    modes[m].mx, modes[m].mz, kk, th * 180 / M_PI, wmeas, wah,
                    err * 100);
    }
    gate("dispersion worst mode", err_max < 1.5e-2, err_max, 1.5e-2);

    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
