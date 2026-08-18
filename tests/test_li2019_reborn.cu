// V3 — Li et al. 2019 Nat. Commun. two-band / 0.5 f_ce gap regression on
// the reborn stack (PLAN_2D_REBORN §5). The kinetic-engine literature
// anchor: the legacy G2.1 replication carved gap/LB 1.07 → 3e-4 over
// ~950 τ_g with a Landau plateau at |v∥| ∈ [0.08, 0.10]c; Darwin-tc got
// 6.8e-4. This test re-runs the identical physics on pic2d.
//
// Setup (decks/li2019_band.ini numbers, quasi-1D in the 2D code):
// 1024×4 cells, dx = 0.05 (Lx = 51.2), periodic, no walls/masks, ALL-PIC
// (nc = 0 — the 10-eV cold component must be kinetic to Landau-resonate),
// B0 uniform tilted: angle(B0, x̂) = 15° (θ_from_ẑ = 75°), ω_pe/Ω_e = 5.
// Species (full-f, nonrel): cold n = 0.8, κ = 4, uth = 0.01 isotropic;
// warm n = 0.2, κ = 1.5, uth∥ = 0.02, uth⊥ = 0.04 (T⊥/T∥ = 4); |u| capped
// at 0.6c in the loader. dt = 0.02, 1.6M steps = 32000/ω_pe ≈ 1018 τ_g.
// ppc 1000+1000 per cell → 8.4M markers (per-x-column statistics = legacy).
//
// Diagnostics: By at 3 probe stations (every 8 steps → 200k samples),
// warm f(u∥) histogram at t = 0 and end; band powers from Hann + DFT of
// the probe series in two windows (early τ_g ∈ [200,250]: one continuous
// band; late τ_g ∈ [890,1010]: two-band + gap), bands in ω/Ω_e:
// LB [0.2,0.44], gap [0.46,0.54], UB [0.56,0.7].
//
// Gates: (a) saturation δBy_rms/B0 ∈ [0.4%, 6%] (legacy δB_rms 1.9%);
// (b) early gap/LB > 0.3 (starts near unity); (c) LATE gap/LB < 1e-2
// (the carving; legacy 3e-4-class); (d) warm plateau fill-in at
// |u∥| ∈ [0.08,0.10]: min(±) f_end/f_init > 2.5 (legacy ~10×);
// (e) UB distinct: late UB/gap > 3.
// Raw products dumped to li3_probe.bin / li3_fpar.bin for offline plots.

#include "pic2d/kinetic2d.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-26s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    const long NSTEP = (argc > 1) ? atol(argv[1]) : 1600000;
    std::printf("test_li2019_reborn — V3 two-band gap anchor (%ld steps = %.0f tau_g)\n",
                NSTEP, NSTEP * 0.02 * 0.2 / (2 * M_PI));

    const int NX = 1024, NZ = 4;
    const double DX = 0.05, DT = 0.02, WCE = 0.2;

    Fields2D F;
    F.allocate(NX, NZ);
    F.dx = DX; F.dz = DX; F.dt = DT; F.cspeed = 1.0; F.nc = 0.0;
    F.x0 = 0; F.z0 = 0;
    F.bg.prof = int(B0Prof::tilted);
    F.bg.B0eq = WCE;
    F.bg.theta = 75.0 * M_PI / 180.0;   // angle(B0, x̂) = 15°
    F.bg.finalize();
    F.allocate_replicas(16);

    auto make_species = [&](int dist, float kap, float n0, float utp, float utq,
                            uint32_t seed, MarkerStore& mk, KineticCfg& C) {
        C.qm = -1.f; C.deltaf = 0; C.rel = 0;
        C.dist = dist; C.kappa = kap;
        C.tpar = utp * utp; C.tperp = utq * utq;
        C.n0 = n0;
        C.L0 = 0.f; C.dL = 0.f; C.edge = 1.f;      // no shell (uniform arm)
        C.wx0 = -1e9f; C.wx1 = 1e9f; C.wz0 = -1e9f; C.wz1 = 1e9f;  // no walls
        const uint64_t N = uint64_t(1000) * NX * NZ;
        mk.allocate(N); mk.n = N;
        MarkerViews mv = mk.views();
        const float wmark = float(n0 * (NX * DX) * (NZ * DX) / double(N));
        k2d::k_load<<<int((N + 255) / 256), 256>>>(mv, C, F.bg, 0.f,
                                                   float(NX * DX), 0.f,
                                                   float(NZ * DX), wmark, seed, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    MarkerStore cold, warm;
    KineticCfg Cc, Cw;
    make_species(1, 4.0f, 0.8f, 0.01f, 0.01f, 20260722u, cold, Cc);
    make_species(1, 1.5f, 0.2f, 0.02f, 0.04f, 20260723u, warm, Cw);
    std::printf("  loaded: cold 4.1M (kappa=4), warm 4.1M (kappa=1.5, A=4)\n");

    // warm f(u∥) histogram (host; b̂ constant on the tilted arm)
    const double bhx = std::sin(F.bg.theta), bhz = std::cos(F.bg.theta);
    const int NB = 120;
    const double UMAX = 0.3;
    auto fpar_hist = [&](std::vector<double>& h) {
        h.assign(NB, 0.0);
        const uint64_t N = warm.n;
        std::vector<float> a(N), b(N), c(N);
        CUDA_CHECK(cudaMemcpy(a.data(), warm.ux.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), warm.uy.data(), N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c.data(), warm.uz.data(), N * 4, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < N; ++i) {
            const double up = a[i] * bhx + c[i] * bhz;
            const int j = int((up + UMAX) / (2 * UMAX) * NB);
            if (j >= 0 && j < NB) h[j] += 1.0;
        }
    };
    std::vector<double> f0h, f1h;
    fpar_hist(f0h);

    // probes: By mid-row at x = 12.8, 25.6, 38.4 (deck stations)
    const int NPR = 3, prx[NPR] = {256, 512, 768}, prk = NZ / 2;
    const int SEVERY = 8;
    const long NSAMP = NSTEP / SEVERY;
    std::vector<float> probe(size_t(NPR) * NSAMP, 0.f);

    const dim3 tb(f2d::TX, f2d::TZ);
    const auto t0 = std::chrono::steady_clock::now();
    for (long n = 0; n < NSTEP; ++n) {
        FieldViews2D v = F.views();
        const Range r = F.full();
        const dim3 nb = f2d::blocks_for(r);
        f2d::k_faraday<<<nb, tb>>>(v, r, float(DT / 2));
        F.zero_j();
        {
            MarkerViews mv = cold.views();
            k2d::k_push_deposit<<<int((cold.n + 255) / 256), 256>>>(
                mv, Cc, v, F.bg, 0.f, 0.f, cold.n);
            MarkerViews mw = warm.views();
            k2d::k_push_deposit<<<int((warm.n + 255) / 256), 256>>>(
                mw, Cw, v, F.bg, 0.f, 0.f, warm.n);
        }
        F.reduce_j();
        F.filter_j();
        f2d::k_faraday<<<nb, tb>>>(v, r, float(DT / 2));
        f2d::k_ampere<<<nb, tb>>>(v, r);
        if (n % SEVERY == 0)
            for (int p = 0; p < NPR; ++p)
                CUDA_CHECK(cudaMemcpyAsync(&probe[size_t(p) * NSAMP + n / SEVERY],
                                           F.by.data() + prk * NX + prx[p], 4,
                                           cudaMemcpyDeviceToHost));
        if (n % 50000 == 49999) {
            CUDA_CHECK(cudaDeviceSynchronize());
            double W[2];
            F.energies(W);
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("  step %7ld/%ld (tau_g %4.0f)  W_EM %.3e  "
                        "%.2f ms/step  ETA %.0f min\n",
                        n + 1, NSTEP, (n + 1) * DT * WCE / (2 * M_PI), W[0],
                        1e3 * el / (n + 1), el / (n + 1) * (NSTEP - n) / 60);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    fpar_hist(f1h);

    // dump raw products
    {
        FILE* f = fopen("li3_probe.bin", "wb");
        fwrite(probe.data(), 4, probe.size(), f);
        fclose(f);
        f = fopen("li3_fpar.bin", "wb");
        fwrite(f0h.data(), 8, NB, f);
        fwrite(f1h.data(), 8, NB, f);
        fclose(f);
        std::printf("  wrote li3_probe.bin (%d x %ld f32), li3_fpar.bin\n", NPR, NSAMP);
    }

    // band powers: Hann + DFT over a sample window [s0, s1)
    const double dts = DT * SEVERY;
    auto band_power = [&](long s0, long s1, double w0, double w1) {
        double P = 0;
        const long M = s1 - s0;
        for (double w = w0; w < w1; w += 2 * M_PI / (M * dts))
            for (int p = 0; p < NPR; ++p) {
                double re = 0, im = 0;
                for (long m = 0; m < M; ++m) {
                    const double hann =
                        0.5 * (1 - std::cos(2 * M_PI * m / (M - 1)));
                    const double ph = w * m * dts;
                    re += hann * probe[size_t(p) * NSAMP + s0 + m] * std::cos(ph);
                    im -= hann * probe[size_t(p) * NSAMP + s0 + m] * std::sin(ph);
                }
                P += re * re + im * im;
            }
        return P;
    };
    auto win = [&](double tg0, double tg1, long& s0, long& s1) {
        s0 = long(tg0 * 2 * M_PI / WCE / dts);
        s1 = std::min(NSAMP, long(tg1 * 2 * M_PI / WCE / dts));
    };
    long e0, e1, l0, l1;
    win(200, 250, e0, e1);
    win(890, 1010, l0, l1);
    if (l1 - l0 < 1000) { l0 = NSAMP * 4 / 5; l1 = NSAMP; }   // smoke runs
    if (e1 - e0 < 100) { e0 = 0; e1 = std::max(1L, NSAMP / 5); }
    const double LBe = band_power(e0, e1, 0.2 * WCE, 0.44 * WCE);
    const double GPe = band_power(e0, e1, 0.46 * WCE, 0.54 * WCE);
    const double LBl = band_power(l0, l1, 0.2 * WCE, 0.44 * WCE);
    const double GPl = band_power(l0, l1, 0.46 * WCE, 0.54 * WCE);
    const double UBl = band_power(l0, l1, 0.56 * WCE, 0.70 * WCE);
    std::printf("  bands: early gap/LB %.3f | late gap/LB %.3e, UB/gap %.2f\n",
                GPe / LBe, GPl / LBl, UBl / GPl);

    // saturation amplitude from the late-window probe rms
    double rms = 0;
    for (int p = 0; p < NPR; ++p)
        for (long m = l0; m < l1; ++m) {
            const double v = probe[size_t(p) * NSAMP + m];
            rms += v * v;
        }
    rms = std::sqrt(rms / (NPR * double(l1 - l0))) / WCE;

    // plateau fill-in at |u∥| ∈ [0.08, 0.10]
    auto fill = [&](double a, double b) {
        double n0 = 0, n1 = 0;
        for (int j = 0; j < NB; ++j) {
            const double u = -UMAX + (j + 0.5) * 2 * UMAX / NB;
            if (u >= a && u <= b) { n0 += f0h[j]; n1 += f1h[j]; }
        }
        return n1 / std::max(n0, 1.0);
    };
    const double fplus = fill(0.08, 0.10), fminus = fill(-0.10, -0.08);

    gate("saturation dBy/B0", rms > 0.004 && rms < 0.06, rms, 0.06);
    gate("early gap/LB (no gap yet)", GPe / LBe > 0.3, GPe / LBe, 0.3);
    gate("LATE gap/LB (carved)", GPl / LBl < 1e-2, GPl / LBl, 1e-2);
    gate("warm plateau fill-in", std::min(fplus, fminus) > 2.5,
         std::min(fplus, fminus), 2.5);
    gate("UB distinct above gap", UBl / GPl > 3.0, UBl / GPl, 3.0);

    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
