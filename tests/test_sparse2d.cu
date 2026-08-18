// S-gate — sparse tile-pool storage (PLAN_SPARSE_GRID 方案 2).
//
// Part A (S1, the strongest gate): with EVERY tile active the pool is a
// pure re-addressing of the dense grid, so the whole field pipeline —
// faraday/cold/ampere, masks, both current-filter paths — must produce
// BITWISE identical fields to the dense build. Any tolerance here would
// hide an addressing bug; none is allowed.
//
// Part B (S2, integration): the mini linedipole Sim2D (ckpt-test deck) run
// three ways — dense, sparse-all, sparse-banded. Atomic deposit ordering
// makes trajectories run-to-run nondeterministic (legacy M7 lesson), so
// the gates are envelopes: healthy(), zero dropped deposits, W_EM ratio,
// and the banded pool actually smaller than the dense grid.

#include "pic2d/sim2d.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace arc2d;

static int npass = 0, nfail = 0;
static void gate(const char* name, bool pass, double val, double lim) {
    std::printf("  [%s] %-32s %.3e (limit %.1e)\n", pass ? "PASS" : "FAIL",
                name, val, lim);
    (pass ? npass : nfail)++;
}

// download (pool or dense) and unpack to dense host layout
static std::vector<float> dense_of(Fields2D& F, arc::DeviceArray<float>& a) {
    std::vector<float> h(F.field_cells), d;
    CUDA_CHECK(cudaMemcpy(h.data(), a.data(), F.field_cells * 4,
                          cudaMemcpyDeviceToHost));
    F.unpack_host(h, d);
    return d;
}

static int diff_cells(Fields2D& A, Fields2D& B, arc::DeviceArray<float> Fields2D::*m) {
    const auto a = dense_of(A, A.*m), b = dense_of(B, B.*m);
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) ++bad;
    return bad;
}

static void setup_uniform(Fields2D& F, int nx, int nz, bool sparse) {
    F.dx = F.dz = 0.1; F.dt = 0.05; F.cspeed = 1.0; F.nc = 1.0;
    F.x0 = 0.0; F.z0 = 0.0;
    F.bg.prof = int(B0Prof::tilted);
    F.bg.B0eq = 0.2; F.bg.theta = 25.0 * M_PI / 180.0;
    F.bg.finalize();
    F.nx = nx; F.nz = nz;
    if (sparse) F.build_tiles_all();
    F.allocate(nx, nz);
}

static void upload(Fields2D& F, arc::DeviceArray<float>& a,
                   const std::vector<float>& dense) {
    std::vector<float> p;
    F.pack_host(dense, p);
    CUDA_CHECK(cudaMemcpy(a.data(), p.data(), p.size() * 4,
                          cudaMemcpyHostToDevice));
}

static Deck2D mini_deck(double Lmin, double Lmax) {
    Deck2D d;
    d.bg.prof = int(B0Prof::linedipole);
    d.bg.B0eq = 0.2; d.bg.L0 = 200.0;
    d.bg.finalize();
    d.lam_w = 50.0 * M_PI / 180.0;
    d.margin = 40.0;
    d.dx = d.dz = 0.25;
    d.dt = 0.15;
    d.nc = 1.0;
    d.absorber_cells = 80;
    d.runway_lam = 35.0 * M_PI / 180.0;
    d.active_Lmin = Lmin; d.active_Lmax = Lmax;
    SpeciesCfg s;
    s.name = "engine";
    s.deltaf = true; s.dist = 0;
    s.n0 = 0.01; s.uthpar = 0.14; s.uthperp = 0.14;
    s.shell_L0 = 200; s.shell_dL = 20; s.edge_dL = 5;
    s.ppc = 20; s.wdnoise = 1e-3;
    d.species.push_back(s);
    finalize_deck2d(d);
    return d;
}

int main() {
    std::printf("test_sparse2d — tile-pool storage gates S1/S2\n");

    // ---- Part A: all-active pool vs dense, bitwise --------------------
    const int NX = 512, NZ = 512;
    Fields2D FD, FS;
    setup_uniform(FD, NX, NZ, false);
    setup_uniform(FS, NX, NZ, true);
    std::printf("  pool: %d slots, %zu cells (dense %d)\n", FS.nslots,
                FS.field_cells, NX * NZ);

    std::mt19937 rng(20260818);
    std::normal_distribution<float> gs(0.f, 1e-4f);
    std::vector<float> seed(size_t(NX) * NZ);
    for (auto& v : seed) v = gs(rng);
    upload(FD, FD.by, seed);
    upload(FS, FS.by, seed);

    for (int n = 0; n < 500; ++n) { FD.step(); FS.step(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    int bad = 0;
    for (auto m : {&Fields2D::ex, &Fields2D::ey, &Fields2D::ez, &Fields2D::bx,
                   &Fields2D::by, &Fields2D::bz, &Fields2D::vcx, &Fields2D::vcy,
                   &Fields2D::vcz, &Fields2D::jx, &Fields2D::jy, &Fields2D::jz})
        bad += diff_cells(FD, FS, m);
    gate("A1 step x500 bitwise", bad == 0, bad, 0.5);

    double WD[2], WS[2];
    FD.energies(WD); FS.energies(WS);
    gate("A1 energies agree", std::fabs(WS[0] / WD[0] - 1) < 1e-9,
         std::fabs(WS[0] / WD[0] - 1), 1e-9);

    FD.build_masks(40, 0.05);
    FS.build_masks(40, 0.05);
    for (int n = 0; n < 100; ++n) { FD.step(); FS.step(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    bad = 0;
    for (auto m : {&Fields2D::ex, &Fields2D::ey, &Fields2D::ez, &Fields2D::bx,
                   &Fields2D::by, &Fields2D::bz})
        bad += diff_cells(FD, FS, m);
    gate("A2 masked x100 bitwise", bad == 0, bad, 0.5);

    for (auto m : {&Fields2D::jx, &Fields2D::jy, &Fields2D::jz}) {
        std::vector<float> j(size_t(NX) * NZ);
        for (auto& v : j) v = gs(rng);
        upload(FD, FD.*m, j);
        upload(FS, FS.*m, j);
    }
    FD.jfilter = FS.jfilter = 1;
    FD.filter_j(); FS.filter_j();
    CUDA_CHECK(cudaDeviceSynchronize());
    bad = 0;
    for (auto m : {&Fields2D::jx, &Fields2D::jy, &Fields2D::jz})
        bad += diff_cells(FD, FS, m);
    gate("A3 filter121 bitwise", bad == 0, bad, 0.5);

    FD.jfilter = FS.jfilter = 3;
    FD.filter_j(); FS.filter_j();
    CUDA_CHECK(cudaDeviceSynchronize());
    bad = 0;
    for (auto m : {&Fields2D::jx, &Fields2D::jy, &Fields2D::jz})
        bad += diff_cells(FD, FS, m);
    gate("A4 filter7 bitwise", bad == 0, bad, 0.5);

    // ---- Part B: Sim2D dense / sparse-all / sparse-banded -------------
    auto run = [&](double Lmin, double Lmax, double& wem,
                   unsigned long long& dropped, size_t& cells, bool& ok,
                   int tiled = 1) {
        Deck2D d = mini_deck(Lmin, Lmax);
        if (!d.ok()) { print_deck2d_report(d); ok = false; return; }
        Sim2D S;
        S.deposit_tiled = tiled;
        S.build(d);
        for (int n = 0; n < 300; ++n) S.step();
        CUDA_CHECK(cudaDeviceSynchronize());
        double W[2];
        S.F.energies(W);
        wem = W[0];
        unsigned long long rr[2];
        CUDA_CHECK(cudaMemcpy(rr, S.runaway.data(), 16, cudaMemcpyDeviceToHost));
        dropped = rr[1];
        cells = S.F.field_cells;
        ok = S.healthy();
        if (dropped) {   // forensics: where do markers actually live in L?
            const uint64_t n = S.sp[0].mk->n;
            std::vector<float> hx(n), hz(n);
            CUDA_CHECK(cudaMemcpy(hx.data(), S.sp[0].mk->x.data(), n * 4,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hz.data(), S.sp[0].mk->z.data(), n * 4,
                                  cudaMemcpyDeviceToHost));
            double Lmn = 1e30, Lmx = 0, xmn = 1e30, xmx = -1e30, zmx = 0;
            uint64_t nout = 0;
            for (uint64_t m = 0; m < n; ++m) {
                const double L = lshell_of<double>(d.bg, hx[m], hz[m]);
                Lmn = std::min(Lmn, L); Lmx = std::max(Lmx, L);
                xmn = std::min(xmn, double(hx[m]));
                xmx = std::max(xmx, double(hx[m]));
                zmx = std::max(zmx, std::fabs(double(hz[m])));
                nout += L < Lmin - BAND_MARGIN || L > Lmax + BAND_MARGIN;
            }
            std::printf("  forensics: L [%.1f, %.1f], x [%.1f, %.1f], "
                        "|z| max %.1f, outside pool %llu of %llu\n",
                        Lmn, Lmx, xmn, xmx, zmx,
                        (unsigned long long)nout, (unsigned long long)n);
        }
    };
    double w_d, w_a, w_b, w_f;
    unsigned long long dr_d, dr_a, dr_b, dr_f;
    size_t c_d, c_a, c_b, c_f;
    bool ok_d, ok_a, ok_b, ok_f;
    run(0, 0, w_d, dr_d, c_d, ok_d);              // dense
    run(1e-3, 1e12, w_a, dr_a, c_a, ok_a);        // sparse, every tile
    run(180.0, 220.0, w_b, dr_b, c_b, ok_b);      // sparse, shell band
    run(180.0, 220.0, w_f, dr_f, c_f, ok_f, 0);   // banded, FLAT deposit
    std::printf("  drops: all %llu | banded-tiled %llu | banded-flat %llu\n",
                dr_a, dr_b, dr_f);
    std::printf("  W_EM dense %.3e | sparse-all %.3e | banded %.3e\n", w_d, w_a, w_b);
    std::printf("  cells dense %zu | sparse-all %zu | banded %zu (%.0f%%)\n",
                c_d, c_a, c_b, 100.0 * c_b / c_d);
    gate("B1 healthy x3", ok_d && ok_a && ok_b, ok_d + ok_a + ok_b, 2.5);
    gate("B2 dropped deposits", dr_a + dr_b == 0, double(dr_a + dr_b), 0.5);
    gate("B3 W_EM sparse-all/dense", w_a / w_d > 0.25 && w_a / w_d < 4.0,
         w_a / w_d, 4.0);
    gate("B4 W_EM banded/dense", w_b / w_d > 0.1 && w_b / w_d < 4.0,
         w_b / w_d, 4.0);
    gate("B5 banded pool smaller", c_b < c_d, double(c_b) / c_d, 1.0);

    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
