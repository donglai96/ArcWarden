// P3c gate — checkpoint round-trip on a mini linedipole sim:
// (a) I/O fidelity: save at step 200, load into a fresh sim → marker and
//     field arrays BITWISE identical;
// (b) resumed continuation is statistically equivalent to the straight
//     run: atomics make trajectories run-to-run nondeterministic (legacy
//     M7 lesson), so the gate is an envelope on W_EM at step 400, not
//     bit-identity: ratio ∈ [1/3, 3].

#include "pic2d/checkpoint2d.hpp"

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

static Deck2D mini_deck() {
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
    d.runway_lam = 35.0 * M_PI / 180.0;   // small box: runway must fit
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

static bool bitwise_equal(arc::DeviceArray<float>& a, arc::DeviceArray<float>& b,
                          size_t n) {
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a.data(), n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b.data(), n * 4, cudaMemcpyDeviceToHost));
    return std::memcmp(ha.data(), hb.data(), n * 4) == 0;
}

int main() {
    std::printf("test_ckpt2d — checkpoint round-trip (mini linedipole)\n");
    Deck2D d = mini_deck();
    if (!d.ok()) {
        print_deck2d_report(d);
        std::printf("mini deck refused\n");
        return 1;
    }

    Sim2D A;
    A.build(d);
    for (int i = 0; i < 200; ++i) A.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    save_checkpoint(A, "ckpt2d_test.bin");

    Sim2D B;
    B.build(d);                       // fresh load (different marker state)
    load_checkpoint(B, "ckpt2d_test.bin");
    const bool bits =
        bitwise_equal(A.sp[0].mk->x, B.sp[0].mk->x, A.sp[0].mk->n) &&
        bitwise_equal(A.sp[0].mk->wd, B.sp[0].mk->wd, A.sp[0].mk->n) &&
        bitwise_equal(A.F.ey, B.F.ey, size_t(A.F.nx) * A.F.nz);
    gate("bitwise I/O fidelity", bits, bits ? 0.0 : 1.0, 0.5);
    gate("step counter restored", B.nstep == 200, double(B.nstep), 200);

    for (int i = 0; i < 200; ++i) { A.step(); B.step(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    double WA[2], WB[2];
    A.F.energies(WA);
    B.F.energies(WB);
    const double ratio = WB[0] / WA[0];
    gate("resumed-vs-straight envelope", ratio > 1. / 3 && ratio < 3.0, ratio, 3.0);

    std::remove("ckpt2d_test.bin");
    std::printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
