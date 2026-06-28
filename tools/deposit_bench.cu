// ArcWarden — charge-deposit microbenchmark: AtomicGlobalDeposit vs SharedTileDeposit.
//
// Loads a bump-on-tail-like particle set (noisy, high ppc) and times the deposit
// kernel for both policies, verifying they produce the same rho. Isolates the
// deposit (the 80%-of-GPU-time hot kernel) so the shared-tile privatization win
// is measured directly.
//
//   ./deposit_bench [ppc]    (default 1000 -> ~4.2M particles on 512x8)

#include "pic/depositor.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/sources.hpp"
#include "pic/species.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace arc;
using CfgGlobal = SimConfig<2, 3, ShapeOrder::CIC, float, true, AtomicGlobalDeposit>;
using CfgShared = SimConfig<2, 3, ShapeOrder::CIC, float, true, SharedTileDeposit>;

template<class C>
static double time_charge(Particles& P, Sources& src, const Grid& g,
                          const RunParams& rp, CudaStream& s, int iters) {
    for (int i = 0; i < 5; ++i) { src.zero(s); Depositor<C>::charge(P, src, g, rp, s); }
    s.synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { src.zero(s); Depositor<C>::charge(P, src, g, rp, s); }
    s.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static std::vector<float> deposit_to_host(Particles& P, Sources& src, const Grid& g,
                                          const RunParams& rp, CudaStream& s, bool shared) {
    src.zero(s);
    if (shared) Depositor<CfgShared>::charge(P, src, g, rp, s);
    else        Depositor<CfgGlobal>::charge(P, src, g, rp, s);
    s.synchronize();
    std::vector<float> h(g.real_size());
    CUDA_CHECK(cudaMemcpy(h.data(), src.rho.data(), src.rho.bytes(), cudaMemcpyDeviceToHost));
    return h;
}

int main(int argc, char** argv) {
    const int ppc = (argc > 1) ? std::atoi(argv[1]) : 1000;

    const double Lx = 16.0 * M_PI;
    Grid g(512, 8, Lx, 1.0);
    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.noisy_load = true; rp.rng_seed = 20260627UL;

    SpeciesList sp = { Species{ "bulk", 1.0, ppc, {0.2, 0.2, 0.2}, {0.0, 0.0, 0.0} } };

    CudaStream s;
    Particles P;
    P.initialize(sp, g, rp, s);
    s.synchronize();
    const long long N = static_cast<long long>(P.n);

    Sources src(g);

    // ---- correctness: both policies must agree (up to atomic-order fp noise) ----
    const std::vector<float> rg = deposit_to_host(P, src, g, rp, s, false);
    const std::vector<float> rs = deposit_to_host(P, src, g, rp, s, true);
    double maxabs = 0.0, maxdiff = 0.0;
    for (std::size_t i = 0; i < rg.size(); ++i) {
        maxabs  = std::max(maxabs, std::fabs((double)rg[i]));
        maxdiff = std::max(maxdiff, std::fabs((double)rg[i] - (double)rs[i]));
    }

    // ---- timing ----
    const int iters = 200;
    const double mg = time_charge<CfgGlobal>(P, src, g, rp, s, iters);
    const double ms = time_charge<CfgShared>(P, src, g, rp, s, iters);

    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    std::printf("ArcWarden deposit bench on %s\n", prop.name);
    std::printf("  particles N = %lld   grid %dx%d (ppc=%d)   rho tile = %zu KB shared\n",
                N, g.nx, g.ny, ppc, (std::size_t)g.real_size() * sizeof(float) / 1024);
    std::printf("  correctness: max|rho_global - rho_shared| = %.3g  (rel %.2e)\n",
                maxdiff, maxdiff / (maxabs > 0 ? maxabs : 1.0));
    std::printf("  AtomicGlobalDeposit : %.3f ms/deposit   %.1f G particle-updates/s\n",
                mg, N / mg / 1e6);
    std::printf("  SharedTileDeposit   : %.3f ms/deposit   %.1f G particle-updates/s\n",
                ms, N / ms / 1e6);
    std::printf("  speedup             : %.2fx\n", mg / ms);
    return 0;
}
