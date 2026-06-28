// ArcWarden — deposit + push microbenchmark across grid shapes.
//
// Times the charge deposit (AtomicGlobalDeposit vs SharedTileDeposit, verifying
// they agree) AND the Boris push, for a given grid. Lets us see how the regime
// changes from ~1D (few cells, huge ppc -> deposit atomic-bound) to 2D (many
// cells, low ppc -> low contention; push gather-bound).
//
//   ./deposit_bench [nx] [ny] [ppc]      default 512 8 1000  (~4.1M particles)

#include "pic/depositor.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/pusher.hpp"
#include "pic/sources.hpp"
#include "pic/species.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace arc;
using CfgGlobal = SimConfig<2, 3, ShapeOrder::CIC, float, true,  AtomicGlobalDeposit>;
using CfgShared = SimConfig<2, 3, ShapeOrder::CIC, float, true,  SharedTileDeposit>;
using CfgNoB0   = SimConfig<2, 3, ShapeOrder::CIC, float, false, SharedTileDeposit>;  // skip Boris rotation

template<class F>
static double time_ms(CudaStream& s, int iters, F&& body) {
    for (int i = 0; i < 5; ++i) body();
    s.synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) body();
    s.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static std::vector<float> deposit_host(Particles& P, Sources& src, const Grid& g,
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
    const int nx  = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int ny  = (argc > 2) ? std::atoi(argv[2]) : 8;
    const int ppc = (argc > 3) ? std::atoi(argv[3]) : 1000;

    Grid g(nx, ny, (double)nx, (double)ny);   // dx = dy = 1 (timing only)
    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0; rp.dt = 0.05;
    rp.noisy_load = true; rp.rng_seed = 20260627UL;
    SpeciesList sp = { Species{ "bulk", 1.0, ppc, {0.2, 0.2, 0.2}, {0.0, 0.0, 0.0} } };

    CudaStream s;
    Particles P; P.initialize(sp, g, rp, s); s.synchronize();
    const long long N = (long long)P.n;
    Sources src(g);
    Fields  fld(g); fld.zero(s);

    const std::size_t shbytes = (std::size_t)g.real_size() * sizeof(float);
    int optin = 0; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    const bool shared_fits = shbytes <= (std::size_t)optin;

    // correctness (both should match; if shared falls back they're the same kernel)
    const std::vector<float> rg = deposit_host(P, src, g, rp, s, false);
    const std::vector<float> rs = deposit_host(P, src, g, rp, s, true);
    double maxabs = 0, maxdiff = 0;
    for (std::size_t i = 0; i < rg.size(); ++i) {
        maxabs = std::max(maxabs, std::fabs((double)rg[i]));
        maxdiff = std::max(maxdiff, std::fabs((double)rg[i] - (double)rs[i]));
    }

    const int it = 200;
    const double mg = time_ms(s, it, [&]{ src.zero(s); Depositor<CfgGlobal>::charge(P, src, g, rp, s); });
    const double ms = time_ms(s, it, [&]{ src.zero(s); Depositor<CfgShared>::charge(P, src, g, rp, s); });
    const double mp  = time_ms(s, it, [&]{ Pusher<CfgShared>::boris(P, fld, g, rp, s); });
    const double mp0 = time_ms(s, it, [&]{ Pusher<CfgNoB0>::boris(P, fld, g, rp, s); });

    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    std::printf("grid %dx%d = %d cells | ppc=%d | N=%lld | rho=%zu KB %s | %s\n",
                nx, ny, g.real_size(), ppc, N, shbytes / 1024,
                shared_fits ? "(fits shared)" : "(TOO BIG -> global fallback)", prop.name);
    std::printf("  deposit global : %7.3f ms  %6.1f G/s\n", mg, N / mg / 1e6);
    std::printf("  deposit shared : %7.3f ms  %6.1f G/s   (%.2fx vs global)\n", ms, N / ms / 1e6, mg / ms);
    std::printf("  push  (boris)  : %7.3f ms  %6.1f G/s   [has_b0=true]\n", mp, N / mp / 1e6);
    std::printf("  push  (no B0)  : %7.3f ms  %6.1f G/s   (%.2fx; skips rotation)\n",
                mp0, N / mp0 / 1e6, mp / mp0);
    std::printf("  correctness    : max|dg-ds|=%.2e (rel %.1e)\n", maxdiff, maxdiff / (maxabs > 0 ? maxabs : 1));
    return 0;
}
