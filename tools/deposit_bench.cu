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
#include "pic/solver_darwin.hpp"
#include "pic/sources.hpp"
#include "pic/spectral.hpp"
#include "pic/species.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace arc;
using CfgGlobal = SimConfig<2, 3, ShapeOrder::CIC, float, true,  AtomicGlobalDeposit>;
using CfgShared = SimConfig<2, 3, ShapeOrder::CIC, float, true,  SharedTileDeposit>;
using CfgTiled  = SimConfig<2, 3, ShapeOrder::CIC, float, true,  TiledBinnedDeposit>;
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

enum class Which { Global, Shared, Tiled };
static std::vector<float> deposit_host(Particles& P, Sources& src, const Grid& g,
                                       const RunParams& rp, CudaStream& s, Which w) {
    src.zero(s);
    if      (w == Which::Shared) Depositor<CfgShared>::charge(P, src, g, rp, s);
    else if (w == Which::Tiled)  Depositor<CfgTiled>::charge(P, src, g, rp, s);
    else                         Depositor<CfgGlobal>::charge(P, src, g, rp, s);
    s.synchronize();
    std::vector<float> h(g.real_size());
    CUDA_CHECK(cudaMemcpy(h.data(), src.rho.data(), src.rho.bytes(), cudaMemcpyDeviceToHost));
    return h;
}

static void compare(const char* tag, const std::vector<float>& ref,
                    const std::vector<float>& v) {
    double maxabs = 0, maxdiff = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        maxabs  = std::max(maxabs,  std::fabs((double)ref[i]));
        maxdiff = std::max(maxdiff, std::fabs((double)ref[i] - (double)v[i]));
    }
    std::printf("  correctness %-7s: max|d|=%.2e (rel %.1e)\n",
                tag, maxdiff, maxdiff / (maxabs > 0 ? maxabs : 1));
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
    Fields  fld(g);
    // Non-zero, per-cell-varying field so the gather actually does work (a zeroed
    // field would make plain vs shared-field gather trivially identical).
    {
        std::vector<float> hx(g.real_size()), hy(g.real_size());
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int c = g.idx(i, j);
                hx[c] = 0.1f * std::sin(6.2831853f * i / g.nx);
                hy[c] = 0.1f * std::cos(6.2831853f * j / g.ny);
            }
        CUDA_CHECK(cudaMemcpy(fld.Ex.data(), hx.data(), fld.Ex.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fld.Ey.data(), hy.data(), fld.Ey.bytes(), cudaMemcpyHostToDevice));
    }

    const std::size_t shbytes = (std::size_t)g.real_size() * sizeof(float);
    int optin = 0; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    const bool shared_fits = shbytes <= (std::size_t)optin;

    // correctness: global is the reference; shared & tiled must match it.
    const std::vector<float> rg = deposit_host(P, src, g, rp, s, Which::Global);
    const std::vector<float> rs = deposit_host(P, src, g, rp, s, Which::Shared);
    const std::vector<float> rt = deposit_host(P, src, g, rp, s, Which::Tiled);

    const int it = 200;
    const double mg = time_ms(s, it, [&]{ src.zero(s); Depositor<CfgGlobal>::charge(P, src, g, rp, s); });
    const double ms = time_ms(s, it, [&]{ src.zero(s); Depositor<CfgShared>::charge(P, src, g, rp, s); });
    const double mt = time_ms(s, it, [&]{ src.zero(s); Depositor<CfgTiled>::charge(P, src, g, rp, s); });
    const double mp  = time_ms(s, it, [&]{ Pusher<CfgShared>::boris(P, fld, g, rp, s); });
    const double mp0 = time_ms(s, it, [&]{ Pusher<CfgNoB0>::boris(P, fld, g, rp, s); });
    // PROTOTYPE: shared-field gather push. Bins are reused from the deposit (free
    // in a real loop), so we build them once and time only the push kernel.
    P.build_tile_bins(g, 16, 16, s); s.synchronize();
    const double mpt = time_ms(s, it, [&]{ Pusher<CfgShared>::boris_tiled(P, fld, g, rp, s); });

    // Correctness: plain vs shared-field gather must match (bit-identical by
    // construction). Reload identical particles (deterministic seed) for each.
    auto push_ux = [&](bool tiled) {
        P.initialize(sp, g, rp, s);
        if (tiled) { P.build_tile_bins(g, 16, 16, s); Pusher<CfgShared>::boris_tiled(P, fld, g, rp, s); }
        else       { Pusher<CfgShared>::boris(P, fld, g, rp, s); }
        s.synchronize();
        std::vector<float> h(P.n);
        CUDA_CHECK(cudaMemcpy(h.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost));
        return h;
    };
    const std::vector<float> up_plain = push_ux(false);
    const std::vector<float> up_tiled = push_ux(true);

    // ---- Darwin (EM) baseline costs, pre-Phase-D: fused ρ+J deposit + E_L+B solve ----
    P.initialize(sp, g, rp, s);              // restore particles after push timing
    src.allocate_em(g); fld.allocate_em(g, s);
    SpectralEngine eng(g); eng.enable_em();
    DarwinSpectralSolver dsolver;
    s.synchronize();
    const double mtj    = time_ms(s, it, [&]{ src.zero_rho_j(s);
                                              Depositor<CfgDarwin>::charge_current(P, src, g, rp, s); });
    // Phase D: physically-sorted ρ+J deposit (coalesced) + fused EM push.
    const double mtjs   = time_ms(s, it, [&]{ src.zero_rho_j(s);
                                              Depositor<CfgDarwin>::charge_current_sorted(P, src, g, rp, s); });
    const double mempush = time_ms(s, it, [&]{ Pusher<CfgDarwin>::boris_em(P, fld, g, rp, s); });
    const double msolve = time_ms(s, it, [&]{ dsolver.solve(src, fld, eng, rp, s); });
    // Correctness: sorted deposit (a permutation) must give the same ρ as unsorted.
    std::vector<float> rj_u(g.real_size()), rj_s(g.real_size());
    P.initialize(sp, g, rp, s);
    src.zero_rho_j(s); Depositor<CfgDarwin>::charge_current(P, src, g, rp, s); s.synchronize();
    CUDA_CHECK(cudaMemcpy(rj_u.data(), src.rho.data(), src.rho.bytes(), cudaMemcpyDeviceToHost));
    P.initialize(sp, g, rp, s);
    src.zero_rho_j(s); Depositor<CfgDarwin>::charge_current_sorted(P, src, g, rp, s); s.synchronize();
    CUDA_CHECK(cudaMemcpy(rj_s.data(), src.rho.data(), src.rho.bytes(), cudaMemcpyDeviceToHost));

    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    std::printf("grid %dx%d = %d cells | ppc=%d | N=%lld | rho=%zu KB %s | %s\n",
                nx, ny, g.real_size(), ppc, N, shbytes / 1024,
                shared_fits ? "(fits shared)" : "(TOO BIG -> global fallback)", prop.name);
    std::printf("  deposit global : %7.3f ms  %6.1f G/s\n", mg, N / mg / 1e6);
    std::printf("  deposit shared : %7.3f ms  %6.1f G/s   (%.2fx vs global)\n", ms, N / ms / 1e6, mg / ms);
    std::printf("  deposit tiled  : %7.3f ms  %6.1f G/s   (%.2fx vs global)\n", mt, N / mt / 1e6, mg / mt);
    std::printf("  push global    : %7.3f ms  %6.1f G/s   [plain gather]\n", mp, N / mp / 1e6);
    std::printf("  push  (no B0)  : %7.3f ms  %6.1f G/s   (%.2fx; skips rotation)\n",
                mp0, N / mp0 / 1e6, mp / mp0);
    std::printf("  push shared-fld: %7.3f ms  %6.1f G/s   (%.2fx vs plain)\n",
                mpt, N / mpt / 1e6, mp / mpt);
    std::printf("  dep rho+J(EM)  : %7.3f ms  %6.1f G/s   (unsorted, bin_idx indirection)\n",
                mtj, N / mtj / 1e6);
    std::printf("  dep rho+J sort : %7.3f ms  %6.1f G/s   (%.2fx vs unsorted; incl. sort)\n",
                mtjs, N / mtjs / 1e6, mtj / mtjs);
    std::printf("  em push fused  : %7.3f ms  %6.1f G/s   (%.2fx vs plain ES push)\n",
                mempush, N / mempush / 1e6, mp / mempush);
    std::printf("  darwin solve   : %7.3f ms   (E_L+B: 4 r2c + 2 kernels + 4 c2r)\n", msolve);
    compare("dep shr", rg, rs);
    compare("dep til", rg, rt);
    compare("push fld", up_plain, up_tiled);
    compare("dep sort", rj_u, rj_s);
    return 0;
}
