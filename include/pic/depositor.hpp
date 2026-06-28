// ArcWarden — charge deposition (Step 10).
//
// Depositor<Cfg> scatters particle charge onto the grid (rho), the first half of
// the PIC cycle's grid<->particle coupling. This is a HOT kernel — one thread per
// particle, no virtual dispatch; the strategy is a compile-time policy on Cfg
// (design §8). v0 = global atomicAdd (simple, correct); v1/v2 (shared-tile /
// cell-owned) plug in at the marked `if constexpr` seam without touching the
// kernel signature or the main loop.
//
// Writes Sources (rho), NOT Fields — deposit and gather are different ends of the
// pipeline (design §6/§8). Uses the shared cic_stencil (grid.hpp) so its weights
// match the Pusher's gather exactly.
//
// NORMALIZATION (config.hpp contract): each macro-particle carries charge
// q*weight (weight = n0 dx dy / ppc); as a density it adds q*weight/(dx*dy) times
// the CIC weight. With the ωpe=1 normalization (me=1) the elementary charge q is
// rp.qm (= q/m). Caller zeroes rho first (Sources::zero) — deposit accumulates.

#ifndef ARC_PIC_DEPOSITOR_HPP
#define ARC_PIC_DEPOSITOR_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/sources.hpp"

#include <type_traits>

namespace arc {

// Free __global__ kernel, templated on Cfg (vague linkage => ODR-safe in header).
template<class Cfg, class DepositPolicy = typename Cfg::deposit>
__global__ void deposit_charge_kernel(ParticleViews p, SourceViews src,
                                      Grid g, RunParams rp) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    // v1: CIC shape only (Cfg::shape == CIC). TSC would branch here.
    const CicStencil st = cic_stencil(g, p.x[t], p.y[t]);

    // charge density contributed per macro-particle = q*weight / cell area
    // (weight is per-particle so species can carry different densities; q/m global)
    const float coef = static_cast<float>(rp.qm * static_cast<double>(p.w[t]) / (g.dx * g.dy));

    if constexpr (std::is_same_v<DepositPolicy, AtomicGlobalDeposit>) {
        // v0: one particle per thread, global atomicAdd into rho.
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            atomicAdd(&src.rho[st.cell[k]], coef * st.w[k]);
        }
    } else {
        // v1 SharedTileDeposit / v2 CellOwnedDeposit land here later.
        static_assert(std::is_same_v<DepositPolicy, AtomicGlobalDeposit>,
                      "deposit policy not implemented yet");
    }
}

template<class Cfg>
struct Depositor {
    // Scatter charge into src.rho. Does NOT zero rho (caller zeroes each step).
    static void charge(Particles& parts, Sources& src, const Grid& g,
                       const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        deposit_charge_kernel<Cfg><<<blocks, threads, 0, s>>>(
            parts.views(), src.views(), g, rp);
        CUDA_CHECK(cudaPeekAtLastError());
    }
};

} // namespace arc

#endif // ARC_PIC_DEPOSITOR_HPP
