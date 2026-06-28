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

// SharedTileDeposit (Tier 1): each RESIDENT block privatizes the deposit grid in
// dynamic shared memory, accumulates its share of particles there (fast shared
// atomics; contention is low because particles are spread over many blocks, so a
// block sees only ~N/nblocks of them), then flushes once per cell to global. The
// shared region is the "tile" — here the whole grid (it fits in <99KB). The same
// structure generalizes to spatial per-block / cluster tiles once particles are
// binned by tile (EM, large grids). Cuts global atomic traffic from 4*N to
// ncell*nblocks and moves per-particle atomics from L2 to shared memory.
template<class Cfg>
__global__ void deposit_charge_shared_kernel(ParticleViews p, SourceViews src,
                                             Grid g, RunParams rp) {
    extern __shared__ float s_rho[];
    const int ncell = g.nx * g.ny;
    for (int c = threadIdx.x; c < ncell; c += blockDim.x) s_rho[c] = 0.0f;
    __syncthreads();

    const double qm_over_area = rp.qm / (g.dx * g.dy);   // matches global-kernel coef
    const int stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < p.n; t += stride) {
        const CicStencil st = cic_stencil(g, p.x[t], p.y[t]);
        const float coef = static_cast<float>(qm_over_area * static_cast<double>(p.w[t]));
        #pragma unroll
        for (int k = 0; k < 4; ++k) atomicAdd(&s_rho[st.cell[k]], coef * st.w[k]);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < ncell; c += blockDim.x) {
        const float v = s_rho[c];
        if (v != 0.0f) atomicAdd(&src.rho[c], v);
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

        if constexpr (std::is_same_v<typename Cfg::deposit, SharedTileDeposit>) {
            const std::size_t shbytes = static_cast<std::size_t>(g.real_size()) * sizeof(float);
            static const int optin = [] { int v = 0;
                cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0); return v; }();
            if (shbytes > static_cast<std::size_t>(optin)) {
                // tile doesn't fit shared yet (needs spatial tiling + binning) -> global atomic
                const int blocks = (n + threads - 1) / threads;
                deposit_charge_kernel<Cfg, AtomicGlobalDeposit><<<blocks, threads, 0, s>>>(
                    parts.views(), src.views(), g, rp);
                CUDA_CHECK(cudaPeekAtLastError());
                return;
            }
            auto kern = deposit_charge_shared_kernel<Cfg>;
            if (shbytes > 48u * 1024u) {   // opt in to >48KB dynamic shared (once)
                static bool once = [&] {
                    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(shbytes)); return true; }();
                (void)once;
            }
            // one resident block per available slot -> each holds a private copy
            static const int sms = [] { int v = 0;
                cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
            int bpsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, kern, threads, shbytes);
            int nblocks = bpsm * sms;
            const int maxb = (n + threads - 1) / threads;
            if (nblocks < 1)     nblocks = 1;
            if (nblocks > maxb)  nblocks = maxb;
            kern<<<nblocks, threads, shbytes, s>>>(parts.views(), src.views(), g, rp);
            CUDA_CHECK(cudaPeekAtLastError());
        } else {
            const int blocks = (n + threads - 1) / threads;
            deposit_charge_kernel<Cfg><<<blocks, threads, 0, s>>>(
                parts.views(), src.views(), g, rp);
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }
};

} // namespace arc

#endif // ARC_PIC_DEPOSITOR_HPP
