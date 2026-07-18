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
#include "pic/fields.hpp"
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

// SharedTileDeposit: each RESIDENT block privatizes the WHOLE deposit grid in
// dynamic shared memory, accumulates its share of particles there (fast shared
// atomics; contention is low because particles are spread over many blocks), then
// flushes once per cell to global. Cuts global atomic traffic from 4*N to
// ncell*nblocks and moves per-particle atomics from L2 to shared memory.
//
// SCOPE / KNOWN-INTERIM: this is whole-grid privatization, so it only works while
// the grid fits the shared opt-in (<99KB on sm_120) AND stays small enough for
// good occupancy. Profiling shows it is a big win for small grids (16KB: 7-34x)
// but DEGRADES as the grid grows — at 64KB only 1 block/SM is resident, so
// parallelism and contention both worsen (128x128: ~5x), and it falls back to the
// global kernel past 99KB. It does NOT scale to large 2D/3D EM grids; the general
// solution is a FIXED-SIZE spatial-tile deposit with particle binning (bounded
// shared footprint regardless of grid size), which will be a separate policy
// plugged in at this same Cfg::deposit seam.
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

// TiledBinnedDeposit: fixed-size SPATIAL-tile deposit (OSIRIS style). A block owns
// one TX*TY-cell tile and privatizes only that tile + a 1-cell halo in shared
// memory — (TX+1)(TY+1) floats, CONSTANT in the grid size. It walks just the
// particles binned to its tile (BinViews from Particles::build_tile_bins), CIC-
// scatters them with fast shared atomics, then flushes its shared tile to global
// rho. Every global cell is contributed to by its home tile plus up to 3 neighbor
// tiles' halos, so the flush is a periodic atomicAdd for all cells; total global
// atomic traffic is ~(TX+1)(TY+1)/(TX*TY) * ncell — independent of ppc and tiny
// next to the 4*N of the global kernel. Unlike SharedTileDeposit this never grows
// past the shared opt-in, so it scales to large 2D/3D grids.
//
// blocks_per_tile lets several blocks share one tile (each privatizes its own copy
// and flushes) so the launch fills the GPU even when ntiles < #SMs (small grids).
template<class Cfg, int TX, int TY>
__global__ void deposit_charge_tiled_kernel(ParticleViews p, BinViews b,
                                            SourceViews src, Grid g, RunParams rp,
                                            int blocks_per_tile) {
    constexpr int SW    = TX + 1;          // shared tile stride (with halo)
    constexpr int SCELL = (TX + 1) * (TY + 1);
    __shared__ float s_rho[SCELL];

    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;

    const int gi0 = (tile % b.ntx) * TX;   // tile origin in global cells
    const int gj0 = (tile / b.ntx) * TY;

    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) s_rho[c] = 0.0f;
    __syncthreads();

    const double qm_over_area = rp.qm / (g.dx * g.dy);   // matches global-kernel coef
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int k = beg + lane * blockDim.x + threadIdx.x; k < end; k += step) {
        const int t = b.idx[k];
        const float x = p.x[t], y = p.y[t];
        const int   i0 = static_cast<int>(x), j0 = static_cast<int>(y);
        const float fx = x - static_cast<float>(i0), fy = y - static_cast<float>(j0);
        const int   li = i0 - gi0, lj = j0 - gj0;        // in [0,TX)x[0,TY)
        const float coef = static_cast<float>(qm_over_area * static_cast<double>(p.w[t]));
        const int base = lj * SW + li;
        // same 4 CIC weights as cic_stencil (grid.hpp) -> bit-matched to the gather
        atomicAdd(&s_rho[base],          coef * (1.0f - fx) * (1.0f - fy));
        atomicAdd(&s_rho[base + 1],      coef *         fx  * (1.0f - fy));
        atomicAdd(&s_rho[base + SW],     coef * (1.0f - fx) *         fy);
        atomicAdd(&s_rho[base + SW + 1], coef *         fx  *         fy);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        const float v = s_rho[c];
        if (v != 0.0f) atomicAdd(&src.rho[g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW))], v);
    }
}

// Darwin fused charge + current deposit (Phase B). Same tiled scatter as above but
// privatizes FOUR fields per cell — ρ and J = q·w·v (Jx,Jy,Jz from ux,uy,uz; γ≡1
// so v=u). One bin walk, one position read, shared tiles 4*(TX+1)(TY+1) floats
// (16x16 -> 4*289*4 = 4.6 KB). Weights are the SAME CIC weights as the gather, so
// ρ matches the ES deposit bit-for-bit and J shares the stencil exactly.
template<class Cfg, int TX, int TY, bool Sorted = false>
__global__ void deposit_rho_j_tiled_kernel(ParticleViews p, BinViews b,
                                           SourceViews src, Grid g, RunParams rp,
                                           int blocks_per_tile) {
    constexpr int SW    = TX + 1;
    constexpr int SCELL = (TX + 1) * (TY + 1);
    __shared__ float s_rho[SCELL];
    __shared__ float s_jx[SCELL];
    __shared__ float s_jy[SCELL];
    __shared__ float s_jz[SCELL];

    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX;
    const int gj0 = (tile / b.ntx) * TY;

    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        s_rho[c] = s_jx[c] = s_jy[c] = s_jz[c] = 0.0f;
    }
    __syncthreads();

    const double qm_over_area = rp.qm / (g.dx * g.dy);
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int k = beg + lane * blockDim.x + threadIdx.x; k < end; k += step) {
        const int t = Sorted ? k : b.idx[k];   // sorted: coalesced direct read
        const float x = p.x[t], y = p.y[t];
        const int   i0 = static_cast<int>(x), j0 = static_cast<int>(y);
        const float fx = x - static_cast<float>(i0), fy = y - static_cast<float>(j0);
        const int   li = i0 - gi0, lj = j0 - gj0;
        const float coef = static_cast<float>(qm_over_area * static_cast<double>(p.w[t]));
        const float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
        const int base = lj * SW + li;
        const float w0 = (1.0f - fx) * (1.0f - fy), w1 = fx * (1.0f - fy);
        const float w2 = (1.0f - fx) * fy,          w3 = fx * fy;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int   off = (e == 0) ? base : (e == 1) ? base + 1
                            : (e == 2) ? base + SW : base + SW + 1;
            const float w   = (e == 0) ? w0 : (e == 1) ? w1 : (e == 2) ? w2 : w3;
            const float cw  = coef * w;
            atomicAdd(&s_rho[off], cw);
            atomicAdd(&s_jx[off],  cw * ux);
            atomicAdd(&s_jy[off],  cw * uy);
            atomicAdd(&s_jz[off],  cw * uz);
        }
    }
    __syncthreads();

    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        const float vr = s_rho[c];
        const float jx = s_jx[c], jy = s_jy[c], jz = s_jz[c];
        if (vr != 0.0f || jx != 0.0f || jy != 0.0f || jz != 0.0f) {
            const int gc = g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW));
            atomicAdd(&src.rho[gc], vr);
            atomicAdd(&src.Jx[gc], jx);
            atomicAdd(&src.Jy[gc], jy);
            atomicAdd(&src.Jz[gc], jz);
        }
    }
}

// Darwin momentum-flux deposit (Phase E). amu = deviatoric momentum flux Π
// (coef qm·w/area = the J coef), 4 components for 2-1/2D so that the transverse
// divergence reconstructs as (∇·Π)_x=∂x amu0+∂y amu1, _y=∂x amu1−∂y amu0,
// _z=∂x amu2+∂y amu3 (the isotropic/trace part is a pure gradient → removed by the
// transverse projection, so only 4 comps are needed):
//   amu0=(vx²−vy²)/2, amu1=vx vy, amu2=vz vx, amu3=vz vy.
// Velocity-only (no field gather) → deposited once per step. Sorted reads.
template<class Cfg, int TX, int TY>
__global__ void deposit_amu_kernel(ParticleViews p, BinViews b, SourceViews src,
                                   Grid g, RunParams rp, int blocks_per_tile) {
    constexpr int SW = TX + 1, SCELL = (TX + 1) * (TY + 1);
    __shared__ float s0[SCELL], s1[SCELL], s2[SCELL], s3[SCELL];
    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX, gj0 = (tile / b.ntx) * TY;
    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) s0[c]=s1[c]=s2[c]=s3[c]=0.0f;
    __syncthreads();
    const double qm_over_area = rp.qm / (g.dx * g.dy);
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int t = beg + lane * blockDim.x + threadIdx.x; t < end; t += step) {
        const float x = p.x[t], y = p.y[t];
        const int i0 = static_cast<int>(x), j0 = static_cast<int>(y);
        const float fx = x - i0, fy = y - j0;
        const int base = (j0 - gj0) * SW + (i0 - gi0);
        const float coef = static_cast<float>(qm_over_area * static_cast<double>(p.w[t]));
        const float vx = p.ux[t], vy = p.uy[t], vz = p.uz[t];
        const float a0 = coef * 0.5f * (vx*vx - vy*vy), a1 = coef * vx*vy;
        const float a2 = coef * vz*vx, a3 = coef * vz*vy;
        const float w0=(1.0f-fx)*(1.0f-fy), w1=fx*(1.0f-fy), w2=(1.0f-fx)*fy, w3=fx*fy;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int off = (e==0)?base:(e==1)?base+1:(e==2)?base+SW:base+SW+1;
            const float w = (e==0)?w0:(e==1)?w1:(e==2)?w2:w3;
            atomicAdd(&s0[off], a0*w); atomicAdd(&s1[off], a1*w);
            atomicAdd(&s2[off], a2*w); atomicAdd(&s3[off], a3*w);
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        if (s0[c]||s1[c]||s2[c]||s3[c]) {
            const int gc = g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW));
            atomicAdd(&src.amu0[gc], s0[c]); atomicAdd(&src.amu1[gc], s1[c]);
            atomicAdd(&src.amu2[gc], s2[c]); atomicAdd(&src.amu3[gc], s3[c]);
        }
    }
}

// Darwin acceleration-density deposit (Phase E). dcu = (q/m)(ρE+J×B); per particle
// = qm²·w/area · (E_total + v×B) with E,B GATHERED at the particle (shared field
// tile). Re-deposited every ndc iteration (E_total changes). Sorted reads.
template<class Cfg, int TX, int TY>
__global__ void deposit_dcu_kernel(ParticleViews p, BinViews b, SourceViews src,
                                   FieldViews f, Grid g, RunParams rp, int blocks_per_tile) {
    constexpr int SW = TX + 1, SCELL = (TX + 1) * (TY + 1);
    __shared__ float sEx[SCELL], sEy[SCELL], sEz[SCELL], sBx[SCELL], sBy[SCELL], sBz[SCELL];
    __shared__ float sdx[SCELL], sdy[SCELL], sdz[SCELL];
    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX, gj0 = (tile / b.ntx) * TY;
    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        const int gc = g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW));
        sEx[c]=f.Ex[gc]; sEy[c]=f.Ey[gc]; sEz[c]=f.Ez[gc];
        sBx[c]=f.Bx[gc]; sBy[c]=f.By[gc]; sBz[c]=f.Bz[gc];
        sdx[c]=sdy[c]=sdz[c]=0.0f;
    }
    __syncthreads();
    const double coef_base = rp.qm * rp.qm / (g.dx * g.dy);   // qm²/area
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int t = beg + lane * blockDim.x + threadIdx.x; t < end; t += step) {
        const float x = p.x[t], y = p.y[t];
        const int i0 = static_cast<int>(x), j0 = static_cast<int>(y);
        const float fx = x - i0, fy = y - j0;
        const int base = (j0 - gj0) * SW + (i0 - gi0);
        const float w0=(1.0f-fx)*(1.0f-fy), w1=fx*(1.0f-fy), w2=(1.0f-fx)*fy, w3=fx*fy;
        #define ARC_G(S) (w0*S[base]+w1*S[base+1]+w2*S[base+SW]+w3*S[base+SW+1])
        const float Ex=ARC_G(sEx), Ey=ARC_G(sEy), Ez=ARC_G(sEz);
        const float Bx=ARC_G(sBx), By=ARC_G(sBy), Bz=ARC_G(sBz);
        #undef ARC_G
        const float vx = p.ux[t], vy = p.uy[t], vz = p.uz[t];
        const float coef = static_cast<float>(coef_base * static_cast<double>(p.w[t]));
        const float ax = coef * (Ex + (vy*Bz - vz*By));
        const float ay = coef * (Ey + (vz*Bx - vx*Bz));
        const float az = coef * (Ez + (vx*By - vy*Bx));
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int off = (e==0)?base:(e==1)?base+1:(e==2)?base+SW:base+SW+1;
            const float w = (e==0)?w0:(e==1)?w1:(e==2)?w2:w3;
            atomicAdd(&sdx[off], ax*w); atomicAdd(&sdy[off], ay*w); atomicAdd(&sdz[off], az*w);
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        if (sdx[c]||sdy[c]||sdz[c]) {
            const int gc = g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW));
            atomicAdd(&src.dcux[gc], sdx[c]); atomicAdd(&src.dcuy[gc], sdy[c]); atomicAdd(&src.dcuz[gc], sdz[c]);
        }
    }
}

template<class Cfg>
struct Depositor {
    static constexpr int kThreads = 256;

    // shared bpt helper for the sorted Darwin deposits (particles already sorted).
    static int bpt_for(int ntiles, int n) {
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int target = 4 * sm;
        int bpt = (ntiles >= target) ? 1 : (target + ntiles - 1) / ntiles;
        const int cap = (n / (ntiles > 0 ? ntiles : 1)) / kThreads;
        if (bpt > cap + 1) bpt = cap + 1;
        return bpt < 1 ? 1 : bpt;
    }

    // Spatial-tile + binning launch (TiledBinnedDeposit). Factored out so the
    // SharedTileDeposit fallback can reuse it when the whole grid no longer fits
    // shared but contention is still high enough for tiling to beat global atomics.
    static void charge_tiled(Particles& parts, Sources& src, const Grid& g,
                             const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        // Fixed tile size (Phase 2 will sweep this). 16x16 -> 17*17*4 = 1156 B
        // shared, so occupancy stays high for any grid.
        constexpr int TX = 16, TY = 16;
        parts.build_tile_bins(g, TX, TY, s);
        const int ntiles = parts.bin_ntiles;
        // Fill the GPU: aim for ~4 blocks/SM. Cap so each block still gets a
        // worthwhile chunk (>= one warp-load of particles) before going wide.
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int target = 4 * sm;
        int bpt = (ntiles >= target) ? 1 : (target + ntiles - 1) / ntiles;
        const int cap = (n / (ntiles > 0 ? ntiles : 1)) / kThreads;   // avg per tile / threads
        if (bpt > cap + 1) bpt = cap + 1;
        if (bpt < 1) bpt = 1;
        deposit_charge_tiled_kernel<Cfg, TX, TY><<<ntiles * bpt, kThreads, 0, s>>>(
            parts.views(), parts.bins(), src.views(), g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Darwin (Phase B): fused ρ + J deposit, same tiled machinery. Caller zeroes
    // ρ,Jx,Jy,Jz first (Sources::zero_rho_j). Builds the tile bins (reused later
    // by the push), then scatters all four fields in one pass.
    static void charge_current(Particles& parts, Sources& src, const Grid& g,
                               const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int TX = 16, TY = 16;
        parts.build_tile_bins(g, TX, TY, s);
        const int ntiles = parts.bin_ntiles;
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int target = 4 * sm;
        int bpt = (ntiles >= target) ? 1 : (target + ntiles - 1) / ntiles;
        const int cap = (n / (ntiles > 0 ? ntiles : 1)) / kThreads;
        if (bpt > cap + 1) bpt = cap + 1;
        if (bpt < 1) bpt = 1;
        deposit_rho_j_tiled_kernel<Cfg, TX, TY, false><<<ntiles * bpt, kThreads, 0, s>>>(
            parts.views(), parts.bins(), src.views(), g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Darwin (Phase D): fused ρ+J deposit on PHYSICALLY-SORTED particles. Sorts the
    // SoA into tile order (coalesced reads), then deposits reading by k directly —
    // no bin_idx indirection. The sort order is reused by the EM push this step.
    static void charge_current_sorted(Particles& parts, Sources& src, const Grid& g,
                                      const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int TX = 16, TY = 16;
        parts.sort_by_tile(g, TX, TY, s);
        const int ntiles = parts.bin_ntiles;
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int target = 4 * sm;
        int bpt = (ntiles >= target) ? 1 : (target + ntiles - 1) / ntiles;
        const int cap = (n / (ntiles > 0 ? ntiles : 1)) / kThreads;
        if (bpt > cap + 1) bpt = cap + 1;
        if (bpt < 1) bpt = 1;
        deposit_rho_j_tiled_kernel<Cfg, TX, TY, true><<<ntiles * bpt, kThreads, 0, s>>>(
            parts.views(), parts.bins(), src.views(), g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Darwin amu (momentum flux) deposit — reuses the existing sort (no re-sort).
    // Caller zeroes amu first (Sources::zero_dcu_amu). Once per step.
    static void deposit_amu(Particles& parts, Sources& src, const Grid& g,
                            const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int TX = 16, TY = 16;
        const int ntiles = parts.bin_ntiles;
        const int bpt = bpt_for(ntiles, n);
        deposit_amu_kernel<Cfg, TX, TY><<<ntiles * bpt, kThreads, 0, s>>>(
            parts.views(), parts.bins(), src.views(), g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Darwin dcu (acceleration density) deposit — gathers E_total,B; reuses sort.
    // Caller zeroes dcu first. Re-run each ndc iteration (E_total changes).
    static void deposit_dcu(Particles& parts, Sources& src, const Fields& flds,
                            const Grid& g, const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int TX = 16, TY = 16;
        const int ntiles = parts.bin_ntiles;
        const int bpt = bpt_for(ntiles, n);
        FieldViews fv = const_cast<Fields&>(flds).views();
        deposit_dcu_kernel<Cfg, TX, TY><<<ntiles * bpt, kThreads, 0, s>>>(
            parts.views(), parts.bins(), src.views(), fv, g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Scatter charge into src.rho. Does NOT zero rho (caller zeroes each step).
    static void charge(Particles& parts, Sources& src, const Grid& g,
                       const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int threads = kThreads;

        if constexpr (std::is_same_v<typename Cfg::deposit, TiledBinnedDeposit>) {
            charge_tiled(parts, src, g, rp, s);
        } else if constexpr (std::is_same_v<typename Cfg::deposit, SharedTileDeposit>) {
            const std::size_t shbytes = static_cast<std::size_t>(g.real_size()) * sizeof(float);
            static const int optin = [] { int v = 0;
                cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0); return v; }();
            if (shbytes > static_cast<std::size_t>(optin)) {
                // Whole-grid privatization no longer fits shared. Pick by contention:
                // the spatial-tile deposit wins while there are enough particles per
                // cell to make atomics serialize (~>=24 ppc, measured crossover); at
                // very low ppc the plain global atomic is already near memory-bound,
                // so its binning overhead is not worth paying.
                const double avg_ppc = static_cast<double>(n) / g.real_size();
                if (avg_ppc >= 24.0) {
                    charge_tiled(parts, src, g, rp, s);
                } else {
                    const int blocks = (n + threads - 1) / threads;
                    deposit_charge_kernel<Cfg, AtomicGlobalDeposit><<<blocks, threads, 0, s>>>(
                        parts.views(), src.views(), g, rp);
                    CUDA_CHECK(cudaPeekAtLastError());
                }
                return;
            }
            auto kern = deposit_charge_shared_kernel<Cfg>;
            if (shbytes > 48u * 1024u) {   // opt in to >48KB dynamic shared (once)
                static bool once = [&] {
                    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(shbytes)); return true; }();
                (void)once;
            }
            // one resident block per available slot -> each holds a private copy.
            // Cache the occupancy-derived block count once (host call is not free;
            // grid/shbytes are fixed within a run).
            static const int resident = [&] {
                int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0);
                int bpsm = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, kern, threads, shbytes);
                return (bpsm * v < 1) ? 1 : bpsm * v;
            }();
            int nblocks = resident;
            const int maxb = (n + threads - 1) / threads;
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
