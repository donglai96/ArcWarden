// ArcWarden — particle push (Step 11).
//
// Pusher<Cfg> advances particle momenta and positions: the "gather" half of the
// grid<->particle coupling (interp E to each particle) followed by the Boris
// velocity update and a leapfrog position move. Like the depositor it is a HOT
// kernel — one thread per particle, no virtual dispatch; the B-field path is a
// compile-time `if constexpr (Cfg::has_b0)`, so a purely electrostatic build pays
// nothing for it (design §8).
//
// Reads Fields (Ex,Ey), takes the Grid (for the CIC stencil + cell-unit moves),
// and uses the SAME cic_stencil as the depositor (grid.hpp) so gather and scatter
// weights are identical (plan §16).
//
// SEMANTICS (design §3): ux,uy,uz are momentum u = γv; v1 is non-relativistic
// (γ≡1) so u = v. Positions are in cell units, so a move is x += vx*dt/dx.
//
// Boris scheme (energy-conserving): half electric kick -> magnetic rotation
// (if has_b0) -> half electric kick. Splitting the velocity update into
// boris_velocity_update(dt_eff) lets us reuse it for the leapfrog half-step
// rollback (Step 9's deferred backstep): dt_eff = -dt/2, no position move.

#ifndef ARC_PIC_PUSHER_HPP
#define ARC_PIC_PUSHER_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"

namespace arc {
namespace detail {

// CIC gather: interpolate (Ex,Ey) at the particle using the shared stencil.
__device__ inline void gather_E(const FieldViews& f, const CicStencil& st,
                                float& Ex, float& Ey) {
    Ex = 0.0f; Ey = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        Ex += st.w[k] * f.Ex[st.cell[k]];
        Ey += st.w[k] * f.Ey[st.cell[k]];
    }
}

// Boris velocity update over an effective time dt_eff (no position move). Half-E,
// optional B rotation, half-E. Ez≡0 in v1 electrostatic. has_b0 is compile-time.
template<class Cfg>
__device__ inline void boris_velocity_update(float& ux, float& uy, float& uz,
                                            float Ex, float Ey,
                                            const RunParams& rp, float dt_eff) {
    const float qmh = static_cast<float>(rp.qm) * 0.5f * dt_eff;   // (q/m)(dt/2)

    // first half electric kick
    ux += qmh * Ex;
    uy += qmh * Ey;
    // uz += qmh * Ez;   (Ez == 0)

    if constexpr (Cfg::has_b0) {
        // rotation vectors t = (q/m)(dt/2) B,  s = 2t/(1+|t|²)
        const float tx = qmh * rp.B0[0];
        const float ty = qmh * rp.B0[1];
        const float tz = qmh * rp.B0[2];
        const float t2 = tx * tx + ty * ty + tz * tz;
        const float sf = 2.0f / (1.0f + t2);
        const float sx = sf * tx, sy = sf * ty, sz = sf * tz;

        // u' = u + u × t
        const float upx = ux + (uy * tz - uz * ty);
        const float upy = uy + (uz * tx - ux * tz);
        const float upz = uz + (ux * ty - uy * tx);
        // u+ = u + u' × s
        ux = ux + (upy * sz - upz * sy);
        uy = uy + (upz * sx - upx * sz);
        uz = uz + (upx * sy - upy * sx);
    }

    // second half electric kick
    ux += qmh * Ex;
    uy += qmh * Ey;
}

// Full Boris with a PER-PARTICLE gathered B (Darwin/EM): half-E → rotate about
// the local B → half-E, over dt (qmh = (q/m)(dt/2)). Same scheme as
// boris_velocity_update but B is the self-consistent gathered field, not rp.B0,
// and Ez participates. Used by the fused EM push.
__device__ inline void boris_update_full(float& ux, float& uy, float& uz,
                                         float Ex, float Ey, float Ez,
                                         float Bx, float By, float Bz, float qmh) {
    ux += qmh * Ex; uy += qmh * Ey; uz += qmh * Ez;          // first half E
    const float tx = qmh * Bx, ty = qmh * By, tz = qmh * Bz;
    const float t2 = tx * tx + ty * ty + tz * tz;
    const float sf = 2.0f / (1.0f + t2);
    const float sx = sf * tx, sy = sf * ty, sz = sf * tz;
    const float upx = ux + (uy * tz - uz * ty);
    const float upy = uy + (uz * tx - ux * tz);
    const float upz = uz + (ux * ty - uy * tx);
    ux += (upy * sz - upz * sy);
    uy += (upz * sx - upx * sz);
    uz += (upx * sy - upy * sx);
    ux += qmh * Ex; uy += qmh * Ey; uz += qmh * Ez;          // second half E
}

// Full leapfrog push: velocity update over dt, then move position (cell units).
template<class Cfg>
__global__ void boris_push_kernel(ParticleViews p, FieldViews f, Grid g, RunParams rp) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    const CicStencil st = cic_stencil(g, p.x[t], p.y[t]);
    float Ex, Ey;
    gather_E(f, st, Ex, Ey);

    float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    boris_velocity_update<Cfg>(ux, uy, uz, Ex, Ey, rp, static_cast<float>(rp.dt));
    p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;

    // move in cell units: dx_cells = v*dt/dx  (v = u, γ≡1)
    p.x[t] += static_cast<float>(ux * rp.dt / g.dx);
    p.y[t] += static_cast<float>(uy * rp.dt / g.dy);
    // wrap + cell recompute is Particles::migrate (called after push in the loop)
}

// Leapfrog initialization backstep: velocity-only Boris over -dt/2 (Step 9 deferred
// rollback). B0=0 reduces to u -= (q/m)(dt/2) E; B0≠0 is the backward half Boris.
template<class Cfg>
__global__ void boris_half_back_kernel(ParticleViews p, FieldViews f, Grid g, RunParams rp) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    const CicStencil st = cic_stencil(g, p.x[t], p.y[t]);
    float Ex, Ey;
    gather_E(f, st, Ex, Ey);

    float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
    boris_velocity_update<Cfg>(ux, uy, uz, Ex, Ey, rp, -0.5f * static_cast<float>(rp.dt));
    p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;
}

// PROTOTYPE (Phase 3 experiment): OSIRIS-style shared-field gather. A block owns
// one TX*TY tile, loads that tile's (Ex,Ey) + 1-cell halo into shared ONCE, then
// every particle binned to the tile gathers from shared instead of global. Reuses
// the BinViews built for the tiled deposit (free in a real loop). Gather is bit-
// identical to the global path (same 4 cells, same weights, same sum order); the
// only change is where the field reads come from. Hypothesis: little ES gain
// because the field already lives in the 96 MB L2 and the push is particle-BW-bound.
template<class Cfg, int TX, int TY>
__global__ void boris_push_tiled_kernel(ParticleViews p, BinViews b, FieldViews f,
                                        Grid g, RunParams rp, int blocks_per_tile) {
    constexpr int SW    = TX + 1;
    constexpr int SCELL = (TX + 1) * (TY + 1);
    __shared__ float s_Ex[SCELL];
    __shared__ float s_Ey[SCELL];

    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX;
    const int gj0 = (tile / b.ntx) * TY;

    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        const int gc = g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW));
        s_Ex[c] = f.Ex[gc];
        s_Ey[c] = f.Ey[gc];
    }
    __syncthreads();

    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int k = beg + lane * blockDim.x + threadIdx.x; k < end; k += step) {
        const int t = b.idx[k];
        const float x = p.x[t], y = p.y[t];
        int i0 = static_cast<int>(x), j0 = static_cast<int>(y);
        const float fx = x - static_cast<float>(i0), fy = y - static_cast<float>(j0);
        int li = i0 - gi0, lj = j0 - gj0;
        // guard: a particle that drifted out of its (stale) tile would index past
        // the shared tile. Clamp so timing never reads OOB; with fresh bins (the
        // correctness path) no particle is ever out of range, so this never fires.
        if (li < 0) li = 0; else if (li >= TX) li = TX - 1;
        if (lj < 0) lj = 0; else if (lj >= TY) lj = TY - 1;
        const int base = lj * SW + li;
        const float w0 = (1.0f - fx) * (1.0f - fy), w1 = fx * (1.0f - fy);
        const float w2 = (1.0f - fx) * fy,          w3 = fx * fy;
        const float Ex = w0 * s_Ex[base] + w1 * s_Ex[base + 1]
                       + w2 * s_Ex[base + SW] + w3 * s_Ex[base + SW + 1];
        const float Ey = w0 * s_Ey[base] + w1 * s_Ey[base + 1]
                       + w2 * s_Ey[base + SW] + w3 * s_Ey[base + SW + 1];

        float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
        boris_velocity_update<Cfg>(ux, uy, uz, Ex, Ey, rp, static_cast<float>(rp.dt));
        p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;
        p.x[t] = x + static_cast<float>(ux * rp.dt / g.dx);
        p.y[t] = y + static_cast<float>(uy * rp.dt / g.dy);
    }
}

// Fused EM push (Phase D): one block per tile loads the E_total(3)+B(3) field tile
// + 1-cell halo into shared ONCE, then pushes the tile's PHYSICALLY-SORTED
// particles (coalesced direct reads, no bin_idx). Gathers E & B from shared, does
// the full Boris with the gathered B, moves. With 6 field components the shared
// field tile pays off (unlike the 2-component ES case where it lost). Requires
// Particles::sort_by_tile already done this step (bin_off valid; same order as the
// ρ+J deposit, and particles have not moved since).
template<class Cfg, int TX, int TY>
__global__ void em_push_tiled_kernel(ParticleViews p, BinViews b, FieldViews f,
                                     Grid g, RunParams rp, int blocks_per_tile) {
    constexpr int SW    = TX + 1;
    constexpr int SCELL = (TX + 1) * (TY + 1);
    __shared__ float sEx[SCELL], sEy[SCELL], sEz[SCELL];
    __shared__ float sBx[SCELL], sBy[SCELL], sBz[SCELL];

    const int tile = blockIdx.x / blocks_per_tile;
    const int lane = blockIdx.x % blocks_per_tile;
    if (tile >= b.ntiles) return;
    const int gi0 = (tile % b.ntx) * TX;
    const int gj0 = (tile / b.ntx) * TY;

    for (int c = threadIdx.x; c < SCELL; c += blockDim.x) {
        const int gc = g.idx_periodic_far(gi0 + (c % SW), gj0 + (c / SW));
        sEx[c] = f.Ex[gc]; sEy[c] = f.Ey[gc]; sEz[c] = f.Ez[gc];
        sBx[c] = f.Bx[gc]; sBy[c] = f.By[gc]; sBz[c] = f.Bz[gc];
    }
    __syncthreads();

    const float qmh = static_cast<float>(rp.qm) * 0.5f * static_cast<float>(rp.dt);
    const int beg = b.off[tile], end = b.off[tile + 1];
    const int step = blocks_per_tile * blockDim.x;
    for (int t = beg + lane * blockDim.x + threadIdx.x; t < end; t += step) {
        const float x = p.x[t], y = p.y[t];
        const int   i0 = static_cast<int>(x), j0 = static_cast<int>(y);
        const float fx = x - static_cast<float>(i0), fy = y - static_cast<float>(j0);
        int li = i0 - gi0, lj = j0 - gj0;
        // no-op when freshly sorted (particle is in its tile); guards the bench's
        // re-use of a stale sort after drift from reading past the shared tile.
        if (li < 0) li = 0; else if (li >= TX) li = TX - 1;
        if (lj < 0) lj = 0; else if (lj >= TY) lj = TY - 1;
        const int   base = lj * SW + li;
        const float w0 = (1.0f - fx) * (1.0f - fy), w1 = fx * (1.0f - fy);
        const float w2 = (1.0f - fx) * fy,          w3 = fx * fy;
        #define ARC_GATHER(S) (w0*S[base] + w1*S[base+1] + w2*S[base+SW] + w3*S[base+SW+1])
        const float Ex = ARC_GATHER(sEx), Ey = ARC_GATHER(sEy), Ez = ARC_GATHER(sEz);
        const float Bx = ARC_GATHER(sBx), By = ARC_GATHER(sBy), Bz = ARC_GATHER(sBz);
        #undef ARC_GATHER

        float ux = p.ux[t], uy = p.uy[t], uz = p.uz[t];
        boris_update_full(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, qmh);
        p.ux[t] = ux; p.uy[t] = uy; p.uz[t] = uz;
        p.x[t] = x + static_cast<float>(ux * rp.dt / g.dx);
        p.y[t] = y + static_cast<float>(uy * rp.dt / g.dy);
    }
}

} // namespace detail

template<class Cfg>
struct Pusher {
    // Full step: gather E -> Boris velocity update (dt) -> move. (migrate after.)
    static void boris(Particles& parts, const Fields& flds, const Grid& g,
                      const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        // FieldViews needs a non-const Fields to call views(); we only read it.
        FieldViews fv = const_cast<Fields&>(flds).views();
        detail::boris_push_kernel<Cfg><<<blocks, threads, 0, s>>>(parts.views(), fv, g, rp);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // PROTOTYPE: shared-field gather push. Requires parts.build_tile_bins() already
    // called (TX=TY=16). One block per tile (x blocks_per_tile to fill the GPU).
    static void boris_tiled(Particles& parts, const Fields& flds, const Grid& g,
                            const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int threads = 256, TX = 16, TY = 16;
        const int ntiles = parts.bin_ntiles;
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int target = 4 * sm;
        int bpt = (ntiles >= target) ? 1 : (target + ntiles - 1) / ntiles;
        const int cap = (n / (ntiles > 0 ? ntiles : 1)) / threads;
        if (bpt > cap + 1) bpt = cap + 1;
        if (bpt < 1) bpt = 1;
        FieldViews fv = const_cast<Fields&>(flds).views();
        detail::boris_push_tiled_kernel<Cfg, TX, TY><<<ntiles * bpt, threads, 0, s>>>(
            parts.views(), parts.bins(), fv, g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Fused EM push (Darwin): gather E_total+B from a shared tile, full Boris with
    // gathered B, move. Requires parts.sort_by_tile() already done this step.
    static void boris_em(Particles& parts, const Fields& flds, const Grid& g,
                         const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int threads = 256, TX = 16, TY = 16;
        const int ntiles = parts.bin_ntiles;
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int target = 4 * sm;
        int bpt = (ntiles >= target) ? 1 : (target + ntiles - 1) / ntiles;
        const int cap = (n / (ntiles > 0 ? ntiles : 1)) / threads;
        if (bpt > cap + 1) bpt = cap + 1;
        if (bpt < 1) bpt = 1;
        FieldViews fv = const_cast<Fields&>(flds).views();
        detail::em_push_tiled_kernel<Cfg, TX, TY><<<ntiles * bpt, threads, 0, s>>>(
            parts.views(), parts.bins(), fv, g, rp, bpt);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Leapfrog half-step rollback at init (after the first field solve): set u at
    // t = -dt/2. Velocity only; no position move. (Step 9's reserved interface.)
    static void half_step_back(Particles& parts, const Fields& flds, const Grid& g,
                               const RunParams& rp, cudaStream_t s) {
        const int n = static_cast<int>(parts.n);
        if (n == 0) return;
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        FieldViews fv = const_cast<Fields&>(flds).views();
        detail::boris_half_back_kernel<Cfg><<<blocks, threads, 0, s>>>(parts.views(), fv, g, rp);
        CUDA_CHECK(cudaPeekAtLastError());
    }
};

} // namespace arc

#endif // ARC_PIC_PUSHER_HPP
