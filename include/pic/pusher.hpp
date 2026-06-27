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
