// ArcWarden — particle container + loading (Step 9).
//
// Particles is a flat SoA (structure-of-arrays) store + a per-particle cell
// index — NO chunk pool / no per-step full sort in v1 (plan §7.1). It owns only
// storage / lifetime / loading / migration; deposit and push are NOT methods
// here — they are Depositor<Cfg> / Pusher<Cfg> (design §5/§8). That boundary lets
// a later tile/chunk-pool backend swap in without touching the algorithm layer.
//
//   ParticleViews — POD handle pack (DeviceViews + n) passed BY VALUE to kernels.
//   Particles     — host-side owner (DeviceArray per attribute).
//
// CONVENTIONS (design §3/§4):
//   - positions x,y are in CELL UNITS: x in [0,nx), y in [0,ny). So the CIC cell
//     is floor(x),floor(y) and the fractional weight is x-floor(x) — deposit and
//     gather both read this directly (no dx division in the hot path).
//   - ux,uy,uz store MOMENTUM u = γv; v1 is non-relativistic (γ ≡ 1) so u = v.
//
// QUIET START: positions are loaded stratified (ppc per cell, sub-cell offsets
// from a van der Corput sequence) and velocities by inverting the Maxwellian CDF
// on a low-discrepancy quantile set — a deterministic, low-noise load (vs. raw
// RNG shot noise). The single-k density perturbation hook (for Langmuir / two-
// stream seeding, Steps 13–14) is documented below; v1 default is uniform.
//
// LEAPFROG HALF-STEP ROLLBACK: leapfrog wants u at the half step. The backstep
// (B0=0: a -dt/2 electrostatic kick; B0≠0: backward half Boris) needs the field
// gathered at each particle with the SAME CIC weights as Pusher's gather, so it
// is implemented in Step 11 (Pusher) to avoid duplicating gather (plan §16). The
// interface is reserved here; loading itself is field-free.

#ifndef ARC_PIC_PARTICLES_HPP
#define ARC_PIC_PARTICLES_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/grid.hpp"

#include <cstddef>

namespace arc {

struct ParticleViews {
    DeviceView<float> x, y, ux, uy, uz;
    DeviceView<int>   cell;
    int n = 0;
};

namespace detail {

// Radical inverse (van der Corput) of i in the given base, in (0,1). The basis
// of the quiet (low-discrepancy) load: smooth, reproducible, far less noisy than
// independent RNG draws.
__host__ __device__ inline double radical_inverse(unsigned int i, unsigned int base) {
    double inv = 1.0 / double(base), f = inv, r = 0.0;
    while (i > 0) { r += double(i % base) * f; i /= base; f *= inv; }
    return r;
}

// One particle per thread. Stratified position + quiet Maxwellian velocity.
template<class Dummy = void>
__global__ void particle_init_kernel(ParticleViews p, Grid g, RunParams rp) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    const int ppc = rp.ppc;
    const int c   = t / ppc;          // which cell
    const int s   = t % ppc;          // which particle within the cell
    const int i   = c % g.nx;
    const int j   = c / g.nx;

    // stratified sub-cell offset (van der Corput base 2 / 3) -> fills the cell
    const float fx = static_cast<float>(radical_inverse(s + 1, 2));
    const float fy = static_cast<float>(radical_inverse(s + 1, 3));
    p.x[t]    = static_cast<float>(i) + fx;   // cell units, in [0,nx)
    p.y[t]    = static_cast<float>(j) + fy;
    p.cell[t] = g.idx(i, j);

    // quiet Maxwellian: u = vd + sqrt(2)*vth * erfinv(2q-1), q from low-discrepancy
    const double qx = radical_inverse(t + 1, 2);
    const double qy = radical_inverse(t + 1, 3);
    const double qz = radical_inverse(t + 1, 5);
    const double s2 = 1.41421356237309515 * rp.vth;   // sqrt(2) * vth
    p.ux[t] = static_cast<float>(rp.vd + s2 * erfinv(2.0 * qx - 1.0));
    p.uy[t] = static_cast<float>(        s2 * erfinv(2.0 * qy - 1.0));
    p.uz[t] = static_cast<float>(        s2 * erfinv(2.0 * qz - 1.0));
}

// Periodic wrap of positions (cell units) + recompute the owning cell.
template<class Dummy = void>
__global__ void particle_migrate_kernel(ParticleViews p, Grid g) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n) return;

    const float nx = static_cast<float>(g.nx);
    const float ny = static_cast<float>(g.ny);

    float x = fmodf(p.x[t], nx); if (x < 0.0f) x += nx;   // robust to multi-period moves
    float y = fmodf(p.y[t], ny); if (y < 0.0f) y += ny;
    p.x[t] = x;
    p.y[t] = y;

    int ci = static_cast<int>(floorf(x));
    int cj = static_cast<int>(floorf(y));
    if (ci >= g.nx) ci = g.nx - 1;   // guard the x==nx-eps rounding corner
    if (cj >= g.ny) cj = g.ny - 1;
    p.cell[t] = g.idx(ci, cj);
}

} // namespace detail

struct Particles {
    DeviceArray<float> x, y;        // position (cell units)
    DeviceArray<float> ux, uy, uz;  // momentum u = γv (γ≡1 in v1)
    DeviceArray<int>   cell;        // owning cell index
    std::size_t        n = 0;

    Particles() = default;

    // Allocate for ppc particles per cell (n = ppc * nx * ny).
    void allocate(const Grid& g, int ppc) {
        n   = static_cast<std::size_t>(g.real_size()) * static_cast<std::size_t>(ppc);
        x   = DeviceArray<float>(n);
        y   = DeviceArray<float>(n);
        ux  = DeviceArray<float>(n);
        uy  = DeviceArray<float>(n);
        uz  = DeviceArray<float>(n);
        cell = DeviceArray<int>(n);
    }

    ParticleViews views() {
        return ParticleViews{ x.view(), y.view(), ux.view(), uy.view(), uz.view(),
                              cell.view(), static_cast<int>(n) };
    }

    // Load positions (stratified quiet) + velocities (quiet Maxwellian) + cell.
    // Allocates from rp.ppc if not already sized. Field-free (see header note on
    // the half-step rollback).
    void initialize(const RunParams& rp, const Grid& g, cudaStream_t s) {
        if (n == 0) allocate(g, rp.ppc);
        constexpr int threads = 256;
        const int blocks = (static_cast<int>(n) + threads - 1) / threads;
        detail::particle_init_kernel<><<<blocks, threads, 0, s>>>(views(), g, rp);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Periodic wrap + recompute cell (v1 migrate; chunk-pool reshuffle later).
    void migrate(const Grid& g, cudaStream_t s) {
        if (n == 0) return;
        constexpr int threads = 256;
        const int blocks = (static_cast<int>(n) + threads - 1) / threads;
        detail::particle_migrate_kernel<><<<blocks, threads, 0, s>>>(views(), g);
        CUDA_CHECK(cudaPeekAtLastError());
    }
};

} // namespace arc

#endif // ARC_PIC_PARTICLES_HPP
