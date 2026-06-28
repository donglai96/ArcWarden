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
#include "pic/species.hpp"

#include <cstddef>

namespace arc {

struct ParticleViews {
    DeviceView<float> x, y, ux, uy, uz;
    DeviceView<float> w;        // per-particle macro weight (sets rho scale)
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

// Stateless hashed RNG for the noisy load: a per-particle uniform in (0,1), open
// interval (so erfinv never sees 0 or 1). `stream` picks an independent draw
// (position-x, position-y, vx, vy, vz). This restores genuine shot noise, the
// opposite of radical_inverse — used only when RunParams::noisy_load is set.
__host__ __device__ inline unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
__host__ __device__ inline double rng_uniform(int t, unsigned int stream, unsigned long seed) {
    unsigned int h = hash_u32(static_cast<unsigned int>(t) * 0x9e3779b9U
                              + stream * 0x85ebca6bU
                              + static_cast<unsigned int>(seed));
    return (double(h) + 0.5) * (1.0 / 4294967296.0);   // (0,1), open
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

    // Stratified sub-cell offset (van der Corput base 2 / 3) -> fills the cell.
    // NOTE: every cell deliberately uses the SAME sub-cell pattern. That coherence
    // is what makes the deposited rho perfectly flat (the residual structure lands
    // at the grid-Nyquist scale, which the spectral solver zeros). It also shows
    // up as a faint "comb" if you scatter-plot individual particles — that is a
    // visualization effect, not density structure (see tools/two_stream_movie).
    // sub-cell offset: quiet (van der Corput, flat rho) or noisy (hashed RNG ->
    // physical shot noise, so instabilities self-excite without a perturb seed).
    const float fx = rp.noisy_load ? static_cast<float>(rng_uniform(t, 0, rp.rng_seed))
                                   : static_cast<float>(radical_inverse(s + 1, 2));
    const float fy = rp.noisy_load ? static_cast<float>(rng_uniform(t, 1, rp.rng_seed))
                                   : static_cast<float>(radical_inverse(s + 1, 3));
    float x = static_cast<float>(i) + fx;     // cell units, in [0,nx)
    const float y = static_cast<float>(j) + fy;

    // optional single-k density seed: displace x by amp·sin(2π·kx·x/nx), then wrap.
    if (rp.perturb_amp != 0.0) {
        const float kx = 6.28318530717958648f * rp.perturb_kx / static_cast<float>(g.nx);
        x += static_cast<float>(rp.perturb_amp) * sinf(kx * x);
        x = fmodf(x, static_cast<float>(g.nx));
        if (x < 0.0f) x += static_cast<float>(g.nx);
    }
    int ci = static_cast<int>(x);
    if (ci >= g.nx) ci = g.nx - 1;

    p.x[t]    = x;
    p.y[t]    = y;
    p.w[t]    = static_cast<float>(rp.weight);   // single-species: uniform weight
    p.cell[t] = g.idx(ci, j);

    // quiet Maxwellian: u = drift + sqrt(2)*vth * erfinv(2q-1), q from low-discrepancy.
    // two-stream: alternate beam sign by within-cell parity (co-located, balanced).
    //
    // IMPORTANT: the velocity bases (5,7,11) must differ from the position bases
    // (2,3) AND be coprime to ppc, or position and velocity correlate. With ppc a
    // power of two, using base 2 here makes radical_inverse(t+1,2) ≈ the base-2
    // position offset (the low bits of t = c*ppc+s are s), so vx would track x —
    // the beams come out scalloped instead of uniform. Coprime bases decorrelate.
    const double qx = rp.noisy_load ? rng_uniform(t, 2, rp.rng_seed) : radical_inverse(t + 1, 5);
    const double qy = rp.noisy_load ? rng_uniform(t, 3, rp.rng_seed) : radical_inverse(t + 1, 7);
    const double qz = rp.noisy_load ? rng_uniform(t, 4, rp.rng_seed) : radical_inverse(t + 1, 11);

    // ux population: bump-on-tail (warm bulk + small tail beam) takes priority;
    // else two-stream counter-streaming beams; else a single drifting Maxwellian.
    // Perp directions (uy,uz) always use the bulk width — B0=0 makes them passive.
    double drift_x = rp.vd, width_x = rp.vth;
    if (rp.bump_on_tail) {
        const int  n_beam  = static_cast<int>(rp.beam_frac * ppc + 0.5);
        const bool is_beam = (s < n_beam);
        drift_x = is_beam ? rp.beam_vd  : 0.0;
        width_x = is_beam ? rp.beam_vth : rp.vth;
    } else if (rp.two_stream) {
        drift_x = (s & 1) ? -rp.vd : rp.vd;
    }
    const double s2 = 1.41421356237309515 * rp.vth;   // sqrt(2) * vth (perp dirs)
    p.ux[t] = static_cast<float>(drift_x + 1.41421356237309515 * width_x * erfinv(2.0 * qx - 1.0));
    p.uy[t] = static_cast<float>(s2 * erfinv(2.0 * qy - 1.0));
    p.uz[t] = static_cast<float>(s2 * erfinv(2.0 * qz - 1.0));
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

// POD describing one species' slice for the device init kernel (passed by value).
struct SpeciesInit {
    int           ppc    = 0;
    int           index  = 0;     // species ordinal (decorrelates quiet positions)
    long          base   = 0;     // write offset into the global arrays
    long          count  = 0;     // ppc * ncell
    float         weight = 0.0f;  // macro weight = density·dx·dy/ppc
    double        uth[3] = {0, 0, 0};
    double        ufl[3] = {0, 0, 0};
    bool          noisy  = false;
    unsigned long seed   = 0;
};

// Load one species into [base, base+count). Same position/velocity machinery as
// particle_init_kernel, but drift/width/weight come from the species, not RunParams.
template<class Dummy = void>
__global__ void species_init_kernel(ParticleViews p, Grid g, SpeciesInit sp) {
    const long l = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (l >= sp.count) return;
    const long t = sp.base + l;        // global particle index
    const int  c = static_cast<int>(l / sp.ppc);   // cell within this species
    const int  s = static_cast<int>(l % sp.ppc);   // particle within the cell
    const int  i = c % g.nx;
    const int  j = c / g.nx;

    // sub-cell offset: noisy (hashed RNG, distinct per species since t spans a
    // unique slice) or quiet (van der Corput + a per-species golden-ratio phase so
    // distinct species do NOT load coincident positions — radical_inverse(s+1,·) is
    // otherwise identical across species. The phase is constant within a species, so
    // rho stays flat per species; index 0 is unshifted). Structure stays at Nyquist.
    float fx, fy;
    if (sp.noisy) {
        fx = static_cast<float>(rng_uniform(static_cast<int>(t), 0, sp.seed));
        fy = static_cast<float>(rng_uniform(static_cast<int>(t), 1, sp.seed));
    } else {
        const float px = sp.index * 0.6180339887498949f;   // golden ratio (φ−1)
        const float py = sp.index * 0.7548776662466927f;   // a second irrational
        fx = static_cast<float>(radical_inverse(s + 1, 2)) + px; fx -= floorf(fx);
        fy = static_cast<float>(radical_inverse(s + 1, 3)) + py; fy -= floorf(fy);
    }
    float x = static_cast<float>(i) + fx;
    const float y = static_cast<float>(j) + fy;
    int ci = static_cast<int>(x);
    if (ci >= g.nx) ci = g.nx - 1;

    p.x[t]    = x;
    p.y[t]    = y;
    p.w[t]    = sp.weight;
    p.cell[t] = g.idx(ci, j);

    const double qx = sp.noisy ? rng_uniform(static_cast<int>(t), 2, sp.seed) : radical_inverse(static_cast<unsigned>(t) + 1, 5);
    const double qy = sp.noisy ? rng_uniform(static_cast<int>(t), 3, sp.seed) : radical_inverse(static_cast<unsigned>(t) + 1, 7);
    const double qz = sp.noisy ? rng_uniform(static_cast<int>(t), 4, sp.seed) : radical_inverse(static_cast<unsigned>(t) + 1, 11);
    const double r2 = 1.41421356237309515;   // sqrt(2)
    p.ux[t] = static_cast<float>(sp.ufl[0] + r2 * sp.uth[0] * erfinv(2.0 * qx - 1.0));
    p.uy[t] = static_cast<float>(sp.ufl[1] + r2 * sp.uth[1] * erfinv(2.0 * qy - 1.0));
    p.uz[t] = static_cast<float>(sp.ufl[2] + r2 * sp.uth[2] * erfinv(2.0 * qz - 1.0));
}

} // namespace detail

struct Particles {
    DeviceArray<float> x, y;        // position (cell units)
    DeviceArray<float> ux, uy, uz;  // momentum u = γv (γ≡1 in v1)
    DeviceArray<float> w;           // per-particle macro weight (rho scale)
    DeviceArray<int>   cell;        // owning cell index
    std::size_t        n = 0;

    Particles() = default;

    // Allocate for n_total particles (caller sets the count).
    void allocate_n(std::size_t n_total) {
        n   = n_total;
        x   = DeviceArray<float>(n);
        y   = DeviceArray<float>(n);
        ux  = DeviceArray<float>(n);
        uy  = DeviceArray<float>(n);
        uz  = DeviceArray<float>(n);
        w   = DeviceArray<float>(n);
        cell = DeviceArray<int>(n);
    }

    // Allocate for ppc particles per cell (n = ppc * nx * ny).
    void allocate(const Grid& g, int ppc) {
        allocate_n(static_cast<std::size_t>(g.real_size()) * static_cast<std::size_t>(ppc));
    }

    ParticleViews views() {
        return ParticleViews{ x.view(), y.view(), ux.view(), uy.view(), uz.view(),
                              w.view(), cell.view(), static_cast<int>(n) };
    }

    // Load positions (stratified quiet) + velocities (quiet Maxwellian) + cell.
    // Allocates from rp.ppc if not already sized. Field-free (see header note on
    // the half-step rollback). Single-species legacy path (uniform weight).
    void initialize(const RunParams& rp, const Grid& g, cudaStream_t s) {
        if (n == 0) allocate(g, rp.ppc);
        constexpr int threads = 256;
        const int blocks = (static_cast<int>(n) + threads - 1) / threads;
        detail::particle_init_kernel<><<<blocks, threads, 0, s>>>(views(), g, rp);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Multi-species load (OSIRIS/UPIC style): each species fills a contiguous slice
    // of the arrays with its own count, drift (ufl), thermal spread (uth) and macro
    // weight (= density·dx·dy/ppc). rp supplies the global normalization + noisy_load
    // / rng_seed. Two-stream / bump-on-tail are expressed purely as species lists.
    void initialize(const SpeciesList& sp, const Grid& g, const RunParams& rp,
                    cudaStream_t s) {
        const long ncell = static_cast<long>(g.real_size());
        std::size_t total = 0;
        for (const auto& q : sp) total += static_cast<std::size_t>(q.ppc) * ncell;
        if (total == 0) return;
        allocate_n(total);

        long base = 0;
        int  idx  = 0;
        for (const auto& q : sp) {
            detail::SpeciesInit si{};
            si.ppc    = q.ppc;
            si.index  = idx++;
            si.base   = base;
            si.count  = static_cast<long>(q.ppc) * ncell;
            si.weight = static_cast<float>(q.density * g.dx * g.dy / q.ppc);
            for (int d = 0; d < 3; ++d) { si.uth[d] = q.uth[d]; si.ufl[d] = q.ufl[d]; }
            si.noisy = rp.noisy_load;
            si.seed  = rp.rng_seed;
            constexpr int threads = 256;
            const int blocks = static_cast<int>((si.count + threads - 1) / threads);
            detail::species_init_kernel<><<<blocks, threads, 0, s>>>(views(), g, si);
            CUDA_CHECK(cudaPeekAtLastError());
            base += si.count;
        }
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
