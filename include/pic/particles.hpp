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

#include "pic/background_b0.hpp"   // M4 mirror-equilibrium load (bg::b0x)
#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/grid.hpp"
#include "pic/species.hpp"

#include <cstddef>
#include <stdexcept> // sort_by_tile guard (weight-precision study is flat-path)
#include <utility>   // std::swap (chunk-pool sort buffer swap)
#include <vector>    // initialize_mirror per-cell offsets

namespace arc {

struct ParticleViews {
    DeviceView<float> x, y, ux, uy, uz;
    DeviceView<float> w;        // per-particle macro weight (sets rho scale)
    DeviceView<float> wd;       // M3 delta-f weight (empty unless enable_deltaf)
    DeviceView<int>   cell;
    int n = 0;
    // Weight-precision study accumulators, APPENDED after n on purpose: shorter
    // brace-inits elsewhere value-initialize these to empty (no call-site churn).
    DeviceView<float>  wc;      // Kahan compensation  (df_wprec == 1)
    DeviceView<double> wdd;     // FP64 wd accumulator (df_wprec == 2)
};

// Per-tile particle binning (coarse counting sort) — the index structure the
// TiledBinnedDeposit kernel consumes. `idx` lists particle indices ordered by
// tile; tile `t` owns the slice [off[t], off[t+1]). Geometry (tx,ty,ntx,ntiles)
// is carried so the deposit kernel maps tile id -> origin without a Grid divide.
struct BinViews {
    DeviceView<int> off;     // ntiles+1 prefix offsets (off[0]=0, off[ntiles]=n)
    DeviceView<int> idx;     // n particle indices, grouped by tile
    int ntiles = 0;
    int ntx    = 0;          // tiles along x = ceil(nx/tx)
    int tx = 0, ty = 0;      // tile size in cells
};

namespace detail {

// tile id from a flat cell index (cell = gj*nx + gi).
__host__ __device__ inline int tile_of(int cell, int nx, int tx, int ty, int ntx) {
    const int gi = cell % nx, gj = cell / nx;
    return (gj / ty) * ntx + (gi / tx);
}

// Pass 1 of the counting sort: histogram particles into per-tile counts.
// Privatized: each (grid-stride) block tallies its particles into a shared
// per-tile histogram (ntiles ints), then flushes once per tile to global. This
// is essential — a direct global atomicAdd into the small count[] array funnels
// all N particles onto ntiles addresses and serializes catastrophically.
template<class Dummy = void>
__global__ void bin_histogram_kernel(ParticleViews p, DeviceView<int> count,
                                     int nx, int tx, int ty, int ntx, int ntiles) {
    extern __shared__ int s_cnt[];
    for (int c = threadIdx.x; c < ntiles; c += blockDim.x) s_cnt[c] = 0;
    __syncthreads();
    const int step = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < p.n; t += step)
        atomicAdd(&s_cnt[tile_of(p.cell[t], nx, tx, ty, ntx)], 1);
    __syncthreads();
    for (int c = threadIdx.x; c < ntiles; c += blockDim.x)
        if (s_cnt[c]) atomicAdd(&count[c], s_cnt[c]);
}

// Pass 2: single-block exclusive scan of count[ntiles] -> off[ntiles+1]
// (off[ntiles]=n). Chunked with a carried prefix so any ntiles is handled.
// Also seeds cursor=off for the scatter pass. Launch <<<1, blockDim>>>.
template<class Dummy = void>
__global__ void bin_scan_kernel(DeviceView<int> count, DeviceView<int> off,
                                DeviceView<int> cursor, int ntiles, int n) {
    extern __shared__ int s[];
    int carry = 0;
    for (int base = 0; base < ntiles; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const int v = (i < ntiles) ? count[i] : 0;
        s[threadIdx.x] = v;
        __syncthreads();
        // Hillis-Steele inclusive scan within the chunk.
        for (int d = 1; d < blockDim.x; d <<= 1) {
            const int add = (threadIdx.x >= d) ? s[threadIdx.x - d] : 0;
            __syncthreads();
            s[threadIdx.x] += add;
            __syncthreads();
        }
        if (i < ntiles) {
            const int excl = carry + s[threadIdx.x] - v;   // exclusive prefix
            off[i]    = excl;
            cursor[i] = excl;
        }
        carry += s[blockDim.x - 1];                          // chunk total
        __syncthreads();
    }
    if (threadIdx.x == 0) off[ntiles] = n;
}

// Pass 3: scatter each particle into its tile's slice. Privatized the same way
// as the histogram (a global cursor would re-serialize on ntiles addresses):
//   (A) tally this block's per-tile counts in shared,
//   (B) reserve a contiguous global range per tile (one atomicAdd on cursor),
//   (C) re-walk the SAME particles, handing out slots from the reserved base
//       (shared atomic). The two walks must enumerate identical particles, so the
//       grid-stride mapping is fixed. Order within a tile is irrelevant to deposit.
template<class Dummy = void>
__global__ void bin_scatter_kernel(ParticleViews p, DeviceView<int> cursor,
                                   DeviceView<int> idx, int nx, int tx, int ty,
                                   int ntx, int ntiles) {
    extern __shared__ int s_cnt[];      // (A) local counts, then (B) reserved base
    for (int c = threadIdx.x; c < ntiles; c += blockDim.x) s_cnt[c] = 0;
    __syncthreads();
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int step  = blockDim.x * gridDim.x;
    for (int t = start; t < p.n; t += step)
        atomicAdd(&s_cnt[tile_of(p.cell[t], nx, tx, ty, ntx)], 1);
    __syncthreads();
    for (int c = threadIdx.x; c < ntiles; c += blockDim.x)
        if (s_cnt[c]) s_cnt[c] = atomicAdd(&cursor[c], s_cnt[c]);   // base of our run
    __syncthreads();
    for (int t = start; t < p.n; t += step)
        idx[atomicAdd(&s_cnt[tile_of(p.cell[t], nx, tx, ty, ntx)], 1)] = t;
}

// Physical chunk-pool sort scatter: same privatized counting-sort as above, but
// instead of recording a permutation it MOVES the whole SoA into tile order
// (out arrays). After this the particles of tile `t` occupy the contiguous range
// [off[t], off[t+1]), so the deposit/push read them with COALESCED loads (no
// bin_idx indirection — the penalty that made the ES shared-field gather lose).
// Tile-granularity (coarse) → unlike the removed cell-sort it does not pile up
// same-cell atomics (those go to shared in the tiled kernels).
template<class Dummy = void>
__global__ void tile_sort_scatter_kernel(ParticleViews p, DeviceView<int> cursor,
                                         float* xo, float* yo, float* uxo, float* uyo,
                                         float* uzo, float* wo, float* wdo, int* cello,
                                         int nx, int tx, int ty, int ntx, int ntiles) {
    extern __shared__ int s_cnt[];
    for (int c = threadIdx.x; c < ntiles; c += blockDim.x) s_cnt[c] = 0;
    __syncthreads();
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int step  = blockDim.x * gridDim.x;
    for (int t = start; t < p.n; t += step)
        atomicAdd(&s_cnt[tile_of(p.cell[t], nx, tx, ty, ntx)], 1);
    __syncthreads();
    for (int c = threadIdx.x; c < ntiles; c += blockDim.x)
        if (s_cnt[c]) s_cnt[c] = atomicAdd(&cursor[c], s_cnt[c]);
    __syncthreads();
    for (int t = start; t < p.n; t += step) {
        const int slot = atomicAdd(&s_cnt[tile_of(p.cell[t], nx, tx, ty, ntx)], 1);
        xo[slot]  = p.x[t];  yo[slot]  = p.y[t];
        uxo[slot] = p.ux[t]; uyo[slot] = p.uy[t]; uzo[slot] = p.uz[t];
        wo[slot]  = p.w[t];  cello[slot] = p.cell[t];
        if (wdo) wdo[slot] = p.wd[t];
    }
}

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

    // cell + fraction can ROUND UP to the next integer in float; at the last
    // cell that puts the particle at x == nx (y == ny) exactly, which sends
    // the staggered-gather stencil TWO periods out on 1-cell-thin axes
    // (caught by compute-sanitizer via ez[nx] on an ny = 1 delta-f run).
    float xw = x, yw = y;
    if (xw >= (float)g.nx) xw = nextafterf((float)g.nx, 0.f);
    if (yw >= (float)g.ny) yw = nextafterf((float)g.ny, 0.f);

    p.x[t]    = xw;
    p.y[t]    = yw;
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
    // the += wrap of a tiny negative rounds to nx/ny EXACTLY in float, which
    // puts the staggered gather stencil two periods out on 1-cell-thin axes
    if (x >= nx) x = 0.0f;
    if (y >= ny) y = 0.0f;
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

    // same float round-up guard as particle_init_kernel (x==nx / y==ny edge)
    float xw = x, yw = y;
    if (xw >= (float)g.nx) xw = nextafterf((float)g.nx, 0.f);
    if (yw >= (float)g.ny) yw = nextafterf((float)g.ny, 0.f);

    p.x[t]    = xw;
    p.y[t]    = yw;
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

// M4 mirror-equilibrium loader (chirp1d k_load port to the 2D branch, ny = 1):
// markers sample the (E,mu)-mapped equatorial bi-Maxwellian in the background
// B0(x) profile (background_b0.hpp):
//   Tpar = const,  1/Tperp(x) = (1 - 1/b)/Tpar + (1/b)/Tperp_eq,
//   b(x) = B0(x)/B0eq,  n(x)/n_eq = Tperp(x)/Tperp_eq
// with equal-weight markers (per-cell count ∝ n, offsets prefix `off`, marker
// cell by binary search). Noisy load (hashed RNG positions + Box-Muller
// velocities, chirp1d convention) — quiet mirror load is a future refinement;
// the delta-f noise floor is set by the wd seed, not marker sampling (see
// test_deltaf_growth header).
__global__ void mirror_init_kernel(ParticleViews p, Grid g, RunParams rp,
                                   const int* __restrict__ off,
                                   double uth_par, double uth_perp_eq,
                                   float weight, unsigned long seed) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= p.n) return;
    int lo = 0, hi = g.nx;                 // largest c with off[c] <= t
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (off[mid] <= t) lo = mid; else hi = mid;
    }
    const int c = lo;
    float x = static_cast<float>(c)
            + static_cast<float>(rng_uniform(static_cast<int>(t), 0, seed));
    if (x >= (float)g.nx) x = nextafterf((float)g.nx, 0.f);

    const double b   = (double)bg::b0x(rp, x * (float)g.dx) / (double)rp.B0[0];
    const double Tpa = uth_par * uth_par;
    const double Tpe = uth_perp_eq * uth_perp_eq;
    const double Tp  = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);

    const double r1 = fmax(rng_uniform(static_cast<int>(t), 1, seed), 1e-12);
    const double r2 = rng_uniform(static_cast<int>(t), 2, seed);
    const double r3 = fmax(rng_uniform(static_cast<int>(t), 3, seed), 1e-12);
    const double r4 = rng_uniform(static_cast<int>(t), 4, seed);
    const double upar  = uth_par * sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);
    const double uperp = sqrt(Tp) * sqrt(-2.0 * log(r3));

    p.x[t]    = x;
    p.y[t]    = 0.5f;                      // ny = 1 (validated host-side)
    p.ux[t]   = static_cast<float>(upar);
    p.uy[t]   = static_cast<float>(uperp * cos(2.0 * M_PI * r4));
    p.uz[t]   = static_cast<float>(uperp * sin(2.0 * M_PI * r4));
    p.w[t]    = weight;
    p.cell[t] = g.idx(c, 0);
}

} // namespace detail

struct Particles {
    DeviceArray<float> x, y;        // position (cell units)
    DeviceArray<float> ux, uy, uz;  // momentum u = γv (γ≡1 in v1)
    DeviceArray<float> w;           // per-particle macro weight (rho scale)
    DeviceArray<int>   cell;        // owning cell index
    std::size_t        n = 0;

    // Tile-binning scratch (allocated lazily by build_tile_bins; only the
    // TiledBinnedDeposit path uses these). bin_idx is sized to n; the per-tile
    // arrays to ntiles. Geometry of the last build is cached to skip re-sizing.
    DeviceArray<int> bin_count, bin_off, bin_cursor, bin_idx;
    int bin_ntiles = 0, bin_ntx = 0, bin_tx = 0, bin_ty = 0;
    bool sorted = false;   // true after sort_by_tile (particles physically tile-ordered)

    // Double-buffer scratch for the physical chunk-pool sort (allocated lazily).
    DeviceArray<float> x2, y2, ux2, uy2, uz2, w2;
    DeviceArray<int>   cell2;

    // M3 delta-f weight wd = δf/f at the marker (empty unless enabled).
    DeviceArray<float> wd, wd2;
    bool has_wd = false;

    // Weight-precision study (docs/WEIGHT_PRECISION.md): auxiliary accumulators
    // for df_wprec 1 (Kahan compensation) / 2 (FP64 reference, mirrored to wd
    // for the deposit). FLAT path only — the tile sort does not scatter these.
    DeviceArray<float>  wc;
    DeviceArray<double> wdd;
    int wprec = 0;

    // Allocate + zero the delta-f weights. Call AFTER initialize(); a fresh
    // delta-f load starts exactly on the reference distribution (wd = 0).
    void enable_deltaf(cudaStream_t s, int wprec_ = 0) {
        wd = DeviceArray<float>(n);
        wd.zero(s);
        has_wd = true;
        wprec = wprec_;
        if (wprec == 1) { wc  = DeviceArray<float>(n);  wc.zero(s);  }
        if (wprec == 2) { wdd = DeviceArray<double>(n); wdd.zero(s); }
    }

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
                              w.view(), wd.view(), cell.view(), static_cast<int>(n),
                              wc.view(), wdd.view() };
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

    // M4: mirror-equilibrium load of ONE species in the background B0(x)
    // profile (mirror_init_kernel above). Per-cell marker count ∝ n(x) =
    // Tperp(x)/Tperp_eq (equal weight = density·dx·dy/ppc at the equator);
    // total n is set by the profile, NOT ppc·ncells. Requires ny == 1,
    // rp.b0_prof, gyrotropy uth[1] == uth[2].
    void initialize_mirror(const Species& q, const Grid& g, const RunParams& rp,
                           cudaStream_t s) {
        if (g.ny != 1)   throw std::runtime_error("initialize_mirror: ny must be 1");
        if (!rp.b0_prof) throw std::runtime_error("initialize_mirror: rp.b0_prof required");
        if (q.uth[1] != q.uth[2])
            throw std::runtime_error("initialize_mirror: gyrotropy uth[1] == uth[2] required");
        const double Tpa = q.uth[0] * q.uth[0], Tpe = q.uth[1] * q.uth[1];
        std::vector<int> off(g.nx + 1, 0);
        long tot = 0;
        for (int c = 0; c < g.nx; ++c) {
            const double b  = (double)bg::b0x(rp, (c + 0.5f) * (float)g.dx)
                            / (double)rp.B0[0];
            const double Tp = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);
            off[c] = static_cast<int>(tot);
            tot += std::lround(q.ppc * (Tp / Tpe));
        }
        off[g.nx] = static_cast<int>(tot);
        allocate_n(static_cast<std::size_t>(tot));
        DeviceArray<int> doff(static_cast<std::size_t>(g.nx) + 1);
        CUDA_CHECK(cudaMemcpyAsync(doff.data(), off.data(),
                                   (g.nx + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        constexpr int threads = 256;
        const int blocks = static_cast<int>((tot + threads - 1) / threads);
        detail::mirror_init_kernel<<<blocks, threads, 0, s>>>(
            views(), g, rp, doff.data(), q.uth[0], q.uth[1],
            static_cast<float>(q.density * g.dx * g.dy / q.ppc), rp.rng_seed);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(s));   // doff is scoped to this call
    }

    // Periodic wrap + recompute cell (v1 migrate; chunk-pool reshuffle later).
    void migrate(const Grid& g, cudaStream_t s) {
        if (n == 0) return;
        constexpr int threads = 256;
        const int blocks = (static_cast<int>(n) + threads - 1) / threads;
        detail::particle_migrate_kernel<><<<blocks, threads, 0, s>>>(views(), g);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    BinViews bins() {
        return BinViews{ bin_off.view(), bin_idx.view(),
                         bin_ntiles, bin_ntx, bin_tx, bin_ty };
    }

    // Build the per-tile particle binning (coarse counting sort) the
    // TiledBinnedDeposit kernel consumes. Reads the current cell[] (set by
    // initialize/migrate), so it must run after positions are settled and before
    // deposit. Re-allocates only when the tile geometry or particle count changes.
    void build_tile_bins(const Grid& g, int tx, int ty, cudaStream_t s) {
        if (n == 0) return;
        const int ntx    = (g.nx + tx - 1) / tx;
        const int nty    = (g.ny + ty - 1) / ty;
        const int ntiles = ntx * nty;
        if (ntiles != bin_ntiles || bin_idx.size() != n) {
            bin_count  = DeviceArray<int>(static_cast<std::size_t>(ntiles));
            bin_off    = DeviceArray<int>(static_cast<std::size_t>(ntiles) + 1);
            bin_cursor = DeviceArray<int>(static_cast<std::size_t>(ntiles));
            bin_idx    = DeviceArray<int>(n);
        }
        bin_ntiles = ntiles; bin_ntx = ntx; bin_tx = tx; bin_ty = ty;

        bin_count.zero(s);
        constexpr int threads = 256;
        // Grid-stride with a capped block count so each block aggregates many
        // particles into its shared histogram (fewer blocks -> fewer global flush
        // atomics). ~4 blocks/SM fills the GPU without over-flushing.
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int maxb = (static_cast<int>(n) + threads - 1) / threads;
        const int blocks = (maxb < 4 * sm) ? maxb : 4 * sm;
        const std::size_t shbytes = static_cast<std::size_t>(ntiles) * sizeof(int);

        detail::bin_histogram_kernel<><<<blocks, threads, shbytes, s>>>(
            views(), bin_count.view(), g.nx, tx, ty, ntx, ntiles);
        CUDA_CHECK(cudaPeekAtLastError());

        // single-block chunked scan (counts -> offsets, seeds cursor)
        constexpr int scan_threads = 1024;
        detail::bin_scan_kernel<><<<1, scan_threads, scan_threads * sizeof(int), s>>>(
            bin_count.view(), bin_off.view(), bin_cursor.view(),
            ntiles, static_cast<int>(n));
        CUDA_CHECK(cudaPeekAtLastError());

        detail::bin_scatter_kernel<><<<blocks, threads, shbytes, s>>>(
            views(), bin_cursor.view(), bin_idx.view(), g.nx, tx, ty, ntx, ntiles);
        CUDA_CHECK(cudaPeekAtLastError());
        sorted = false;   // index permutation only; SoA still in original order
    }

    // Physical chunk-pool sort (Phase D): reorder the whole SoA into tile order so
    // the tiled deposit/push read particles with coalesced loads. Produces bin_off
    // (tile offsets); bin_idx is unused on this path (particles ARE in tile order,
    // so kernels read by k directly). Same histogram+scan as build_tile_bins, then
    // the array-moving scatter + a buffer swap.
    void sort_by_tile(const Grid& g, int tx, int ty, cudaStream_t s) {
        if (n == 0) return;
        if (wprec != 0)
            throw std::runtime_error("sort_by_tile: df_wprec > 0 (weight-precision "
                                     "study) is flat-path only — disable tile_sort");
        const int ntx    = (g.nx + tx - 1) / tx;
        const int nty    = (g.ny + ty - 1) / ty;
        const int ntiles = ntx * nty;
        if (ntiles != bin_ntiles || bin_off.size() != static_cast<std::size_t>(ntiles) + 1) {
            bin_count  = DeviceArray<int>(static_cast<std::size_t>(ntiles));
            bin_off    = DeviceArray<int>(static_cast<std::size_t>(ntiles) + 1);
            bin_cursor = DeviceArray<int>(static_cast<std::size_t>(ntiles));
        }
        if (x2.size() != n) {
            x2 = DeviceArray<float>(n); y2 = DeviceArray<float>(n);
            ux2 = DeviceArray<float>(n); uy2 = DeviceArray<float>(n); uz2 = DeviceArray<float>(n);
            w2 = DeviceArray<float>(n); cell2 = DeviceArray<int>(n);
        }
        if (has_wd && wd2.size() != n) wd2 = DeviceArray<float>(n);
        bin_ntiles = ntiles; bin_ntx = ntx; bin_tx = tx; bin_ty = ty;

        bin_count.zero(s);
        constexpr int threads = 256;
        static const int sm = [] { int v = 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
        const int maxb = (static_cast<int>(n) + threads - 1) / threads;
        const int blocks = (maxb < 4 * sm) ? maxb : 4 * sm;
        const std::size_t shbytes = static_cast<std::size_t>(ntiles) * sizeof(int);

        detail::bin_histogram_kernel<><<<blocks, threads, shbytes, s>>>(
            views(), bin_count.view(), g.nx, tx, ty, ntx, ntiles);
        CUDA_CHECK(cudaPeekAtLastError());
        constexpr int scan_threads = 1024;
        detail::bin_scan_kernel<><<<1, scan_threads, scan_threads * sizeof(int), s>>>(
            bin_count.view(), bin_off.view(), bin_cursor.view(), ntiles, static_cast<int>(n));
        CUDA_CHECK(cudaPeekAtLastError());
        detail::tile_sort_scatter_kernel<><<<blocks, threads, shbytes, s>>>(
            views(), bin_cursor.view(),
            x2.data(), y2.data(), ux2.data(), uy2.data(), uz2.data(), w2.data(),
            has_wd ? wd2.data() : nullptr, cell2.data(),
            g.nx, tx, ty, ntx, ntiles);
        CUDA_CHECK(cudaPeekAtLastError());

        std::swap(x, x2);   std::swap(y, y2);
        std::swap(ux, ux2); std::swap(uy, uy2); std::swap(uz, uz2);
        std::swap(w, w2);   std::swap(cell, cell2);
        if (has_wd) std::swap(wd, wd2);
        sorted = true;
    }
};

} // namespace arc

#endif // ARC_PIC_PARTICLES_HPP
