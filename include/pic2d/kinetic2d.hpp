// pic2d — P2 kinetic core: (E,μ) shell loader, multi-δf push/deposit,
// field-aligned wall reflection. (PLAN_2D_REBORN §2.2, §2.4, phase P2.)
//
// f₀ contract (per species, u-space, legacy chirp2d convention: momenta u,
// temperatures in u² units): the equatorial bi-Maxwellian written in the
// (adiabatic) invariants of the unperturbed motion,
//     ln f₀ = ln n_sh(L) − (E_u − μ_u B_eq(L))/T∥ − μ_u B_eq(L)/T⊥ + const,
//     E_u = ½|u|²,  μ_u = u⊥²/(2B₀(x,z)),  B_eq(L) = M/L²,
// which reproduces the Chan-1994 mapping (n = n₀/ζ, T⊥ = T⊥eq/ζ,
// ζ = 1 + A_eq(1 − B_eq/B)) along every field line — the same algebra as
// the legacy (E,μ) loader and Lu's Eq. 14–16, plus the raised-cosine shell
// profile n_sh(L) as an explicit factor (L(x,z) = r²/x analytic).
//
// δf weight equation (two-weight lineage): dw/dt = −(1−w)·dlnf₀/dt.
// GYRO-CENTER RULING (P2 first-run lesson, measured): if n_sh depends on
// the PARTICLE L(x,z), every edge marker's weight rings at the gyro-phase
// with amplitude dlnprof·ρ·sec²λ ≈ 0.3–0.5 (the instantaneous L swings by
// ±ρ|∇L|), swamping the δf floor and back-reacting through the deposit
// (measured: wd_rms 1e-3 → 0.55, W_EM floor 6e-3, hot KE +3.3% over 2T_b).
// Therefore f₀'s profile argument is the GUIDING-CENTER L:
//     r_gc = r − γ (u×b̂)/(|qm| B₀)   (in-plane offset carries only u_y:
//     x_gc = x − γ u_y b_z/B₀·s, z_gc = z + γ u_y b_x/B₀·s, s = −sign(qm)),
// an adiabatic invariant to O(ρ²) — its unperturbed drift vanishes, and the
// wave-driven dL_gc/dt (E×B-scale) satisfies |dL_gc/dt·∂_L lnf₀| ≪
// |a_w·∂_E lnf₀·u| for every planned arm, so the explicit ∂_L weight term
// is DROPPED at P2 (pre-registered: reinstate in gc form when cross-L
// transport becomes a measured target, e.g. duct arms). What remains is
// wave-only and gyrophase-clean:
//     dlnf₀/dt = a_w·[∂_E u + ∂_μ u⊥/B₀],
//     ∂_E = −1/T∥, ∂_μ = B_eq(L_gc)(1/T∥ − 1/T⊥), a_w = qm(E_w + v×B_w).
// Identities: a₀·∇_u lnf₀ = 0 exactly; u·(v×B_w) = 0 drops the magnetic
// force from the E-slot. The μ-slot's own position-ripple (μ_u at particle
// B₀) is dT-weighted — zero for isotropic V1; the anisotropic hold is the
// next pre-registered rung (μ_gc treatment if it rings).
//
// Deposit: Esirkepov charge-conserving (CIC/S1), transcribed from the
// validated legacy yee2d.hpp with the axis relabel (legacy in-plane y → z,
// out-of-plane z → y): Wx = ΔSx(Sz0+½ΔSz) prefix-summed in x onto
// Jx(i+½,k); Wz = ΔSz(Sx0+½ΔSx) prefix-summed in z onto Jz(i,k+½);
// Wy = Sx0Sz0 + ½ΔSxSz0 + ½Sx0ΔSz + ⅓ΔSxΔSz onto node Jy — exactly the
// staggering fields2d's Ampère consumes. δf deposits q·w·wd (full-f keeps
// wd ≡ 1, same kernel). Flat atomics (tile-sort = P3 performance item).
//
// Walls: field-aligned reflection u ← u − 2(u·b̂)b̂ about the local
// analytic b̂ — (E,μ)-conserving at any wall inclination, δf-safe.

#ifndef ARC_PIC2D_KINETIC2D_HPP
#define ARC_PIC2D_KINETIC2D_HPP

#include "pic2d/fields2d.hpp"
#include "pic2d/particles2d.hpp"

namespace arc2d {

struct KineticCfg {
    float qm      = -1.f;    // charge/mass (electrons)
    int   dist    = 0;       // 0 = bi-Maxwellian; 1 = bi-kappa (product form,
    float kappa   = 0.f;     //     shared χ²_{2κ−1} factor, integer 2κ−1 ≤ 7,
                             //     |u| capped at 0.6c — the legacy Li lesson;
                             //     mirror (E,μ) mapping for kappa is a P2
                             //     open item: dist=1 requires uniform/tilted)
    float tpar    = 0.f;     // T∥ in u² units (= uthpar²)
    float tperp   = 0.f;     // T⊥eq in u² units
    float n0      = 0.f;     // peak density at shell centre, equator
    float L0      = 0.f;     // shell centre / full width / edge (raised cos)
    float dL      = 0.f;
    float edge    = 0.f;
    int   deltaf  = 1;
    int   rel     = 0;       // relativistic push; weight equation in u
    int   wdfreeze = 0;      // diagnostic: freeze wd at load values
    float taud    = 0.f;     // δf weight relaxation time (0 = off). FORBIDDEN
                             // in H2 gap-memory arms (PLAN_2D v4 ruling);
                             // ALLOWED for strong-drive chirping arms — the
                             // Lu 2021 precedent (τ_D ≈ 2000/Ω_e): without it
                             // trapped-marker weights grow secularly, blow
                             // past |wd| ~ 1 and the deposit runs away
                             // (V4 first-run failure, imaged at step 46500).
    float wdnoise = 0.f;
    float ucap    = 1.0f;    // runaway guard: |u| clamp (counted, reported) —
                             // turns the failure mode into a counter
    float wx0 = 0, wx1 = 0, wz0 = 0, wz1 = 0;   // reflecting wall rectangle
};

namespace k2d {

// portable π-scaled trig (cospif is CUDA-only)
__host__ __device__ inline float cospi_(float x) { return cosf(float(M_PI) * x); }
__host__ __device__ inline float sinpi_(float x) { return sinf(float(M_PI) * x); }

// ---- counter-based RNG (independent streams per marker/purpose) -----------
__host__ __device__ inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
// full hash-combined (id, stream) separation — the earlier id*64+s scheme
// had only 64 clean slots per marker while the rejection loop consumes up
// to ~208, aliasing neighbours' streams and (worse) correlating the kappa
// χ² draws with the position draws. Statistical gates re-run after this
// change (all sampled numbers shift).
__host__ __device__ inline float rng_u01(uint32_t seed, uint32_t id, uint32_t s) {
    return (hash_u32(seed ^ hash_u32(id) ^ (0x9E3779B9u * (s + 1u))) >> 8) *
               (1.f / 16777216.f) +
           (0.5f / 16777216.f);
}
__host__ __device__ inline float rng_gauss(uint32_t seed, uint32_t id, uint32_t s) {
    const float u1 = rng_u01(seed, id, 2u * s), u2 = rng_u01(seed, id, 2u * s + 1u);
    return sqrtf(-2.f * logf(u1)) * cospi_(2.f * u2);
}

// shell profile in L: flat top of full width dL, GAUSSIAN edges of width
// σ = edge. NOT raised-cosine (P2 design correction): the two-weight
// equation needs |d ln prof/dL| bounded where markers exist — the cosine
// edge's tan(πa/2e) diverges at the profile foot and explodes edge-marker
// weights; the Gaussian keeps |dln| ≤ 3/σ over the populated 3σ tail.
__host__ __device__ inline float shell_prof(float L, const KineticCfg& c) {
    const float a = fabsf(L - c.L0) - 0.5f * c.dL;
    if (a <= 0.f) return 1.f;
    return expf(-a * a / (2.f * c.edge * c.edge));
}
__host__ __device__ inline float shell_dlnprof(float L, const KineticCfg& c) {
    const float a = fabsf(L - c.L0) - 0.5f * c.dL;
    if (a <= 0.f) return 0.f;
    return -(a / (c.edge * c.edge)) * copysignf(1.f, L - c.L0);
}

// guiding-center position (in-plane offset; only u_y enters)
__host__ __device__ inline void gc_pos(float x, float z, float uy, float gam,
                                       const KineticCfg& c,
                                       const Background2D& bg,
                                       float& xg, float& zg) {
    const Vec2<float> B = b0_field<float>(bg, x, z);
    const float B2 = B.x * B.x + B.z * B.z;      // γ u_y b̂/|B| = γ u_y B/B²
    const float s = c.qm < 0.f ? 1.f : -1.f;     // −sign(qm)
    xg = x - s * gam * uy * B.z / B2;
    zg = z + s * gam * uy * B.x / B2;
}

// local mapped density factor n/n0 = prof(L_gc)/ζ(λ); the profile argument
// is the gc L (see header ruling), the ζ mapping uses the particle-position
// B (legacy convention; anisotropy rung revisits)
__host__ __device__ inline float density_factor(float x, float z, float uy,
                                                float gam, const KineticCfg& c,
                                                const Background2D& bg) {
    if (B0Prof(bg.prof) != B0Prof::linedipole)
        return 1.f;                     // uniform/tilted arms: no shell, ζ = 1
    float xg, zg;
    gc_pos(x, z, uy, gam, c, bg, xg, zg);
    const float Lg = lshell(xg, zg);
    const float L = lshell(x, z);
    const float Beq = float(bg.M) / (L * L);
    const float b = b0_abs<float>(bg, x, z) / Beq;
    const float Aeq = c.tperp / c.tpar - 1.f;
    const float zeta = 1.f + Aeq * (1.f - 1.f / b);
    return shell_prof(Lg, c) / zeta;
}

// host quadrature of ∫ n/n0 dA (marker-weight norm). The gc offset is a
// shift at fixed u_y — it moves markers, not the integral — so uy = 0 here.
inline double shell_density_integral(const KineticCfg& c, const Background2D& bg,
                                     double x0, double x1, double z0, double z1,
                                     double h) {
    double acc = 0;
    for (double z = z0 + 0.5 * h; z < z1; z += h)
        for (double x = x0 + 0.5 * h; x < x1; x += h)
            acc += density_factor(float(x), float(z), 0.f, 1.f, c, bg);
    return acc * h * h;
}

#ifdef __CUDACC__

// ---- loader ---------------------------------------------------------------
// Equal-weight markers, distributed ∝ n(x,z) by rejection inside the shell
// bounding box (markers exist only where f₀ > 0 — the shell-compact memory
// contract). w = n0·∫(n/n0)dA / N. Velocities: local mapped bi-Maxwellian
// in the field-aligned frame (ê∥ = b̂, ê1 = ŷ×b̂ = (b̂z,0,−b̂x), ê2 = ŷ).
// u_y is drawn BEFORE placement so the position can be accepted against
// prof(L_gc(x,z,u_y)) — the sampled phase-space density then matches
// f₀(E,μ,L_gc) including the gyro-phase structure at the shell edges.
// (u_y's Gaussian width uses the shell-centre ζ; the O(edge) ζ variation
// across the shell is a second-order load residual, V1-measured.)
static __global__ void k_load(MarkerViews p, KineticCfg c, Background2D bg,
                              float bx0, float bx1, float bz0, float bz1,
                              float wmark, uint32_t seed, uint64_t n) {
    const uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float u2 = sqrtf(c.tperp) * rng_gauss(seed, uint32_t(i), 3u);
    float x = 0, z = 0;
    for (uint32_t t = 0; t < 64; ++t) {
        x = bx0 + (bx1 - bx0) * rng_u01(seed, uint32_t(i), 16u + 3u * t);
        z = bz0 + (bz1 - bz0) * rng_u01(seed, uint32_t(i), 17u + 3u * t);
        if (rng_u01(seed, uint32_t(i), 18u + 3u * t) <=
            density_factor(x, z, u2, 1.f, c, bg))
            break;
    }
    p.x[i] = x; p.z[i] = z;
    float zeta = 1.f;                   // uniform/tilted arms: no mapping
    if (B0Prof(bg.prof) == B0Prof::linedipole) {
        const float L = lshell(x, z);
        const float Beq = float(bg.M) / (L * L);
        const float b = b0_abs<float>(bg, x, z) / Beq;
        zeta = 1.f + (c.tperp / c.tpar - 1.f) * (1.f - 1.f / b);
    }
    float upar = sqrtf(c.tpar) * rng_gauss(seed, uint32_t(i), 1u);
    float u1 = sqrtf(c.tperp / zeta) * rng_gauss(seed, uint32_t(i), 2u);
    float u2f = u2;
    if (c.dist == 1) {                  // bi-kappa: shared sqrt(κ/W) factor
        const int ndof = int(2.f * c.kappa - 1.f + 0.5f);
        float W = 0.f;
        for (int j = 0; j < ndof; ++j) {
            const float g = rng_gauss(seed, uint32_t(i), 8u + j);
            W += g * g;
        }
        const float fac = sqrtf(c.kappa / W);
        upar *= fac; u1 *= fac; u2f *= fac;
        const float uu = upar * upar + u1 * u1 + u2f * u2f;
        if (uu > 0.36f) {               // 0.6c cap (legacy superluminal lesson)
            const float r = 0.6f / sqrtf(uu);
            upar *= r; u1 *= r; u2f *= r;
        }
    }
    const Vec2<float> bh = b0_bhat<float>(bg, x, z);
    p.ux[i] = upar * bh.x + u1 * bh.z;
    p.uz[i] = upar * bh.z - u1 * bh.x;
    p.uy[i] = u2f;
    p.w[i]  = wmark;
    p.wd[i] = c.deltaf ? c.wdnoise * rng_gauss(seed, uint32_t(i), 4u) : 1.f;
    p.cell[i] = 0;
}

// ---- fused gather → δf weight → Boris → move/reflect → Esirkepov ----------

__device__ inline float gather(const float* f, const FieldViews2D& v,
                               float gx, float gz) {
    const int i0 = int(floorf(gx)), k0 = int(floorf(gz));
    const float fx = gx - i0, fz = gz - k0;
    return (1.f - fx) * (1.f - fz) * f[v.idx(i0, k0)] +
           fx * (1.f - fz) * f[v.idx(i0 + 1, k0)] +
           (1.f - fx) * fz * f[v.idx(i0, k0 + 1)] +
           fx * fz * f[v.idx(i0 + 1, k0 + 1)];
}

static __global__ void k_push_deposit(MarkerViews p, KineticCfg c,
                                      FieldViews2D v, Background2D bg,
                                      float x0, float z0, uint64_t n,
                                      unsigned long long* runaway = nullptr) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const float x = p.x[m], z = p.z[m];
    const float gx = (x - x0) / v.dx, gz = (z - z0) / v.dz;
    const float Ex = gather(v.ex, v, gx - 0.5f, gz);
    const float Ey = gather(v.ey, v, gx, gz);
    const float Ez = gather(v.ez, v, gx, gz - 0.5f);
    const float Bxw = gather(v.bx, v, gx, gz - 0.5f);
    const float Byw = gather(v.by, v, gx - 0.5f, gz - 0.5f);
    const float Bzw = gather(v.bz, v, gx - 0.5f, gz);
    const Vec2<float> B0 = b0_field<float>(bg, x, z);

    float ux = p.ux[m], uy = p.uy[m], uz = p.uz[m];
    const float gam0 = c.rel ? sqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;

    if (c.deltaf) {
        const float vx = ux / gam0, vy = uy / gam0, vz = uz / gam0;
        const float ax = c.qm * (Ex + vy * Bzw - vz * Byw);
        const float ay = c.qm * (Ey + vz * Bxw - vx * Bzw);
        const float az = c.qm * (Ez + vx * Byw - vy * Bxw);
        const float B0a = sqrtf(B0.x * B0.x + B0.z * B0.z);
        const float bhx = B0.x / B0a, bhz = B0.z / B0a;
        const float upar = ux * bhx + uz * bhz;
        const float upx = ux - upar * bhx, upz = uz - upar * bhz;
        // gc-form weight equation (header ruling): wave-only, ∂_L dropped;
        // uniform/tilted arms have no mapping: B_eq = local B₀ exactly
        float Beq = B0a;
        if (B0Prof(bg.prof) == B0Prof::linedipole) {
            float xg, zg;
            gc_pos(x, z, uy, gam0, c, bg, xg, zg);
            const float Lg = lshell(xg, zg);
            Beq = float(bg.M) / (Lg * Lg);
        }
        const float dE = -1.f / c.tpar;
        const float dMu = Beq * (1.f / c.tpar - 1.f / c.tperp);
        const float udota = ux * ax + uy * ay + uz * az;
        const float updota = upx * ax + uy * ay + upz * az;
        const float dlnf = dE * udota + dMu * updota / B0a;
        if (!c.wdfreeze) {
            float wd = p.wd[m] + -(1.f - p.wd[m]) * dlnf * v.dt;
            if (c.taud > 0.f) wd -= wd * v.dt / c.taud;
            p.wd[m] = wd;
        }
    }

    {   // Boris with total B = wave + analytic background
        const float Bx = Bxw + B0.x, By = Byw, Bz = Bzw + B0.z;
        const float hk = 0.5f * c.qm * v.dt;
        ux += hk * Ex; uy += hk * Ey; uz += hk * Ez;
        const float gam = c.rel ? sqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
        const float f = c.qm * v.dt / (2.f * gam);
        const float tx = f * Bx, ty = f * By, tz = f * Bz;
        const float t2 = tx * tx + ty * ty + tz * tz;
        const float s = 2.f / (1.f + t2);
        const float sx = s * tx, sy = s * ty, sz = s * tz;
        const float px = ux + (uy * tz - uz * ty);
        const float py = uy + (uz * tx - ux * tz);
        const float pz = uz + (ux * ty - uy * tx);
        ux += py * sz - pz * sy;
        uy += pz * sx - px * sz;
        uz += px * sy - py * sx;
        ux += hk * Ex; uy += hk * Ey; uz += hk * Ez;
    }
    {   // runaway guard: a marker past ucap is already unphysical for these
        // arms; clamping (with a reported count) keeps the deposit stencil
        // in the single-wrap range instead of corrupting memory
        const float uu = ux * ux + uy * uy + uz * uz;
        if (uu > c.ucap * c.ucap) {
            const float r = c.ucap * rsqrtf(uu);
            ux *= r; uy *= r; uz *= r;
            if (runaway) atomicAdd(runaway, 1ull);
        }
    }
    const float gam1 = c.rel ? sqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
    float xn = x + ux / gam1 * v.dt;
    float zn = z + uz / gam1 * v.dt;

    // wall test on the GUIDING CENTER (a gyro-excursion of the particle
    // position past the wall is not an escape — flipping u∥ on it would
    // not stop the gyration anyway); u∥-flip reverses the bounce motion,
    // conserving |u| and μ exactly at any wall inclination
    {
        float xg, zg;
        gc_pos(xn, zn, uy, gam1, c, bg, xg, zg);
        // Walls exist ONLY at the high-|λ| line ends (x < wx0 and |z| > wz):
        // in-plane gc drift is identically zero, so no marker can cross L —
        // the outer-radial edge (x > wx1) must NOT be a particle wall.
        // (P2 forensic: testing x > wx1 pinned the shell's outer-tail
        // markers in a flip loop at the equator — z_gc jitters around 0, so
        // "outward" fired every gyro-wobble — creating a coherent antenna
        // layer at (L ≈ 226, λ = 0) that drove a γ ≈ 0.07 Ω_e spurious
        // instability through the live-wd loop. Mode imaged at x = 226,
        // z = 0.8, k ∥ b̂, before the fix.)
        if (xg < c.wx0 || zg < c.wz0 || zg > c.wz1) {
            const Vec2<float> bh = b0_bhat<float>(bg, xn, zn);
            const float up = ux * bh.x + uz * bh.z;
            // "outward" = u∥·sign(z_gc) > 0 (b̂ points north); flip only
            // then, or a just-reflected marker would flip every step
            if (up * (zg >= 0.f ? 1.f : -1.f) > 0.f) {
                ux -= 2.f * up * bh.x;
                uz -= 2.f * up * bh.z;
            }
        }
    }
    // periodic wrap of the STORED position (box = grid extent); the deposit
    // below uses the PRE-wrap worldline — its stencil indices sit at most a
    // few cells outside [0,n) where idx()'s single wrap is exact. Walls, if
    // configured inside the box, fire first and make the wrap a no-op.
    {
        const float Lx = v.nx * v.dx, Lz = v.nz * v.dz;
        p.x[m] = xn - Lx * floorf((xn - x0) / Lx);
        p.z[m] = zn - Lz * floorf((zn - z0) / Lz);
    }
    p.ux[m] = ux; p.uy[m] = uy; p.uz[m] = uz;

    // ---- Esirkepov CIC deposit over the worldline x→xn -------------------
    // block-strided replica choice (contention relief; nrep = 1 ⇒ jx itself)
    const size_t ncell = size_t(v.nx) * v.nz;
    float* JX = v.jxr + size_t(blockIdx.x % v.nrep) * ncell;
    float* JY = v.jyr + size_t(blockIdx.x % v.nrep) * ncell;
    float* JZ = v.jzr + size_t(blockIdx.x % v.nrep) * ncell;
    const float qw = c.qm > 0.f ? p.w[m] * p.wd[m] : -p.w[m] * p.wd[m];
    const float g1x = (xn - x0) / v.dx, g1z = (zn - z0) / v.dz;
    const int ib = int(floorf(fminf(gx, g1x)));
    const int kb = int(floorf(fminf(gz, g1z)));
    float S0x[3], S1x[3], S0z[3], S1z[3], DSx[3], DSz[3];
    for (int a = 0; a < 3; ++a) {
        S0x[a] = fmaxf(0.f, 1.f - fabsf(gx - (ib + a)));
        S1x[a] = fmaxf(0.f, 1.f - fabsf(g1x - (ib + a)));
        S0z[a] = fmaxf(0.f, 1.f - fabsf(gz - (kb + a)));
        S1z[a] = fmaxf(0.f, 1.f - fabsf(g1z - (kb + a)));
        DSx[a] = S1x[a] - S0x[a];
        DSz[a] = S1z[a] - S0z[a];
    }
    const float fx = -qw * v.dx / (v.dt * v.dx * v.dz);  // = −qw/(dt·dz)
    const float fz = -qw * v.dz / (v.dt * v.dx * v.dz);
    const float fy = qw * (uy / gam1) / (v.dx * v.dz);
    for (int b = 0; b < 3; ++b) {                 // Jx(i+½): prefix over a
        float acc = 0.f;
        for (int a = 0; a < 2; ++a) {
            acc += DSx[a] * (S0z[b] + 0.5f * DSz[b]);
            atomicAdd(&JX[v.idx(ib + a, kb + b)], fx * acc);
        }
    }
    for (int a = 0; a < 3; ++a) {                 // Jz(k+½): prefix over b
        float acc = 0.f;
        for (int b = 0; b < 2; ++b) {
            acc += DSz[b] * (S0x[a] + 0.5f * DSx[a]);
            atomicAdd(&JZ[v.idx(ib + a, kb + b)], fz * acc);
        }
    }
    for (int a = 0; a < 3; ++a)                   // node Jy (Wy weights)
        for (int b = 0; b < 3; ++b) {
            const float Wy = S0x[a] * S0z[b] +
                             0.5f * (DSx[a] * S0z[b] + S0x[a] * DSz[b]) +
                             (1.f / 3.f) * DSx[a] * DSz[b];
            atomicAdd(&JY[v.idx(ib + a, kb + b)], fy * Wy);
        }
}

// wd statistics (rms, max) for the δf validity monitor
static __global__ void k_wd_stats(MarkerViews p, double* acc, uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    atomicAdd(&acc[0], double(p.wd[m]) * p.wd[m]);
    // max via integer-encoded atomicMax would need casts; rms suffices here
}

// finiteness scan (debug/validity): acc[0] += count of non-finite markers
static __global__ void k_finite_scan(MarkerViews p, double* acc, uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const float s = p.x[m] + p.z[m] + p.ux[m] + p.uy[m] + p.uz[m] + p.wd[m];
    if (!isfinite(s)) atomicAdd(&acc[0], 1.0);
}

#endif  // __CUDACC__

}  // namespace k2d
}  // namespace arc2d

#endif  // ARC_PIC2D_KINETIC2D_HPP
