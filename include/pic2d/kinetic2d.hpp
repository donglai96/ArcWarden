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
    float wdrms_max = 0.f;   // healthy(): fail if wd_rms exceeds (0 = off)
    float ucap    = 4.0f;    // runaway DISASTER guard only: |u| clamp with a
                             // reported count. Must sit far above the
                             // physical thermal tail — the first V4 rerun
                             // at ucap=1.0 clipped ~5.7k genuine tail
                             // markers per step (u_perp=0.345c Maxwellian),
                             // eroding f0 and sourcing spurious wd. Load-
                             // time caps (kappa 0.6c) handle distributions.
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

// guiding-center position (in-plane offset; only u_y enters). u is
// MOMENTUM (u = γv), so ρ = p⊥/(|q|B) = u_y/B with no extra γ — the
// γ factor removed 2026-08-19 (review blocker 3: it was a velocity-
// contract holdover; loader passed γ=1 and was numerically right,
// runtime passed γ and overshot the offset by (γ−1)ρ ≈ 0.05 c/ωpe).
__host__ __device__ inline void gc_pos(float x, float z, float uy,
                                       const KineticCfg& c,
                                       const Background2D& bg,
                                       float& xg, float& zg) {
    const Vec2<float> B = b0_field<float>(bg, x, z);
    const float B2 = B.x * B.x + B.z * B.z;      // u_y b̂/|B| = u_y B/B²
    const float s = c.qm < 0.f ? 1.f : -1.f;     // −sign(qm)
    xg = x - s * uy * B.z / B2;
    zg = z + s * uy * B.x / B2;
}

// local mapped density factor n/n0 = prof(L_gc)/ζ(λ); the profile argument
// is the gc L (see header ruling), the ζ mapping uses the particle-position
// B (legacy convention; anisotropy rung revisits)
__host__ __device__ inline float density_factor(float x, float z, float uy,
                                                const KineticCfg& c,
                                                const Background2D& bg) {
    if (!has_lines(bg))
        return 1.f;                     // uniform/tilted arms: no shell, ζ = 1
    float xg, zg;
    gc_pos(x, z, uy, c, bg, xg, zg);
    const float Lg = lshell_of<float>(bg, xg, zg);
    const float L = lshell_of<float>(bg, x, z);
    const float Beq = float(beq_of(bg, L));
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
            acc += density_factor(float(x), float(z), 0.f, c, bg);
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
                              float wmark, uint32_t seed, uint64_t n,
                              unsigned long long* exhaust) {
    const uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float u2 = sqrtf(c.tperp) * rng_gauss(seed, uint32_t(i), 3u);
    // Rejection sampling. The old 64-attempt loop KEPT the last rejected
    // candidate on exhaust — at few-% mean acceptance (big loader boxes)
    // that silently loaded a percent-class population ∝ uniform instead of
    // ∝ f₀ (found 2026-08-19 by the sparse-pool dropped-deposit gate: two
    // strays at L ≈ 400 where prof = e⁻⁷²²). Attempts 0–63 keep their
    // original RNG slots (bit-preserving the accepted 1−ε of every prior
    // load); attempts 64+ draw from slots 4096+ (clear of the kappa χ²
    // streams at 210+ — the collision class fixed in the 08-18 review).
    // True exhaust (P < 1e-40 at any sane acceptance) collapses to the
    // shell centre (f₀ > 0 by construction) and is counted.
    float x = 0, z = 0;
    bool accepted = false;
    for (uint32_t t = 0; t < 1024 && !accepted; ++t) {
        const uint32_t s0 = t < 64 ? 16u + 3u * t : 4096u + 3u * (t - 64u);
        x = bx0 + (bx1 - bx0) * rng_u01(seed, uint32_t(i), s0);
        z = bz0 + (bz1 - bz0) * rng_u01(seed, uint32_t(i), s0 + 1u);
        accepted = rng_u01(seed, uint32_t(i), s0 + 2u) <=
                   density_factor(x, z, u2, c, bg);
    }
    if (!accepted) {
        x = fminf(fmaxf(c.L0, bx0), bx1);
        z = 0.f;
        if (exhaust) atomicAdd(exhaust, 1ull);
    }
    p.x[i] = x; p.z[i] = z;
    float zeta = 1.f;                   // uniform/tilted arms: no mapping
    if (has_lines(bg)) {
        const float L = lshell_of<float>(bg, x, z);
        const float Beq = float(beq_of(bg, L));
        const float b = b0_abs<float>(bg, x, z) / Beq;
        zeta = 1.f + (c.tperp / c.tpar - 1.f) * (1.f - 1.f / b);
    }
    float upar = sqrtf(c.tpar) * rng_gauss(seed, uint32_t(i), 1u);
    float u1 = sqrtf(c.tperp / zeta) * rng_gauss(seed, uint32_t(i), 2u);
    // GYROTROPY (2026-08-19 review): rescale u_y to the same local width
    // as u1. The pre-draw kept the equatorial width so the rejection could
    // use u2's gyro offset; the offset-scale mismatch left behind is
    // (1−ζ^{−1/2})ρ ≲ 0.5 L-units against the σ=8 edge — second order,
    // unlike the (1−1/ζ) variance error this fixes.
    float u2f = u2 / sqrtf(zeta);
    if (c.dist == 1) {                  // bi-kappa: shared sqrt(κ/W) factor
        const int ndof = int(2.f * c.kappa - 1.f + 0.5f);
        float W = 0.f;
        // gauss streams >= 105 (u01 slots >= 210): the position-rejection
        // loop consumes slots 16..207 — streams 8..14 COLLIDED with it,
        // correlating each marker's position with its chi^2 speed factor
        // (bug found in the 2026-08-18 code review; the V3 Li run carried
        // it — integrated verdicts robust, recorded in the review doc).
        for (int j = 0; j < ndof; ++j) {
            const float g = rng_gauss(seed, uint32_t(i), 105u + j);
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
    return (1.f - fx) * (1.f - fz) * v.ld(f, i0, k0) +
           fx * (1.f - fz) * v.ld(f, i0 + 1, k0) +
           (1.f - fx) * fz * v.ld(f, i0, k0 + 1) +
           fx * fz * v.ld(f, i0 + 1, k0 + 1);
}

// ---- single-marker physics, shared by the flat and tiled kernels ----------
// Everything except the deposit target: gather → δf weight → Boris →
// runaway guard → wall reflect → periodic wrap. Returns the pre-wrap
// worldline in grid units + the deposit factors. ONE implementation —
// the two kernels cannot diverge physically.
struct PushOut {
    float g0x, g0z, g1x, g1z;   // worldline in grid units (pre-wrap end)
    float qw, vy1;              // signed charge·weight, v_y after push
};

__device__ inline PushOut push_move_one(MarkerViews& p, uint64_t m,
                                        const KineticCfg& c,
                                        const FieldViews2D& v,
                                        const Background2D& bg, float x0,
                                        float z0,
                                        unsigned long long* runaway) {
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
    const float u0x = ux, u0y = uy, u0z = uz;   // pre-push momentum (for the
                                                // time-centred weight update)

    {   // Boris with total B = wave + analytic background
        const float Bx = Bxw + B0.x, By = Byw, Bz = Bzw + B0.z;
        const float hk = 0.5f * c.qm * v.dt;
        ux += hk * Ex; uy += hk * Ey; uz += hk * Ez;
        const float gam = c.rel ? sqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;
        const float f = c.qm * v.dt / (2.f * gam);
        const float tx = f * Bx, ty = f * By, tz = f * Bz;
        const float t2 = tx * tx + ty * ty + tz * tz;
        const float sfac = 2.f / (1.f + t2);
        const float sx = sfac * tx, sy = sfac * ty, sz = sfac * tz;
        const float px = ux + (uy * tz - uz * ty);
        const float py = uy + (uz * tx - ux * tz);
        const float pz = uz + (ux * ty - uy * tx);
        ux += py * sz - pz * sy;
        uy += pz * sx - px * sz;
        uz += px * sy - py * sx;
        ux += hk * Ex; uy += hk * Ey; uz += hk * Ez;
    }
    {   // runaway disaster guard (counted)
        const float uu = ux * ux + uy * uy + uz * uz;
        if (uu > c.ucap * c.ucap) {
            const float r = c.ucap * rsqrtf(uu);
            ux *= r; uy *= r; uz *= r;
            if (runaway) atomicAdd(runaway, 1ull);
        }
    }
    const float gam1 = c.rel ? sqrtf(1.f + ux * ux + uy * uy + uz * uz) : 1.f;

    // time-centred delta-f weight update at u_mid ~ u(t^n) (2026-08-19
    // review: the old pre-Boris explicit-Euler form lagged the resonant
    // phase by O(dt)); wave-force-only, P2 ruling on the dropped dL term
    if (c.deltaf && !c.wdfreeze) {
        const float mx = 0.5f * (u0x + ux), my = 0.5f * (u0y + uy),
                    mz = 0.5f * (u0z + uz);
        const float gm = c.rel ? sqrtf(1.f + mx * mx + my * my + mz * mz) : 1.f;
        const float vx = mx / gm, vy = my / gm, vz = mz / gm;
        const float ax = c.qm * (Ex + vy * Bzw - vz * Byw);
        const float ay = c.qm * (Ey + vz * Bxw - vx * Bzw);
        const float az = c.qm * (Ez + vx * Byw - vy * Bxw);
        const float B0a = sqrtf(B0.x * B0.x + B0.z * B0.z);
        const float bhx = B0.x / B0a, bhz = B0.z / B0a;
        const float upar = mx * bhx + mz * bhz;
        const float upx = mx - upar * bhx, upz = mz - upar * bhz;
        float Beq = B0a;
        if (has_lines(bg)) {
            float xg, zg;
            gc_pos(x, z, my, c, bg, xg, zg);
            Beq = float(beq_of(bg, lshell_of<float>(bg, xg, zg)));
        }
        const float dE = -1.f / c.tpar;
        const float dMu = Beq * (1.f / c.tpar - 1.f / c.tperp);
        const float udota = mx * ax + my * ay + mz * az;
        const float updota = upx * ax + my * ay + upz * az;
        const float dlnf = dE * udota + dMu * updota / B0a;
        // structure-preserving update (user audit 2026-08-19): the exact
        // solution of dw/dt = -(1-w)*D over the step is
        //   1 - w_new = (1 - w_old) * exp(D*dt)
        // -> w_new = w_old - (1-w_old)*expm1(D*dt). Preserves wd < 1
        // EXACTLY (no clamp), agrees with the old explicit Euler to O(dt),
        // and is time-symmetric at the midpoint force used here.
        float wd = p.wd[m] - (1.f - p.wd[m]) * expm1f(dlnf * v.dt);
        if (c.taud > 0.f) wd *= expf(-v.dt / c.taud);
        p.wd[m] = wd;
    }

    float xn = x + ux / gam1 * v.dt;
    float zn = z + uz / gam1 * v.dt;

    {   // walls at the high-|λ| line ends only (P2 ruling)
        float xg, zg;
        gc_pos(xn, zn, uy, c, bg, xg, zg);
        if (xg < c.wx0 || zg < c.wz0 || zg > c.wz1) {
            const Vec2<float> bh = b0_bhat<float>(bg, xn, zn);
            const float up = ux * bh.x + uz * bh.z;
            if (up * (zg >= 0.f ? 1.f : -1.f) > 0.f) {
                ux -= 2.f * up * bh.x;
                uz -= 2.f * up * bh.z;
            }
        }
    }
    {   // periodic wrap of the STORED position; deposit uses pre-wrap
        const float Lx = v.nx * v.dx, Lz = v.nz * v.dz;
        p.x[m] = xn - Lx * floorf((xn - x0) / Lx);
        p.z[m] = zn - Lz * floorf((zn - z0) / Lz);
    }
    p.ux[m] = ux; p.uy[m] = uy; p.uz[m] = uz;

    PushOut o;
    o.g0x = gx; o.g0z = gz;
    o.g1x = (xn - x0) / v.dx; o.g1z = (zn - z0) / v.dz;
    o.qw = c.qm > 0.f ? p.w[m] * p.wd[m] : -p.w[m] * p.wd[m];
    o.vy1 = uy / gam1;
    return o;
}

// Esirkepov CIC over the worldline; ACC provides jx/jy/jz(i, k, val) with
// UNWRAPPED global indices (accumulators handle wrap / shared windows).
template <class ACC>
__device__ inline void esirkepov_2d(const PushOut& o, const FieldViews2D& v,
                                    ACC& acc) {
    const int ib = int(floorf(fminf(o.g0x, o.g1x)));
    const int kb = int(floorf(fminf(o.g0z, o.g1z)));
    float S0x[3], S1x[3], S0z[3], S1z[3], DSx[3], DSz[3];
    for (int a = 0; a < 3; ++a) {
        S0x[a] = fmaxf(0.f, 1.f - fabsf(o.g0x - (ib + a)));
        S1x[a] = fmaxf(0.f, 1.f - fabsf(o.g1x - (ib + a)));
        S0z[a] = fmaxf(0.f, 1.f - fabsf(o.g0z - (kb + a)));
        S1z[a] = fmaxf(0.f, 1.f - fabsf(o.g1z - (kb + a)));
        DSx[a] = S1x[a] - S0x[a];
        DSz[a] = S1z[a] - S0z[a];
    }
    const float fx = -o.qw / (v.dt * v.dz);
    const float fz = -o.qw / (v.dt * v.dx);
    const float fy = o.qw * o.vy1 / (v.dx * v.dz);
    for (int b = 0; b < 3; ++b) {
        float a2 = 0.f;
        for (int a = 0; a < 2; ++a) {
            a2 += DSx[a] * (S0z[b] + 0.5f * DSz[b]);
            acc.jx(ib + a, kb + b, fx * a2);
        }
    }
    for (int a = 0; a < 3; ++a) {
        float a2 = 0.f;
        for (int b = 0; b < 2; ++b) {
            a2 += DSz[b] * (S0x[a] + 0.5f * DSx[a]);
            acc.jz(ib + a, kb + b, fz * a2);
        }
    }
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            const float Wy = S0x[a] * S0z[b] +
                             0.5f * (DSx[a] * S0z[b] + S0x[a] * DSz[b]) +
                             (1.f / 3.f) * DSx[a] * DSz[b];
            acc.jy(ib + a, kb + b, fy * Wy);
        }
}

// deposits outside the active set are DROPPED AND COUNTED — markers live
// deep inside the band, so a nonzero count is a configuration error
// surfaced by the health check, never silent charge loss.
struct FlatAcc {
    const FieldViews2D& v;
    float *JX, *JY, *JZ;
    unsigned long long* dropped;
    __device__ void add(float* a, int i, int k, float val) {
        const int s = v.idx(i, k);
        if (s >= 0) atomicAdd(&a[s], val);
        else if (dropped) atomicAdd(dropped, 1ull);
    }
    __device__ void jx(int i, int k, float val) { add(JX, i, k, val); }
    __device__ void jy(int i, int k, float val) { add(JY, i, k, val); }
    __device__ void jz(int i, int k, float val) { add(JZ, i, k, val); }
};

static __global__ void k_push_deposit(MarkerViews p, KineticCfg c,
                                      FieldViews2D v, Background2D bg,
                                      float x0, float z0, uint64_t n,
                                      unsigned long long* runaway = nullptr) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const size_t ncell = size_t(v.nx) * v.nz;
    const int rep = blockIdx.x % v.nrep;
    FlatAcc acc{v, v.jxr + size_t(rep) * ncell, v.jyr + size_t(rep) * ncell,
                v.jzr + size_t(rep) * ncell, runaway ? runaway + 1 : nullptr};
    const PushOut o = push_move_one(p, m, c, v, bg, x0, z0, runaway);
    esirkepov_2d(o, v, acc);
}

// ---- P3d tiled deposit ----------------------------------------------------
// One block per 16×16-cell tile; markers come from the sort-time CSR
// (cell_start), so every marker is processed exactly once even after
// drifting. Deposits land in a shared (16+2H)² window (H = 6 covers the
// stencil + ≤5-cell drift between sorts); strays fall back to global
// atomics. 98% of tiles are empty in shell-compact runs and exit
// immediately, so the window flush cost lives only on shell tiles.
constexpr int TS_TILE = 16, TS_HALO = 6, TS_W = TS_TILE + 2 * TS_HALO;

struct TileAcc {
    const FieldViews2D& v;
    float *sjx, *sjy, *sjz;     // shared window [TS_W][TS_W]
    int wx0, wz0;               // window origin (unwrapped cell coords)
    unsigned long long* dropped;
    __device__ void add(float* sh, float* gl, int i, int k, float val) {
        const int li = i - wx0, lk = k - wz0;
        if (li >= 0 && li < TS_W && lk >= 0 && lk < TS_W) {
            atomicAdd(&sh[lk * TS_W + li], val);
            return;
        }
        const int s = v.idx(i, k);
        if (s >= 0) atomicAdd(&gl[s], val);
        else if (dropped) atomicAdd(dropped, 1ull);
    }
    __device__ void jx(int i, int k, float val) { add(sjx, v.jx, i, k, val); }
    __device__ void jy(int i, int k, float val) { add(sjy, v.jy, i, k, val); }
    __device__ void jz(int i, int k, float val) { add(sjz, v.jz, i, k, val); }
};

static __global__ void k_push_deposit_tiled(MarkerViews p, KineticCfg c,
                                            FieldViews2D v, Background2D bg,
                                            float x0, float z0,
                                            const uint32_t* cell_start,
                                            unsigned long long* runaway) {
    __shared__ float sjx[TS_W * TS_W], sjy[TS_W * TS_W], sjz[TS_W * TS_W];
    __shared__ uint32_t rs[TS_TILE], re[TS_TILE];
    __shared__ uint32_t total;
    const int ti = blockIdx.x * TS_TILE, tk = blockIdx.y * TS_TILE;
    const int wcols = min(TS_TILE, v.nx - ti);
    const int wrows = min(TS_TILE, v.nz - tk);
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    if (threadIdx.x < uint32_t(wrows)) {
        const size_t c0 = size_t(tk + threadIdx.x) * v.nx + ti;
        rs[threadIdx.x] = cell_start[c0];
        re[threadIdx.x] = cell_start[c0 + wcols];
        atomicAdd(&total, re[threadIdx.x] - rs[threadIdx.x]);
    }
    __syncthreads();
    if (total == 0) return;
    for (int i = threadIdx.x; i < TS_W * TS_W; i += blockDim.x)
        sjx[i] = sjy[i] = sjz[i] = 0.f;
    __syncthreads();

    TileAcc acc{v, sjx, sjy, sjz, ti - TS_HALO, tk - TS_HALO,
                runaway ? runaway + 1 : nullptr};
    for (int r = 0; r < wrows; ++r)
        for (uint32_t m = rs[r] + threadIdx.x; m < re[r]; m += blockDim.x) {
            const PushOut o = push_move_one(p, m, c, v, bg, x0, z0, runaway);
            esirkepov_2d(o, v, acc);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < TS_W * TS_W; i += blockDim.x) {
        const int gi = acc.wx0 + (i % TS_W), gk = acc.wz0 + (i / TS_W);
        const int s = v.idx(gi, gk);
        if (s < 0) {
            if ((sjx[i] != 0.f || sjy[i] != 0.f || sjz[i] != 0.f) && acc.dropped)
                atomicAdd(acc.dropped, 1ull);
            continue;
        }
        if (sjx[i] != 0.f) atomicAdd(&v.jx[s], sjx[i]);
        if (sjy[i] != 0.f) atomicAdd(&v.jy[s], sjy[i]);
        if (sjz[i] != 0.f) atomicAdd(&v.jz[s], sjz[i]);
    }
}

// wd statistics for the δf validity monitor: acc[0] += Σwd², and
// acc[1] tracks max|wd| via the monotone float→uint encoding (review
// 2026-08-19: G4 needs the max, rms alone hides a runaway tail)
static __global__ void k_wd_stats(MarkerViews p, double* acc,
                                  unsigned int* wmax, uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const float w = p.wd[m];
    const double w2 = double(w) * w;
    atomicAdd(&acc[0], w2);
    if (fabsf(w) > 3.f) { atomicAdd(&acc[1], 1.0); atomicAdd(&acc[3], w2); }
    if (fabsf(w) > 1.f) atomicAdd(&acc[2], 1.0);
    if (w >= 1.f + 1e-3f) atomicAdd(&acc[1], 1e12);  // bound violation flag
    if (wmax) atomicMax(wmax, __float_as_uint(fabsf(w)));
}

// node-centred charge deposit (q w wd, CIC S1 about NODES — the shape the
// Esirkepov identity pairs with): the live delta-f continuity gate input
// (user audit 2026-08-19: time-varying wd used as fixed segment charge
// leaves residual R = q w (wd_new-wd_old) S/dt; this measures it).
static __global__ void k_rho_node(MarkerViews p, FieldViews2D v, float x0,
                                  float z0, float qsign, float* rho,
                                  uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const float gx = (p.x[m] - x0) / v.dx, gz = (p.z[m] - z0) / v.dz;
    const int i0 = int(floorf(gx)), k0 = int(floorf(gz));
    const float fx = gx - i0, fz = gz - k0;
    const float qw = qsign * p.w[m] * p.wd[m] / (v.dx * v.dz);
    for (int c = 0; c < 4; ++c) {
        const int s = v.idx(i0 + (c & 1), k0 + (c >> 1));
        if (s >= 0)
            atomicAdd(&rho[s], qw * ((c & 1) ? fx : 1.f - fx) *
                                   ((c >> 1) ? fz : 1.f - fz));
    }
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
