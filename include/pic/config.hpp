// ArcWarden — compile-time + run-time configuration (Step 4).
//
// Two layers, deliberately separated (design §3):
//
//   SimConfig<...>  — COMPILE-TIME policy. Pins every hot-path decision
//                     (dimensions, shape function, precision, B0, deposit
//                     strategy) into the type system so kernels inline with
//                     zero virtual dispatch. Swapping a deposit implementation
//                     is a one-line type change; the main loop never moves.
//   RunParams       — RUN-TIME physics/numerics. A POD passed by value into
//                     kernels. Holds dt, plasma quantities and the normalization
//                     fields — but NOT geometry (nx,ny,dx,dy,Lx,Ly,idx live in
//                     Grid, the single source of truth, design §4) to avoid the
//                     two-places-drift bug.
//
// Why template policy and not virtual: device-side vtables + indirect calls are
// a performance disaster and awkward across the host/device boundary. A policy
// fixed at compile time lets the deposit/push kernels inline fully — no branch,
// no indirection (plan §10 "hot kernels compile-time fixed").
//
// ---------------------------------------------------------------------------
// NORMALIZATION CONTRACT (nailed down before any kernel; guarantees ωpe = 1).
// This is the single authoritative copy; deposit/solver kernels reference it.
//
//   weight = n0·dx·dy / ppc            macro-particle weight (phys / macro)
//   rho   += q·weight·CIC / (dx·dy)    charge density (CIC weights sum to 1)
//   phi_k  = rho_k / (eps0·k²)         periodic Poisson; k=0 & Nyquist -> 0
//   E_k    = -i·k·phi_k                = -i·k·rho_k / (eps0·k²)
//   after C2R: × 1/(nx·ny)             cuFFT does not normalize
//
// Self-consistency (uniform case): total charge
//   = ppc·nx·ny · q · (n0·dx·dy/ppc) = q·n0·Area  ->  rho = q·n0,
// and with eps0 = me = |e| = 1 this gives ωpe² = n0 = 1.
//
// Velocity/momentum semantics: particle components ux,uy,uz store MOMENTUM
// u = γv; v1 is non-relativistic with γ ≡ 1. gather->push uses v = u/γ.
// (Chosen up front so a later relativistic retrofit costs nothing.)
// ---------------------------------------------------------------------------

#ifndef ARC_PIC_CONFIG_HPP
#define ARC_PIC_CONFIG_HPP

namespace arc {

// ---- particle shape function (compile-time) -------------------------------

enum class ShapeOrder {
    CIC,  // cloud-in-cell (linear), v1 default
    TSC,  // triangular-shaped cloud (quadratic), reserved
};

// ---- charge-deposition strategy: compile-time policy tags -----------------
//
// Empty tag types selected by SimConfig::deposit. The single
// deposit_charge_kernel<Cfg> picks its body with `if constexpr` on this type,
// so going v0 -> v1 -> v2 changes ONE Cfg type argument and nothing else
// (design §8). They carry no state — only their identity matters.
struct AtomicGlobalDeposit {};  // v0: one particle per thread + global atomicAdd
struct SharedTileDeposit   {};  // v1: one CTA per tile + shared-memory atomics
struct CellOwnedDeposit    {};  // v2: one warp per cell, high-PPC reduction

// ---- SimConfig: the compile-time configuration ----------------------------
//
// Every member is a compile-time constant or a type, so it costs nothing at
// run time and lets kernels specialize on it. The deposit policy defaults to
// the simplest correct implementation (v0).
template<int Dim, int VelDim, ShapeOrder Shape,
         typename Real, bool HasB0,
         class DepositPolicy = AtomicGlobalDeposit>
struct SimConfig {
    static constexpr int        dim    = Dim;      // spatial dims (2)
    static constexpr int        vdim   = VelDim;   // velocity dims (3 -> 2D3V)
    static constexpr ShapeOrder shape  = Shape;    // CIC
    static constexpr bool       has_b0 = HasB0;    // magnetized electrostatic?
    using real    = Real;                          // particle/field precision (float)
    using deposit = DepositPolicy;                 // swap impl: change only this

    // Sanity: this project is 2D3V; a velocity space narrower than configuration
    // space is never intended.
    static_assert(Dim >= 1, "SimConfig: dim must be >= 1");
    static_assert(VelDim >= Dim, "SimConfig: vdim must be >= dim");
};

// Default v1 configuration: 2D3V, CIC, float, magnetized-capable, global-atomic
// deposit. Components reach for `arc::Cfg` unless they need a different policy.
using Cfg = SimConfig<2, 3, ShapeOrder::CIC, float, true, AtomicGlobalDeposit>;

// Compile-time guarantees the hot path is locked to 2D3V (plan Step 4 check).
static_assert(Cfg::dim == 2,  "Cfg must be 2D");
static_assert(Cfg::vdim == 3, "Cfg must be 3V");

// ---- normalization mode (run-time selector) -------------------------------

enum class NormMode {
    OmegaPeUnity,  // ωpe = 1, eps0 = 1, me = 1, |e| = 1 (the contract above)
};

// ---- RunParams: run-time physics / numerics (POD, passed by value) --------
//
// NO geometry here — that is Grid's job (design §3/§4). Defaults encode the
// ωpe = 1 normalization so a freshly-constructed RunParams already satisfies
// the closure relations above.
struct RunParams {
    double dt = 0.0;

    double n0  = 1.0;   // background number density
    double vth = 0.0;   // thermal velocity
    double vd  = 0.0;   // drift velocity

    double wpe = 1.0;   // plasma frequency
    double wce = 0.0;   // cyclotron frequency
    float  B0[3] = {0.0f, 0.0f, 0.0f};  // uniform external B direction/magnitude

    int  ppc        = 0;   // particles per cell
    long nsteps     = 0;
    long dump_every = 0;

    // ---- normalization closure (design §14.2) ----
    NormMode norm   = NormMode::OmegaPeUnity;
    double   eps0   = 1.0;   // vacuum permittivity (= 1)
    double   qm     = -1.0;  // q/m (electron = -1)
    double   weight = 1.0;   // macro weight = n0·dx·dy/ppc; sets rho scale
    double   ax = 0.0;       // particle-shape smoothing scale: s(k)=exp(-(kx·ax)²/2 …)
    double   ay = 0.0;
    double   k_filter = 0.0; // spectral filter cutoff (plan §4.2)

    unsigned long rng_seed = 0;  // reproducible particle loading

    // —— single-k initial perturbation (quiet-start seed for Langmuir / two-stream) ——
    // x-displacement (cell units) applied at load: x += perturb_amp·sin(2π·perturb_kx·x/nx).
    // Default 0 → uniform load (all earlier tests unaffected).
    double perturb_amp = 0.0;
    int    perturb_kx  = 1;

    // —— two-stream load: split each cell's particles into counter-streaming beams
    // ±vd along x (default false → single Maxwellian drifting at vd). ——
    bool two_stream = false;
};

} // namespace arc

#endif // ARC_PIC_CONFIG_HPP
