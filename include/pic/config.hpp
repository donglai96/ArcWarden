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
struct SharedTileDeposit   {};  // v1: WHOLE-grid privatized rho (small grids only)
struct CellOwnedDeposit    {};  // v2: one warp per cell, high-PPC reduction
// v3: fixed-size SPATIAL tile (TX*TY cells + 1-cell halo) privatized in shared,
// fed by a coarse per-tile particle binning (counting sort). Shared footprint is
// (TX+1)(TY+1)*4B — CONSTANT in the grid size — so unlike SharedTileDeposit it
// scales to large 2D/3D grids without falling back to global atomics. This is the
// OSIRIS-style deposit (per-block tiles + binning + shared accumulate + flush).
struct TiledBinnedDeposit  {};  // v3: spatial-tile + binning, grid-size-independent

// ---- field model: compile-time selector (electrostatic vs spectral Darwin) --
//
// Picks which spectral field solve + step ordering the simulation runs. ES is
// the v1 default (ρ→E_L only); Darwin adds the magnetic field B(J) and the
// retained transverse inductive field E_T (UPIC mpdbeps2). Compile-time so the
// main loop specializes with `if constexpr` and the ES path pays nothing.
enum class FieldModel { Electrostatic, Darwin };

// ---- SimConfig: the compile-time configuration ----------------------------
//
// Every member is a compile-time constant or a type, so it costs nothing at
// run time and lets kernels specialize on it. The deposit policy defaults to
// the simplest correct implementation (v0).
template<int Dim, int VelDim, ShapeOrder Shape,
         typename Real, bool HasB0,
         class DepositPolicy = AtomicGlobalDeposit,
         FieldModel Model = FieldModel::Electrostatic>
struct SimConfig {
    static constexpr int        dim    = Dim;      // spatial dims (2)
    static constexpr int        vdim   = VelDim;   // velocity dims (3 -> 2D3V)
    static constexpr ShapeOrder shape  = Shape;    // CIC
    static constexpr bool       has_b0 = HasB0;    // magnetized electrostatic?
    static constexpr FieldModel field_model = Model;  // ES or Darwin spectral solve
    using real    = Real;                          // particle/field precision (float)
    using deposit = DepositPolicy;                 // swap impl: change only this

    // Sanity: this project is 2D3V; a velocity space narrower than configuration
    // space is never intended.
    static_assert(Dim >= 1, "SimConfig: dim must be >= 1");
    static_assert(VelDim >= Dim, "SimConfig: vdim must be >= dim");
};

// Default v1 configuration: 2D3V, CIC, float, magnetized-capable, shared-tile
// deposit (per-block privatized rho; ~7-34x faster than global atomics at high
// ppc, falls back to global if the grid exceeds shared-memory opt-in). Components
// reach for `arc::Cfg` unless they need a different policy.
using Cfg = SimConfig<2, 3, ShapeOrder::CIC, float, true, SharedTileDeposit>;

// Spectral Darwin EM configuration: 2D3V, CIC, float, magnetized, tiled deposit
// (J/dcu/amu reuse it), Darwin field model. Opt-in — components stay on `Cfg`
// (ES) unless they explicitly want the EM path.
using CfgDarwin = SimConfig<2, 3, ShapeOrder::CIC, float, true,
                            TiledBinnedDeposit, FieldModel::Darwin>;

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

    // —— spectral Darwin (FieldModel::Darwin) ——
    // c = speed of light in code units; the Darwin coupling μ₀ = 1/(eps0·c²)
    // scales B and E_T. ndc = transverse-field inner iterations (E_T depends on
    // dcu which depends on E_total = E_L + E_T → solved by ndc fixed-point sweeps;
    // UPIC default 1–2). Unused by the ES path.
    double   c      = 1.0;
    int      ndc    = 1;

    // —— Yee branch (FieldModel::YeeMaxwell) ——
    // jfilter = number of 3×3 binomial smoothing passes applied to J each step
    // (OSIRIS "smooth" equivalent; 0 = off). Suppresses CIC deposit noise on
    // long runs where dx ≳ λ_D.
    int      jfilter = 0;
    // tile_sort = re-sort particles into 16×16 spatial tiles every N steps and
    // run the tiled shared-memory Esirkepov deposit (0 = flat global-atomic
    // kernel). Particles drifting > 2 cells between sorts fall back to global
    // atomics, so any cadence is correct; N·v_max·dt/dx ≲ 2 keeps it fast.
    int      tile_sort = 0;

    // —— M2 absorbing boundary in x (Yee branch; y stays periodic) ——
    // bnd_x = 1: Umeda-style multiplicative damping masks exp(-numax·d²·dt)
    // over bnd_nd cells at both x ends, applied to all wave fields each step;
    // particles reflect specularly at the layer inner edges x = bnd_nd and
    // x = nx - bnd_nd (cell units), so the layers stay plasma-free. Field
    // indexing REMAINS periodic — the damped layers suppress the wrap-around
    // leakage (chirp1d-proven masking scheme, ported to 2D).
    // bnd_x = 2 ("hybrid") additionally damps each layer particle's TRANSVERSE
    // momentum (uy, uz) by the same mask each step. Field-only masks (mode 1)
    // do NOT absorb whistlers: most of the wave energy rides in the coherent
    // electron transverse current, which re-radiates the damped field (measured
    // R ≈ 1 in tools/boundary_reflection.cu). This is the 2D analogue of
    // chirp1d damping the cold-fluid vcy/vcz along with the wave fields.
    int      bnd_x     = 0;     // 0 = periodic, 1 = field damping, 2 = hybrid
    int      bnd_nd    = 64;    // damping cells per side
    double   bnd_numax = 1.0;   // peak damping rate

    // —— M3 delta-f (Yee branch): nonlinear two-weight scheme, g = f0 ——
    // Markers sample the bi-Maxwellian reference f0 ∝ exp(-ux²/2Tpar
    // - (uy²+uz²)/2Tperp) with B0 ∥ x̂ (Tpar/Tperp = uth², gyrotropy in y,z
    // required). Weight equation (chirp1d / Tao PPCF 2017 eq. 19 form):
    //   dwd/dt = -(1 - wd)·(q/m)·F·∂ln f0/∂u,   F = δE + v×δB (WAVE fields
    // only — the uniform B0 rotation conserves f0), discretized explicitly
    // with post-push u. Deposit weights every current by wd (DeltaF policy).
    int      deltaf    = 0;
    double   df_tpar   = 1.0;   // uth_par²  (f0 parallel temperature, m = 1)
    double   df_tperp  = 1.0;   // uth_perp²

    // —— M2/M10 antenna: localized rotating transverse current column ——
    // J_y += amp·g(x)·cos(w0 t)·ramp, J_z -= amp·g(x)·sin(w0 t)·ramp with
    // g(x) = exp(-(x-x0)²/2σ²) (cell units), ramp linear over ant_trmp, off
    // after ant_toff. Radiates R-helicity whistler packets in ±x (chirp1d
    // triggering antenna, ported to the 2D Yee branch). amp = 0 disables.
    double   ant_amp   = 0.0;
    double   ant_x0    = 0.0;   // column center (cells)
    double   ant_sigma = 2.0;   // column width (cells)
    double   ant_w0    = 0.0;   // drive frequency
    double   ant_trmp  = 0.0;   // ramp-up time
    double   ant_toff  = 0.0;   // turn-off time

    // —— external pump field (whistler driver, An et al. 2019, Eq. S1/S2) ——
    // E_pump_α(x,t) = Re{ Ẽ_α e^{i(k0·x − w0·t)} } · ramp(t), added to the total E
    // used in the dcu deposit + push (the whistler B then forms self-consistently
    // via the Darwin solve). Complex amps: Ex real, Ey purely imaginary (90° shift),
    // Ez real. Code units already (caller converts the paper's Ẽ normalization).
    bool   pump      = false;
    double pump_ex   = 0.0;   // Re(Ẽx)  [code E units]
    double pump_ey   = 0.0;   // Im(Ẽy)  (Ey contributes −pump_ey·sinθ)
    double pump_ez   = 0.0;   // Re(Ẽz)
    double pump_k0   = 0.0;   // wavenumber 2πM/Lx
    double pump_w0   = 0.0;   // frequency ω0
    double pump_trmp = 0.0;   // linear ramp up/down duration
    double pump_toff = 0.0;   // pump turn-off time
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

    // —— bump-on-tail load: a warm bulk Maxwellian (width vth, no drift) plus a
    // small beam in the tail. A fraction beam_frac of each cell's particles is
    // loaded as the beam (drift beam_vd, width beam_vth); the rest is the bulk.
    // Takes priority over two_stream when true. The positive slope ∂f/∂v on the
    // beam's inner edge is the free energy that drives the Langmuir instability. ——
    bool   bump_on_tail = false;
    double beam_frac    = 0.0;   // fraction of ppc placed in the beam (e.g. 0.08)
    double beam_vd      = 0.0;   // beam drift velocity (where the bump sits)
    double beam_vth     = 0.0;   // beam thermal width

    // —— noisy load: random within-cell positions + random Maxwellian velocities
    // (per-particle hashed RNG, seeded by rng_seed) instead of the quiet-start
    // van der Corput lattice. This restores physical shot noise so instabilities
    // self-excite from fluctuations — no perturb_amp seed needed. Default false
    // keeps the quiet start (flat rho) used by all earlier tests. ——
    bool noisy_load = false;
};

} // namespace arc

#endif // ARC_PIC_CONFIG_HPP
