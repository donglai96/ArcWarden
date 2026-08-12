# Learning Note — Step 4: Configuration (`config.hpp`)

> Goal: understand **why a PIC code splits its configuration into two layers** —
> a *compile-time* policy (`SimConfig`) and a *run-time* parameter struct
> (`RunParams`) — what each holds, and the C++ mechanics (template policies,
> `static_assert`, POD-by-value) behind them. After this you should be able to
> read `include/pic/config.hpp` and explain every line, and know *which* layer a
> new knob belongs in.

---

## 0. What Step 4 is, in one paragraph

Every later piece of the simulator (deposit, solver, push, diagnostics) needs to
know *how* it is configured: 2D or 3D? CIC or TSC shape? float or double? is
there a background magnetic field? which deposition algorithm? plus the physical
numbers — `dt`, density, thermal speed, the normalization constants. Step 4 sorts
all of that into **two** types in one header, `include/pic/config.hpp`, namespace
`arc`: `SimConfig<...>` for things fixed at **compile time**, and `RunParams` for
things known only at **run time**. No GPU code yet — this is the type-system spine
the rest of the project hangs on.

---

## 1. Background: two kinds of "configuration"

A configuration knob falls into one of two buckets, and which bucket decides how
you should store it:

- **Known at compile time, never changes during a run, and sits on the hot
  path.** Example: "particles are 2D, velocities are 3D, the shape function is
  CIC." The deposit/push kernels run billions of times; you want the compiler to
  *bake these in* and inline everything — no branch, no indirection.
- **Known only at run time, read as data.** Example: `dt = 0.01`, `n0 = 1.0`.
  These are just numbers you load from input and pass around; a kernel reads them
  from a struct argument.

Mixing the two is the classic mistake. If you store "is there a B field?" as a
runtime `bool` and write `if (has_b0) { ...Boris rotation... }` inside the push
kernel, *every thread* pays for that branch on *every step*, even in runs that
never use a B field. If instead it's a compile-time `bool`, the kernel is
specialized with `if constexpr` and the dead path **does not exist** in the
generated code.

---

## 2. The core idea: compile-time policy vs. run-time params

| | `SimConfig<...>` | `RunParams` |
|---|---|---|
| Bound | compile time (it's a *type*) | run time (it's *data*) |
| Holds | dim, vdim, shape, precision, has_b0, deposit policy | dt, n0, vth, vd, wpe, wce, B0, ppc, normalization fields |
| Cost at run time | **zero** (constants/types) | a struct passed by value |
| Changing it | edit a type argument, recompile | set a field, no recompile |
| Goes into a kernel? | as a template parameter `<Cfg>` | as a by-value argument |

> Mental model: `SimConfig` is the **blueprint stamped into the machine's
> castings** — change it and you re-machine the part. `RunParams` is the **dial
> settings** you turn before each run without touching the machine.

The design doc calls this out as "where the modern-C++ sophistication lives — in
the policy layer, not in an inheritance tree."

---

## 3. `ShapeOrder` and the deposit policy tags

```cpp
enum class ShapeOrder { CIC, TSC };

struct AtomicGlobalDeposit {};  // v0: one particle/thread + global atomicAdd
struct SharedTileDeposit   {};  // v1: one CTA/tile + shared-memory atomics
struct CellOwnedDeposit    {};  // v2: one warp/cell, high-PPC reduction
```

- `ShapeOrder` is a plain scoped enum — the particle's interpolation order. v1
  uses `CIC` (linear, cloud-in-cell); `TSC` is reserved for later.
- The three deposit structs are **empty tag types**. They hold no data; their
  only purpose is to *name a strategy* so the type system can select between
  them. This is the **policy pattern**: instead of a runtime `enum deposit_mode`
  + a `switch` inside the kernel, the choice is a *type*, resolved at compile
  time.

Why three? Charge deposition is the hardest kernel to make fast (many particles
writing into the same grid cells → atomic contention). There's a natural ladder
of implementations — naive global atomics (v0), shared-memory tiles (v1), warp-
per-cell reductions (v2). We want to swap among them **without touching the main
loop or the kernel signature**. The tags make that a one-line change (see §5).

---

## 4. `SimConfig` — the compile-time configuration

```cpp
template<int Dim, int VelDim, ShapeOrder Shape,
         typename Real, bool HasB0,
         class DepositPolicy = AtomicGlobalDeposit>
struct SimConfig {
    static constexpr int        dim    = Dim;      // 2
    static constexpr int        vdim   = VelDim;   // 3  -> "2D3V"
    static constexpr ShapeOrder shape  = Shape;    // CIC
    static constexpr bool       has_b0 = HasB0;
    using real    = Real;                          // float
    using deposit = DepositPolicy;                 // swap impl: change only this

    static_assert(Dim >= 1, "SimConfig: dim must be >= 1");
    static_assert(VelDim >= Dim, "SimConfig: vdim must be >= dim");
};

using Cfg = SimConfig<2, 3, ShapeOrder::CIC, float, true, AtomicGlobalDeposit>;
static_assert(Cfg::dim == 2,  "Cfg must be 2D");
static_assert(Cfg::vdim == 3, "Cfg must be 3V");
```

Key mechanics:

- **`static constexpr` members** are compile-time constants attached to the type.
  A kernel templated on `Cfg` can write `if constexpr (Cfg::has_b0)` and the
  compiler discards the other branch entirely.
- **`using real = Real;` / `using deposit = DepositPolicy;`** are *member type
  aliases*. They let other code say `typename Cfg::real` (the float type) or
  `typename Cfg::deposit` (the policy tag) without re-specifying the parameters.
- **"2D3V"** = 2 spatial dimensions, 3 velocity/momentum components. That's why
  `dim == 2` but `vdim == 3`: positions are (x, y) but momenta are
  (ux, uy, uz), because a magnetic field couples motion into the third component.
- **`DepositPolicy` has a default** (`AtomicGlobalDeposit`) so most call sites
  write the short form and only override when they want v1/v2.
- The two in-class `static_assert`s are **guard rails**: a velocity space
  narrower than the configuration space is never intended, so an accidental
  `SimConfig<3,2,...>` fails to compile with a readable message instead of
  producing nonsense downstream.
- `Cfg` is the **default project configuration** (2D3V, CIC, float, B0-capable,
  v0 deposit). Components reach for `arc::Cfg` unless they specifically need a
  different policy. The two `static_assert`s after it are exactly the Step 4
  verification: they prove at compile time that the hot path is locked to 2D3V.

### Why template policy and not `virtual`

- A virtual function means a **vtable + indirect call**. On a GPU that's a
  performance disaster, and virtual dispatch across the host/device boundary is
  awkward at best.
- A policy fixed at compile time lets the deposit/push kernels **inline fully** —
  no branch, no indirect jump (plan §10, "hot kernels compile-time fixed").
- Swapping v0 → v1 → v2 is a single type-argument change; the main loop and
  kernel signatures don't move.

---

## 5. How the policy actually gets used (the payoff)

You won't see the swap pay off until Step 10, but here's the shape of it:

```cpp
template<class Cfg, class DepositPolicy = typename Cfg::deposit>
__global__ void deposit_charge_kernel(/* views... */) {
    if constexpr (std::is_same_v<DepositPolicy, AtomicGlobalDeposit>) {
        // ... global atomicAdd path ...
    } else if constexpr (std::is_same_v<DepositPolicy, SharedTileDeposit>) {
        // ... shared-memory tile path ...
    }
}
```

The branch is `if constexpr`, so **only the selected path is compiled** for a
given `Cfg`. To switch implementations you change one type argument in `Cfg`:

```cpp
using Cfg = SimConfig<2,3,ShapeOrder::CIC,float,true, SharedTileDeposit>;
//                                                     ^^^^^^^^^^^^^^^^^ only this
```

The main loop, which just calls `Depositor<Cfg>::charge(...)`, never changes.

---

## 6. `RunParams` — the run-time physics/numerics

```cpp
struct RunParams {
    double dt = 0.0;
    double n0 = 1.0, vth = 0.0, vd = 0.0;   // density, thermal speed, drift
    double wpe = 1.0, wce = 0.0;            // plasma / cyclotron frequency
    float  B0[3] = {0,0,0};                 // external uniform B
    int    ppc = 0;                         // particles per cell
    long   nsteps = 0, dump_every = 0;

    NormMode norm = NormMode::OmegaPeUnity;
    double eps0 = 1.0, qm = -1.0, weight = 1.0;
    double ax = 0.0, ay = 0.0, k_filter = 0.0;
    unsigned long rng_seed = 0;
};
```

Two design points worth internalizing:

- **No geometry.** There is deliberately no `nx, ny, dx, dy, Lx, Ly` here. Those
  live in `Grid` (Step 5), the *single source of truth*. If geometry lived in
  both `Grid` and `RunParams`, the two copies would eventually drift and you'd
  spend a debugging afternoon discovering deposit used one `dx` and the solver
  another. One owner, always.
- **It's a POD passed by value into kernels.** `RunParams` has no pointers it
  owns, no destructor — it's *trivially copyable*, so CUDA can bit-copy the whole
  struct as a kernel argument (same requirement as `DeviceView` in Step 3). The
  defaults already encode the ωpe = 1 normalization, so a freshly built
  `RunParams{}` is self-consistent before you touch it.

---

## 7. The normalization contract (the part that saves you in Step 9)

The header's top comment nails down, *before any kernel exists*, the scaling that
makes ωpe = 1:

```text
weight = n0·dx·dy / ppc            macro-particle weight (phys / macro)
rho   += q·weight·CIC / (dx·dy)    charge density (CIC weights sum to 1)
phi_k  = rho_k / (eps0·k²)         periodic Poisson; k=0 & Nyquist -> 0
E_k    = -i·k·phi_k
after C2R: × 1/(nx·ny)             cuFFT does not normalize
```

Self-consistency check (uniform plasma): total charge
`= ppc·nx·ny · q · (n0·dx·dy/ppc) = q·n0·Area`, so `rho = q·n0`; with
`eps0 = me = |e| = 1` that gives `ωpe² = n0 = 1`. Writing this down *now* (and
echoing it into the deposit/solver kernels later) is the plan's explicit
guard against the classic "ωpe comes out wrong and you can't tell which kernel
dropped a factor" bug. The `RunParams` defaults (`eps0=1, qm=-1, weight=1`) are
the run-time half of that same contract.

**Momentum, not velocity:** `ux,uy,uz` store `u = γv` (v1 sets `γ ≡ 1`). Decided
up front so a later relativistic retrofit costs nothing; gather→push uses
`v = u/γ`.

---

## 8. How we verified Step 4

Verification goal (plan): *`static_assert(Cfg::dim==2, vdim==3)` holds; the header
compiles.* We:

1. Compiled a translation unit including `config.hpp` with the in-header
   `static_assert`s active (they fire at compile time, so a successful build *is*
   the proof), plus extra checks: `SharedTileDeposit` yields a *different*
   `Cfg::deposit` type than the default, and `std::is_trivially_copyable_v<RunParams>`
   (so it's a legal by-value kernel argument).
2. Built it two ways:
   - **nvcc 13.3** `-std=c++20 -arch=sm_120`, including a real `__global__` kernel
     that takes `RunParams` by value → **OK**.
   - **host `g++ -std=c++20`** (the header has no CUDA dependency, so it must
     compile as plain C++ too) → **OK**.

No runtime test is needed: everything in Step 4 is resolved at compile time.

---

## 9. Concepts you learned in Step 4 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Compile-time vs. run-time config | type-level constants vs. data | hot-path decisions baked in → zero-overhead kernels |
| Policy pattern (tag types) | empty structs naming a strategy | swap deposit algo by changing one type, loop unchanged |
| `static constexpr` / member `using` | constants & type aliases on a type | `Cfg::has_b0`, `Cfg::real`, `Cfg::deposit` |
| `if constexpr` (preview) | compile-time branch | dead path not even generated (no B0 cost) |
| `static_assert` guard rails | compile-time invariants | `dim==2,vdim==3`; `vdim>=dim` catches typos |
| POD by value (reprise) | trivially copyable struct | `RunParams` legal as a kernel argument |
| Single source of truth | geometry only in `Grid` | `RunParams` omits nx,ny,dx… to avoid drift |
| Normalization contract | the ωpe=1 scaling, written down once | prevents undebuggable factor-of-N errors later |

---

## 10. One-sentence summary

Step 4 splits configuration into a **compile-time policy** (`SimConfig<...>`:
dim/vdim/shape/precision/has_b0 + a deposit-policy tag, locked by `static_assert`
to 2D3V, so hot kernels inline with zero dispatch) and a **run-time POD**
(`RunParams`: dt and physics plus the ωpe=1 normalization fields, *no geometry* —
that's Grid's job), giving the rest of the simulator one authoritative,
zero-overhead place to ask "how am I configured?"
