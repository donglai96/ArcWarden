# Learning Note — Step 10: Charge Deposition (`depositor.hpp`)

> Goal: understand **how particles become a grid charge density** — the "scatter"
> half of the PIC cycle — via CIC interpolation, why the four weights must sum to 1
> (charge conservation), how a GPU does the concurrent writes safely (atomicAdd),
> and where the compile-time deposit policy plugs in. This is the first *hot*
> kernel: one thread per particle, millions of times per run.

---

## 0. What Step 10 is, in one paragraph

The field solver (Step 8) needs `rho` on the grid, but charge lives on particles at
arbitrary positions. **Deposition** spreads each particle's charge onto the nearby
grid points. Step 10 builds `Depositor<Cfg>` and the `deposit_charge_kernel<Cfg>` in
`include/pic/depositor.hpp`, namespace `arc`, plus the shared `cic_stencil` helper in
`grid.hpp`. The test proves the operation conserves total charge exactly — for a
single particle, a particle straddling the periodic edge, and a full uniform load.

---

## 1. Background: CIC, the cloud-in-cell weighting

A particle at cell-unit position `(x,y)` sits inside a grid cell with corner indices
`i0=floor(x)`, `j0=floor(y)`, at fractional offset `fx=x-i0`, `fy=y-j0`. **Cloud-in-
cell (CIC)** treats the particle as a unit cloud and splits its charge among the four
surrounding grid points by **bilinear** weights:

```
w00 = (1-fx)(1-fy)    at (i0  , j0  )
w10 =    fx (1-fy)    at (i0+1, j0  )
w01 = (1-fx)   fy     at (i0  , j0+1)
w11 =    fx    fy     at (i0+1, j0+1)
```

The crucial property: `w00 + w10 + w01 + w11 = 1` for *any* `fx, fy`. So whatever a
particle's position, the total charge it deposits is exactly its own charge — nothing
is created or lost. That identity is the entire basis of charge conservation, and the
test checks it directly.

---

## 2. One stencil, shared by deposit and gather

```cpp
struct CicStencil { int cell[4]; float w[4]; };   // 4 wrapped cells + weights (in grid.hpp)
__host__ __device__ CicStencil cic_stencil(const Grid& g, float x, float y);
```

This helper lives in `grid.hpp`, **not** in `depositor.hpp`, on purpose. Deposition
(scatter: particle → grid) and the Pusher's gather (interp: grid → particle, Step 11)
must use the *identical* weights, or the self-force a particle feels won't cancel and
you get a spurious numerical force — one of the most infamous PIC bugs (plan §16).
Defining the stencil exactly once guarantees they can never drift apart. It also
applies the periodic `wrap` (Step 5) to `i0+1`/`j0+1`, so a particle in the last cell
correctly spills onto cell 0.

---

## 3. The deposit kernel — one thread per particle

```cpp
template<class Cfg, class DepositPolicy = typename Cfg::deposit>
__global__ void deposit_charge_kernel(ParticleViews p, SourceViews src, Grid g, RunParams rp) {
    int t = ...; if (t >= p.n) return;
    CicStencil st = cic_stencil(g, p.x[t], p.y[t]);
    float coef = q * weight / (dx*dy);                 // charge density per macro-particle
    if constexpr (is_same<DepositPolicy, AtomicGlobalDeposit>) {
        for (k=0..3) atomicAdd(&src.rho[st.cell[k]], coef * st.w[k]);
    }
}
```

- **Hot kernel, compile-time policy.** Each thread handles one particle. The
  *strategy* (`AtomicGlobalDeposit`) is a type carried by `Cfg`, selected with
  `if constexpr` — no virtual call, fully inlined (design §8). v1 is the simplest
  correct version; the `else` branch is the seam where shared-tile (v1) and
  cell-owned (v2) implementations land later, with the kernel signature and main loop
  unchanged.
- **Writes `Sources` (rho), not `Fields`.** Deposit and gather are opposite ends of
  the pipeline; keeping them in separate containers (Step 7) makes that explicit.

### 3.1 Why atomicAdd

Many particles in the same cell run on different threads and all add to the same
`rho[cell]` *at the same time*. A plain `rho[c] += v` would race (read-modify-write
collisions lose updates). `atomicAdd(&rho[c], v)` makes each add indivisible, so the
sum is correct regardless of contention. It's the simplest correct GPU scatter — at
the cost of serialization when many threads hit one cell, which is exactly what the
later shared-tile/cell-owned policies optimize. Correctness first, speed later, with
the policy seam making the swap a one-type change.

---

## 4. The normalization (closing the ωpe=1 contract)

From the `config.hpp` contract: a macro-particle represents `weight = n0·dx·dy/ppc`
physical particles, each of charge `q`, so it deposits charge `q·weight`. As a
*density* (per unit area) it adds `q·weight/(dx·dy)` times the CIC weight:

```cpp
float coef = q * weight / (dx*dy);   // q = rp.qm (= q/m; equals q since me=1)
rho[cell] += coef * w_cic;
```

The self-consistency, for a uniform plasma (`ppc` particles per cell): each cell
accumulates `ppc · q·weight = ppc·q·(n0 dx dy/ppc) = q·n0·dx·dy` of charge, i.e.
`rho = q·n0`. With `q=-1, n0=1` that's `rho = -1` everywhere — and `ωpe² = n0 = 1`.
The test confirms `rho` comes out flat at exactly `-1.00000`.

> Note `Depositor::charge` does *not* zero `rho` — the main loop calls
> `Sources::zero` first each step (deposit *accumulates*). Separating the two keeps
> the zeroing visible in the loop (Step 7's accumulate-vs-overwrite point).

---

## 5. How we verified Step 10

Charge is conserved exactly because the CIC weights sum to 1, so the test asserts
`sum(rho)·dx·dy == N·q·weight` in three regimes:

1. **Single particle** at `(2.3, 3.7)`: deposits to exactly **4** cells, total charge
   `= q·weight`. ✓
2. **Boundary particle** at `(7.6, 7.6)` on an 8×8 grid: its `i0+1`, `j0+1` wrap to
   cell 0, so charge lands on cells `(7,7),(0,7),(7,0),(0,0)` — total still
   `q·weight`, nothing lost at the periodic edge. ✓
3. **Uniform quiet load** (`ppc`/cell): total `= N·q·weight = -1`, and `rho` is flat
   at `-1.00000` (min == max — the quiet start gives zero CIC ripple), matching
   `q·n0`. ✓

CTest 5/5; `compute-sanitizer --leak-check full` → 0 leaked / 0 errors (atomics and
all).

---

## 6. Concepts you learned in Step 10 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| CIC interpolation | bilinear split to 4 grid points | smooth deposit; weights sum to 1 |
| Weights sum to 1 | `Σ w = 1` for any position | exact charge conservation |
| Shared stencil | one `cic_stencil` for deposit + gather | kills the self-force/weight-mismatch bug |
| Hot kernel + policy | one thread/particle, `if constexpr` on `Cfg` | zero-overhead strategy swap |
| atomicAdd | indivisible concurrent add | correct GPU scatter under contention |
| Deposit normalization | `q·weight/(dx·dy)·CIC` | closes ωpe=1; uniform ρ = q·n0 |
| accumulate, don't zero | caller zeroes rho each step | deposit adds; zeroing stays in the loop |
| periodic wrap in deposit | `i0+1` wraps to 0 | edge particles conserve charge |

---

## 7. One-sentence summary

Step 10's `Depositor<Cfg>` scatters each particle's charge `q·weight/(dx·dy)` onto its
4 CIC cells with `atomicAdd` (one thread per particle, compile-time deposit policy),
using the same `cic_stencil` the gather will use — and because the CIC weights sum to
1, total charge is conserved exactly (verified for single, periodic-boundary, and
uniform loads, with `rho` flat at `q·n0 = -1`).
