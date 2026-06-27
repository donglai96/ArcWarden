# Learning Note — Step 7: Source & Field Containers (`sources.hpp` / `fields.hpp`)

> Goal: understand the **three-way split of grid data** in a PIC code — what the
> deposit step *writes* (`Sources`), what the push step *reads* (`Fields`), and the
> k-space scratch (`SpectralWorkspace`, Step 6) — and why keeping them in separate
> containers (rather than one big `Fields` blob) is what keeps the code extensible.
> This is a short, mechanical step that reuses everything from Step 3.

---

## 0. What Step 7 is, in one paragraph

A PIC step moves data around the grid: particles deposit charge → `rho`; the solver
turns `rho` into fields `Ex, Ey`; particles gather those fields to get pushed. Step 7
defines the two real-space containers for that flow — `Sources` (holds `rho`) and
`Fields` (holds `Ex, Ey`) — in `include/pic/sources.hpp` and
`include/pic/fields.hpp`, namespace `arc`. Each pairs a host-side **owner**
(`DeviceArray`) with a kernel-facing **view pack** (`SourceViews` / `FieldViews`),
exactly the owner/view pattern from Step 3. No new GPU mechanics — the value here is
the *separation of concerns*.

---

## 1. Background: why not one `Fields` struct?

The tempting design is a single `Fields { rho, Ex, Ey, rho_k, phi_k, ... }`. It works
for a minimal electrostatic run and then collapses the moment you extend to Darwin
(electromagnetic) physics, where you suddenly have `jx, jy, jz`, `dcu`, `amu`,
`Ez, Bx, By, Bz`, and a pile of k-space buffers. Everything ends up coupled to
everything.

The established fix (UPIC's `qe` / `fxye` / `qt` split, design §6) is to group grid
data by **who touches it and when**:

| Container | Holds | Written by | Read by |
|---|---|---|---|
| `Sources` | `rho` (later `jx,jy,jz,…`) | deposit | solver |
| `Fields` | `Ex, Ey` (later `Ez,Bx,…`) | solver | push (gather) |
| `SpectralWorkspace` | `rho_k, phi_k, Ex_k, Ey_k` | solver (internal) | solver (internal) |

Now a Darwin extension is *additive*: you add members to the relevant container, and
the main loop plus the deposit/solve/push responsibility boundaries don't move. That
"add a field without rewiring the pipeline" property is the whole point.

---

## 2. The owner / view pattern (reprise from Step 3)

Each container has two parts:

```cpp
// kernel-facing: trivially copyable POD, no ownership — passed BY VALUE
struct SourceViews { DeviceView<float> rho; };
struct FieldViews  { DeviceView<float> Ex, Ey; };

// host-side owner: RAII DeviceArray, allocated once
struct Sources {
    DeviceArray<float> rho;
    explicit Sources(const Grid& g) : rho(g.real_size()) {}
    void allocate(const Grid& g) { rho = DeviceArray<float>(g.real_size()); }
    void zero(cudaStream_t s = nullptr) { rho.zero(s); }
    SourceViews views() { return SourceViews{ rho.view() }; }
};
```

- `Sources`/`Fields` live on the **host**, own GPU memory via `DeviceArray`, and are
  move-only by inheritance (a `DeviceArray` member can't be copied).
- `views()` packages the borrowed `DeviceView` handles into a small POD struct that
  the kernel takes **by value** — one argument carries both `Ex` and `Ey`. The
  `static_assert(std::is_trivially_copyable_v<FieldViews>)` in the test guarantees
  that bundle is legal as a kernel parameter.
- **Sizing comes from `Grid`** (`g.real_size() == nx*ny`) — the single source of
  truth again. The containers never store geometry; they ask the `Grid`.

---

## 3. Two construction paths, and why both exist

```cpp
Sources src(g);          // (a) construct already-sized
Sources s2; s2.allocate(g);   // (b) default-construct empty, size later
```

Path (a) is for locals/tests. Path (b) is the **build-empty-then-size** pattern from
Step 3 §5: a container that *contains* `Sources`/`Fields` as members (the
`Simulation` object, Step 13) default-constructs them empty, then calls `allocate(g)`
once the grid is known at `init()`. Both end at the same place; `allocate` just uses
`DeviceArray`'s move-assignment (`rho = DeviceArray<float>(N)`), which frees any old
buffer and steals the new one — no leak, no copy.

---

## 4. `zero()` — and a subtle difference between the two

```cpp
void Sources::zero(cudaStream_t s);   // called EVERY step
void Fields::zero(cudaStream_t s);    // optional
```

- `Sources::zero` is **mandatory each step**: deposit *accumulates* into `rho` with
  `atomicAdd`, so the buffer must start at 0 or last step's charge leaks in. It's a
  stream-ordered `cudaMemsetAsync` (Step 3), so it queues cheaply on the pipeline
  stream.
- `Fields::zero` is **optional**: the solver *overwrites* `Ex, Ey` every solve, so
  they don't need pre-clearing in steady operation. It's provided for init and tests
  (a known-zero starting state).

This asymmetry (accumulate vs. overwrite) is worth internalizing — it's the reason
the deposit step has a `zero` partner and the solver step does not.

---

## 5. How we verified Step 7

Verification goal (plan): *alloc / zero / views packaging correct; compiles.* In one
`.cu` we:

1. `static_assert` both view packs are trivially copyable (kernel-arg safe).
2. Allocated `Sources`/`Fields` from a small grid, `zero()`-ed, copied back, checked
   every element is `0`.
3. Launched kernels that write **through `views()`** — `rho[i]=3i`, `Ex[i]=i`,
   `Ey[i]=-2i` — copied back, and verified the values (proves the view handles point
   at the owners' real buffers).
4. Exercised the `allocate()` (default-then-size) path and checked the size.

All asserts pass under `nvcc -arch=sm_120`.

---

## 6. Concepts you learned in Step 7 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Three-way grid-data split | Sources / Fields / SpectralWorkspace | Darwin extension is additive, loop unchanged |
| Group by writer/reader | deposit→Sources, solver→Fields, push reads Fields | clean responsibility boundaries |
| Owner + view pack | `DeviceArray` owner, POD `*Views` for kernels | one by-value arg carries Ex+Ey |
| Size from `Grid` | `g.real_size()` | geometry stays single-source-of-truth |
| build-empty-then-size | default ctor + `allocate(g)` | container members sized at `init()` |
| accumulate vs overwrite | `Sources::zero` each step; `Fields` not | deposit adds (needs clear), solver writes |

---

## 7. One-sentence summary

Step 7 adds the two real-space containers — `Sources` (owns `rho`, zeroed each step
because deposit accumulates) and `Fields` (owns `Ex, Ey`, written by the solver) —
each pairing a `DeviceArray` owner with a trivially-copyable `*Views` pack for
kernels, kept separate from the k-space workspace so the deposit→solve→push pipeline
stays cleanly bounded and trivially extensible to Darwin.
