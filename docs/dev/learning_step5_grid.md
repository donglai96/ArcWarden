# Learning Note — Step 5: Grid Geometry (`grid.hpp`)

> Goal: understand **how a spectral PIC code describes its mesh** — the real-space
> `Grid` and the Fourier-space `KGrid` — and *why* the index maps and wavenumber
> formulas look the way they do. The two trickiest ideas here are the **R2C
> half-spectrum layout** and the **negative-wavenumber wrap past Nyquist**; get
> those right once, here, and every later kernel inherits them for free.

---

## 0. What Step 5 is, in one paragraph

The simulation lives on a 2D periodic mesh. Two things need a precise definition:
**where a grid point sits in memory** (real space) and **what wavenumber a Fourier
mode carries** (k-space). Step 5 puts both in one header, `include/pic/grid.hpp`,
namespace `arc`, as two tiny POD structs: `Grid` (nx, ny, dx, dy, Lx, Ly + the
flat index map) and `KGrid` (nkx, nky + `kx`, `ky`, `k2`). No GPU memory, no
kernels — just geometry. Both are trivially copyable so they ride into kernels by
value, exactly like `DeviceView` and `RunParams` before them.

---

## 1. Background: the mesh and periodicity

We discretize a rectangular domain `[0,Lx) × [0,Ly)` into `nx × ny` cells. A cell
has width `dx = Lx/nx`, `dy = Ly/ny`. The domain is **periodic**: going off the
right edge brings you back to the left (`x = Lx` *is* `x = 0`). That's why `nx`
cells of width `Lx/nx` tile the domain exactly — there's no extra "fence-post"
point at `x = Lx`, because that point is identified with `x = 0`.

A real field (charge density `rho`, potential, `Ex`, `Ey`) is one number per grid
point, stored as a flat array. We need a rule mapping a 2D coordinate `(i, j)` to a
1D array offset. That rule is `Grid::idx`.

---

## 2. `Grid` — real-space geometry (single source of truth)

```cpp
struct Grid {
    int    nx, ny;
    double dx, dy, Lx, Ly;

    Grid(int nx_, int ny_, double Lx_, double Ly_)
        : nx(nx_), ny(ny_), dx(Lx_/nx_), dy(Ly_/ny_), Lx(Lx_), Ly(Ly_) {}

    __host__ __device__ int idx(int i, int j) const { return j*nx + i; }
    __host__ __device__ int real_size() const { return nx*ny; }
    ...
};
```

Key points:

- **`idx(i,j) = j*nx + i`.** This says the field is stored **row-major with x
  fastest**: element `(0,0)`, `(1,0)`, …, `(nx-1,0)`, then `(0,1)`, … In memory,
  marching `i` (x) by 1 moves one slot; marching `j` (y) by 1 jumps `nx` slots.
  Layout is `[ny][nx]`. This choice is not arbitrary — it has to agree with how
  cuFFT reads the array (see §4).
- **Single source of truth.** `Grid` is the *only* place geometry lives. The
  design (§3/§4) deliberately keeps `nx, dx, …` out of `RunParams`. If geometry
  lived in two structs, one kernel could read `dx` from one and another from the
  other, they'd drift, and you'd chase a scaling bug for hours. One owner.
- **`__host__ __device__`.** `idx` is pure arithmetic on the struct's own ints, so
  it's safe to call from CPU *and* GPU code. Marking it both lets host setup code
  and device kernels share the identical formula — no risk of two definitions
  disagreeing. (The qualifiers come from `<cuda_runtime.h>`, which defines them to
  nothing in a host-only translation unit, so the header also compiles under plain
  `g++`.)

### 2.1 Periodic wrapping

```cpp
__host__ __device__ static int wrap(int i, int n) {
    if (i >= n) i -= n;
    if (i < 0)  i += n;
    return i;
}
__host__ __device__ int idx_periodic(int i, int j) const {
    return wrap(j, ny)*nx + wrap(i, nx);
}
```

CIC deposition writes to a cell *and its neighbor* `i+1`; a particle in the last
cell has neighbor `i+1 = nx`, which must wrap to `0`. Likewise a particle that
drifts to `x < 0` wraps to the right side. `wrap` handles indices that are **at
most one period out of range** (true for a single CIC neighbor or one leapfrog
step), so two `if`s are enough and cheaper than a modulo. Centralizing it here
means deposit, gather, and migrate all wrap *identically* — the design flags
mismatched wrapping as a classic bug source.

---

## 3. Background: the real-to-complex (R2C) FFT and its half-spectrum

The spectral solver works by Fourier transforming `rho`. Because `rho` is **real**
(not complex), its FFT has a symmetry: the negative-frequency half is the complex
conjugate of the positive half. Storing both halves would be redundant, so cuFFT's
**R2C** transform stores only the non-redundant half of the *fastest* axis: an
input of `nx` reals becomes `nx/2 + 1` complex numbers along that axis.

So if the real field is `[ny][nx]`, the spectrum is `[ny][nkx]` with
**`nkx = nx/2 + 1`**. The slow axis (`y`) keeps all `ny` modes; only the fast axis
(`x`) is halved. That asymmetry is the single most important thing to internalize
in Step 5, and it's exactly why `kx` and `ky` are computed by *different* rules.

---

## 4. The R2C layout contract (memorize this block)

```text
real field : [ny][nx],  idx(i,j) = j*nx + i,   x (i) is the FAST axis
cuFFT plan : cufftPlan2d(&p, ny, nx)           slow dim ny first, fast dim nx
R2C output : [ny][nkx], nkx = nx/2 + 1,        kidx(i,j) = j*nkx + i
```

Three things must agree or the transform is silently wrong:

1. Our memory layout puts **x fastest** (`idx = j*nx+i`).
2. `cufftPlan2d(&p, ny, nx)` is told **ny is the slow dim, nx the fast dim** — cuFFT
   takes dimensions slowest-first.
3. The R2C output therefore halves the **fast (x)** axis → `nkx = nx/2+1`, full `ny`
   on the slow axis.

This contract lives verbatim in the header's top comment so there's one
authoritative copy; the FFT plan (Step 6) and every kernel that indexes a field
must match it.

---

## 5. `KGrid` — wavenumbers, and the Nyquist wrap

```cpp
struct KGrid {
    int    nkx, nky;     // nkx = nx/2+1, nky = ny
    double dkx, dky;     // 2π/Lx, 2π/Ly

    explicit KGrid(const Grid& g)
        : nkx(g.nx/2+1), nky(g.ny), dkx(kTwoPi/g.Lx), dky(kTwoPi/g.Ly) {}

    __host__ __device__ double kx(int i) const { return i*dkx; }

    __host__ __device__ double ky(int j) const {
        const int m = (j <= nky/2) ? j : j - nky;   // wrap past Nyquist
        return m*dky;
    }

    __host__ __device__ double k2(int i,int j) const {
        return kx(i)*kx(i) + ky(j)*ky(j);
    }
};
```

The wavenumber spacing is `dk = 2π/L` (one full wave across the domain is the
lowest mode). Then mode index → physical wavenumber differs by axis:

- **`kx(i) = i·dkx`.** The x-axis is the *reduced* (R2C) axis: indices
  `i ∈ [0, nkx)` are exactly the **non-negative** modes `0, dkx, 2dkx, …,
  (nx/2)·dkx`. No negatives appear, because the conjugate (negative) half was
  dropped by the R2C transform. Simple.

- **`ky(j)`** needs the wrap. The y-axis is the *full* FFT axis, where an array of
  `ny` complex modes packs frequencies in FFT order:
  `0, 1, 2, …, ny/2, -(ny/2−1), …, −1` (times `dky`). The first half are positive;
  once `j` passes `ny/2`, the stored mode actually represents a **negative**
  wavenumber, so we map `m = j - ny`. This is the `(j <= ny/2 ? j : j-ny)` line —
  identical to `numpy.fftfreq`'s ordering.

  > Why does the high half mean negative? A discrete sample can't tell a wave of
  > frequency `f` from `f − ny` (aliasing). FFT convention assigns the upper half
  > to the negative-frequency alias, because that's the one with the smallest
  > magnitude — the physically meaningful representative.

- **The Nyquist point** (`j = ny/2`, only exists for even `ny`) is genuinely
  ambiguous (`+ny/2` and `−ny/2` are the same sampled wave). We follow the design's
  stated convention "j > ny/2 takes negative", so `j = ny/2` stays **positive**.
  It doesn't matter physically: the solver **zeros the Nyquist mode** anyway
  (a first-derivative operator like `E_k = -i k φ_k` is ill-posed on a pure-real
  Nyquist mode), so the sign there is never used.

- **`k2(i,j) = kx² + ky²`** is what the Poisson solver divides by:
  `φ_k = ρ_k/(ε0·k²)`. Crucially `k2(0,0) = 0` *exactly* (both `kx(0)` and `ky(0)`
  are `0`), and the solver special-cases that DC mode to avoid dividing by zero
  (it sets `φ_{k=0}=0`, which also enforces charge neutrality). The unit test
  asserts `k2(0,0)==0.0` precisely because the solver depends on it.

---

## 6. Why both structs are trivially copyable (and must stay so)

`Grid` and `KGrid` hold only ints and doubles — no pointers they own, no
destructor. That makes them **trivially copyable**, the same property `DeviceView`
and `RunParams` needed: CUDA bit-copies a kernel argument to the GPU, so a deposit
kernel can take `Grid g` *by value* and call `g.idx(...)` on the device. Adding a
member that owns a resource (or a virtual function) would silently break this. The
Step 5 test pins it down with `static_assert(std::is_trivially_copyable_v<Grid>)`.

> Note the constructors don't break trivial copyability: a user-defined *non-copy*
> constructor is fine; only a non-trivial copy/move/destructor would disqualify it.

---

## 7. How we verified Step 5

Verification goal (plan): *`kx/ky/k2` correct incl. Nyquist/negative indexing; `idx`
periodic-consistent.* Because every method is `__host__ __device__`, we tested them
on the **host** (no GPU needed) for a small `8×4` grid, `Lx=16, Ly=8`:

1. `dx==2, dy==2`, `real_size()==32`, `idx(7,3)==31` (row-major, x fastest).
2. Periodic wrap: `idx_periodic(8,0)==idx(0,0)`, `idx_periodic(-1,0)==idx(7,0)`, the
   same in `j`, and in-range coordinates unchanged.
3. `nkx==5` (=8/2+1), `nky==4`, `dkx=2π/16`, `dky=2π/8`.
4. `kx(i)==i·dkx` for all `i`, all non-negative; `kx(4)` (Nyquist) positive.
5. `ky`: `ky(0)=0`, `ky(1)=+dky`, `ky(2)` (Nyquist) `=+2dky`, `ky(3)=−dky`
   (the wrap past Nyquist).
6. `k2(0,0)==0.0` exactly; `k2(3,3)` equals `kx(3)²+ky(3)²`.
7. `static_assert` both structs trivially copyable.

All asserts pass under `g++ -std=c++20`, and the same file compiles under
`nvcc -arch=sm_120 -x cu` (device path), confirming the `__host__ __device__`
methods are valid GPU code too.

---

## 8. Concepts you learned in Step 5 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Row-major index, x-fast | `idx = j*nx + i` | one flat layout all kernels + cuFFT share |
| Single source of truth | geometry only in `Grid` | no `dx` drift between kernels |
| Periodic wrap | bring out-of-range `i,j` back into range | CIC `i+1` and particle drift across edges |
| R2C half-spectrum | real FFT stores `nx/2+1` on the fast axis | `nkx=nx/2+1`, full `ny`; asymmetric k-grids |
| `cufftPlan2d(ny,nx)` order | dims passed slow-first | must match our x-fast memory layout |
| `dk = 2π/L` | wavenumber spacing | lowest mode = one wave across the box |
| Nyquist negative wrap | upper-half indices are negative freqs | correct `ky`; solver zeros Nyquist anyway |
| `k2(0,0)==0` | DC mode | solver special-cases it (no 1/0, neutrality) |
| trivially copyable | POD, no owning members | `Grid`/`KGrid` pass into kernels by value |

---

## 9. One-sentence summary

Step 5 defines the mesh as two trivially-copyable POD structs — `Grid` (real space:
`nx,ny,dx,dy,Lx,Ly`, the row-major `idx=j*nx+i` map, and periodic wrapping, the
single source of geometric truth) and `KGrid` (the R2C half-spectrum: `nkx=nx/2+1`,
non-negative `kx`, Nyquist-wrapped `ky`, and `k2` with an exact-zero DC) — pinning
down the layout-and-wavenumber contract that the FFT and every later kernel must
obey.
