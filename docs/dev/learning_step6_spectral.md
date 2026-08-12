# Learning Note — Step 6: FFT + Spectral Engine (`fft.hpp` / `spectral.hpp`)

> Goal: understand **how a spectral PIC code turns charge density into fields via
> the FFT**, and the specific engineering around cuFFT: why the plan is built once,
> what the R2C "half-spectrum" really stores, and where the missing normalization
> factor goes. This is the first step with *running GPU compute* — the round-trip
> test actually transforms data on the RTX 5090.

---

## 0. What Step 6 is, in one paragraph

The electrostatic solver works in Fourier space: transform `rho` to `rho_k`, divide
by `k²` to get the potential, multiply by `i k` to get the field, transform back.
All of that rests on a fast, correct FFT. Step 6 builds the FFT layer — three types
in `include/pic/fft.hpp` and `include/pic/spectral.hpp`, namespace `arc`:
`CufftPlan2D` (owns the cuFFT plans), `SpectralWorkspace` (the k-space buffers),
`SpectralFormFactor` (the `1/(eps0 k²)` Green's function), and `SpectralEngine`
(ties them together, exposes `r2c`/`c2r`). The *solver itself* is Step 8; Step 6
just proves the transform round-trips correctly (`c2r(r2c(f)) ≈ f`) and conserves
energy (Parseval).

---

## 1. Background: why Fourier space, and what an FFT gives you

On a **periodic** grid, the natural basis is plane waves `e^{i k·x}`. The key magic:
**differentiation becomes multiplication.** A derivative `∂/∂x` in real space is just
`× (i k_x)` in Fourier space, and the Laplacian `∇²` is `× (−k²)`. So the Poisson
equation `∇²φ = −ρ/ε0`, which is an annoying coupled linear system in real space,
becomes a trivial *per-mode division* in Fourier space:

```
−k² φ_k = −ρ_k/ε0     →     φ_k = ρ_k / (ε0 k²)
```

The **FFT** (Fast Fourier Transform) is the O(N log N) algorithm that moves a field
between real space and this Fourier (k) space. cuFFT is NVIDIA's GPU implementation.
Step 6 wraps it; Step 8 uses the `φ_k = ρ_k/(ε0 k²)` relation.

---

## 2. cuFFT plans, and why you build them once

cuFFT separates *planning* from *executing*. A **plan** is a precomputed strategy
for a transform of a given size and type (it allocates scratch, picks algorithms).
Building a plan is comparatively expensive; executing it is fast. Crucially, **a plan
depends only on the transform size, not on the data** — so for a simulation that
FFTs every step, you build the plan **once** and reuse it thousands of times.
Rebuilding per step would be pure waste (design §7.1).

That lifetime — "create once, hold, destroy at the end" — is exactly the RAII,
move-only ownership pattern from Steps 2–3. Hence `CufftPlan2D`:

```cpp
class CufftPlan2D {
    cufftHandle r2c_ = 0, c2r_ = 0;
    bool created_ = false;
public:
    CufftPlan2D(int nx, int ny) {
        CUFFT_CHECK(cufftPlan2d(&r2c_, ny, nx, CUFFT_R2C));  // forward
        CUFFT_CHECK(cufftPlan2d(&c2r_, ny, nx, CUFFT_C2R));  // inverse
        created_ = true;
    }
    ~CufftPlan2D() { /* cufftDestroy both */ }
    // move-only; copy deleted
};
```

Three details worth noting:

- **Two plans, not one.** cuFFT plans are *typed*: a `CUFFT_R2C` plan does real→
  complex (forward) and a separate `CUFFT_C2R` plan does complex→real (inverse). One
  `CufftPlan2D` owns both so the engine has a complete round trip.
- **`cufftPlan2d(&p, ny, nx, ...)` — dimensions slowest-first.** cuFFT wants the
  slow (outer) dimension first. Our real array is `[ny][nx]` with x contiguous, so
  the slow dim is `ny`, the fast dim is `nx`. Passing them in the wrong order is a
  silent correctness bug — this is why the R2C layout contract from Step 5 matters.
- **`created_` instead of a `0` sentinel.** A freshly default-constructed handle is
  `0`, but `0` isn't a *guaranteed-invalid* cuFFT handle, so using "handle == 0?" to
  decide whether to destroy is slightly unsafe. An explicit `bool created_` is
  unambiguous. (The destructor is best-effort — it swallows errors rather than
  throwing, same rule as `CudaStream`.)

---

## 3. The R2C half-spectrum (the part everyone gets wrong once)

Because `rho` is **real**, its spectrum is conjugate-symmetric: `F(−k) = conj(F(k))`.
Storing both halves wastes memory, so cuFFT's **R2C** transform keeps only the
non-redundant half of the *fast* axis — `nx` real inputs become `nkx = nx/2 + 1`
complex outputs along x, with the full `ny` kept along y. So:

```
real    [ny][nx]      --R2C-->     complex [ny][nkx],  nkx = nx/2 + 1
```

This is the `KGrid` from Step 5 (`nkx = nx/2+1`, `nky = ny`), and it's why `kx` only
ever has non-negative values (the negative-k_x half was dropped) while `ky` wraps to
negatives past Nyquist (the y axis is full). Everything in this step is consistent
with that grid.

---

## 4. The missing normalization (and where we put it)

cuFFT is **unnormalized**: a forward-then-inverse round trip multiplies your data by
`N = nx*ny`. Mathematically `c2r(r2c(f)) = N·f`. Someone has to divide by `N`. The
contract (design §3, grid.hpp) puts that factor on the **inverse**:

```cpp
void c2r(cufftComplex* in, float* out, cudaStream_t s) const {
    plan_.exec_c2r(in, out, s);                       // raw cuFFT -> out = N·(ifft)
    float scale = 1.0f / (nx*ny);
    scale_inplace_kernel<<<...,s>>>(out, N, scale);   // out *= 1/N
    CUDA_CHECK(cudaPeekAtLastError());
}
```

So callers of `engine.c2r(...)` get the properly scaled inverse and never think about
it. Two subtleties:

- **cuFFT C2R is destructive.** The inverse transform *overwrites its input buffer*.
  That's why the round-trip test does the Parseval check (which needs `F`) *before*
  calling `c2r`, which consumes `F`.
- **The scale kernel is a function template.** A plain `__global__` defined in a
  header would violate the One Definition Rule if two `.cu` files included it.
  Making it `template<class T> __global__ void scale_inplace_kernel(...)` gives it
  *vague linkage* — the compiler dedups instantiations across translation units — so
  the header stays self-contained and safe to include anywhere.

---

## 5. The other two pieces: workspace and form factor

```cpp
struct SpectralWorkspace {              // long-lived k-space buffers
    DeviceArray<cufftComplex> rho_k, phi_k, Ex_k, Ey_k;
    explicit SpectralWorkspace(int complex_size) : rho_k(complex_size), ... {}
};

struct SpectralFormFactor {             // Green's fn + shape smoothing, one place
    double eps0 = 1.0;
    __host__ __device__ double smoothing(double k2) const { return 1.0; }     // s(k)=1 (v1)
    __host__ __device__ double green(double k2) const {                       // g(k)=1/(eps0 k²)
        return k2 > 0.0 ? 1.0/(eps0*k2) : 0.0;                                 // DC -> 0, no 1/0
    }
};
```

- `SpectralWorkspace` allocates the four complex buffers **once** (sized by
  `KGrid::complex_size() = nkx*ny`), reusing the Step-3 `DeviceArray` ownership.
- `SpectralFormFactor` is where `1/(eps0 k²)` lives — *not* inlined into the solver.
  The reason is forward-looking: adding particle-shape smoothing `s(k)` or a spectral
  filter later becomes a one-spot edit instead of hunting through kernels. v1 keeps
  `s(k)=1`. `green(0)` returns `0` so the k=0 (DC) mode never divides by zero — and
  zeroing DC is also how a periodic electrostatic system enforces charge neutrality
  (design §7.2). It's a POD with `__host__ __device__` accessors so a future solver
  kernel can use it by value.

`SpectralEngine` simply **owns** all of it (plan + workspace + `KGrid` + form factor)
and exposes `r2c`/`c2r`. Member declaration order is `grid_, kgrid_, plan_, ws_`
because `ws_(kgrid_.complex_size())` depends on `kgrid_` already being built — C++
initializes members in declaration order, so the dependency must come first.

---

## 6. The two tests, and what they prove

**Round trip** — `c2r(r2c(f)) ≈ f`. We build a smooth field `f` (a few sinusoids
plus a DC offset), forward-transform, inverse-transform, and compare to the original.
This validates the *whole* pipeline at once: plan dimensions, the R2C layout, and the
`1/N` normalization. If the dims were swapped or the scale were wrong, this fails.
Result: `max|c2r(r2c(f)) − f| = 4.8e-7` — right at single-precision noise for
`N = 2048`.

**Parseval** — energy is the same in both spaces. Parseval's theorem for the DFT:
`Σ_x f(x)² = (1/N) Σ_k |F(k)|²` over the **full** spectrum. But R2C stores only half
the x-axis, so we reconstruct the full-spectrum sum by weighting: x-modes that aren't
DC (`i=0`) or Nyquist (`i=nx/2`) each stand in for their dropped conjugate, so they
count **twice**:

```cpp
double w = (i == 0 || i == nx/2) ? 1.0 : 2.0;
e_k += w * (c.x*c.x + c.y*c.y);
// check:  e_k / N  ≈  Σ_x f²
```

This is an *independent* check from the round trip — it catches sign/packing errors
in how we interpret the half-spectrum (exactly the indexing we'll lean on in the
solver). Result: relative mismatch `4.5e-8`.

Both run under CTest (`fft_roundtrip`), and `compute-sanitizer --leak-check full`
reports **0 bytes leaked / 0 errors** — the RAII destructors free the cuFFT plans and
device arrays cleanly.

---

## 7. Concepts you learned in Step 6 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Fourier space | plane-wave basis on a periodic grid | derivative → ×(ik), Laplacian → ×(−k²) |
| FFT / cuFFT | O(N log N) real↔k transform on GPU | the engine under the whole ES solver |
| Plan vs. execute | precompute strategy once, run many times | build plan once, reuse every step (RAII) |
| `cufftPlan2d(ny,nx)` | dims passed slowest-first | must match real `[ny][nx]`, x-fast layout |
| R2C half-spectrum | real input → `nx/2+1` complex on fast axis | matches `KGrid`; only `kx ≥ 0` stored |
| Unnormalized FFT | round trip scales by N | `c2r` applies `1/(nx*ny)` so callers don't |
| C2R is destructive | inverse overwrites its input | do Parseval before `c2r` consumes `F` |
| Template `__global__` | vague linkage dedups across TUs | header-safe scale kernel, no ODR clash |
| Form factor in one place | `1/(eps0 k²)` not inlined in solver | smoothing/filter later = one-spot change |
| Parseval check | energy equal in both spaces | independent validation of half-spectrum packing |

---

## 8. One-sentence summary

Step 6 builds the FFT layer — `CufftPlan2D` (RAII, build-once R2C+C2R plans),
`SpectralWorkspace` (k-space buffers), `SpectralFormFactor` (`1/(eps0 k²)` in one
place), and `SpectralEngine` (which exposes `r2c` and a `c2r` that applies the
`1/(nx*ny)` cuFFT omits) — then proves it correct on the GPU with a round-trip test
(`max err 4.8e-7`) and a half-spectrum Parseval energy check (`4.5e-8`), leak-free.
