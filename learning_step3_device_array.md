# Learning Note — Step 3: GPU Memory Ownership (`device_array.hpp`)

> Goal: understand **why GPU memory needs two separate types**, what each does,
> and the C++/CUDA mechanics behind them. After this you should be able to read
> `include/pic/device_array.hpp` and explain every line — and know *which* type
> to use in a kernel vs. on the host.

---

## 0. What Step 3 is, in one paragraph

Every array of numbers our PIC code touches — particle positions, charge density
`rho`, fields `Ex/Ey`, the FFT spectral buffers — lives in **GPU memory**. We
need a safe way to **allocate it, zero it, hand it to kernels, and free it
exactly once**. Step 3 builds that as *two* cooperating types: `DeviceArray<T>`
(owns the memory, host-side, RAII) and `DeviceView<T>` (a borrowed handle you
pass into kernels). One header, `include/pic/device_array.hpp`, namespace `arc`.

---

## 1. Background: host memory vs. device memory

A CUDA program runs on two processors with **separate memories**:

- **Host** = the CPU + normal RAM. Your `main()`, `std::vector`, etc. live here.
- **Device** = the GPU + its own VRAM (31 GB on the RTX 5090). Kernels run here.

A pointer returned by `cudaMalloc` points into **GPU** memory. The CPU **cannot
dereference it** (`*p` on the host crashes); only GPU code can. Conversely a
kernel can't read a normal host pointer. Data crosses the gap explicitly with
`cudaMemcpy`. Keeping this straight is the whole reason Step 3 has the shape it
does.

The raw CUDA memory API:

```cpp
float* p;
cudaMalloc(&p, n * sizeof(float));   // allocate n floats in GPU memory
cudaMemset(p, 0, n * sizeof(float)); // zero them
... use p in kernels ...
cudaFree(p);                         // release (must happen exactly once)
```

Same leak/double-free hazards as Step 2's streams — so we wrap it in RAII.

---

## 2. The core idea: owning vs. non-owning (the most important decision)

A single allocation has **one owner** responsible for freeing it. But a kernel
just needs to *read/write* the memory — it shouldn't own anything (a kernel can't
call `cudaFree`, and copying an owner into a kernel would be a disaster). So we
split the responsibilities into two types:

| | `DeviceArray<T>` | `DeviceView<T>` |
|---|---|---|
| Role | **owns** the allocation | **borrows** it |
| Lives on | host (CPU) side | passed into kernels (device) |
| Lifetime | RAII: frees in destructor | none — it's just a pointer + size |
| Copyable? | **no** (move-only) | **yes** (trivially copyable POD) |
| Goes into a kernel? | **never** | **always** (by value) |

> Mental model: `DeviceArray` is the *landlord* (owns the building, handles
> demolition). `DeviceView` is a *key* you hand out (lets the holder use the
> rooms, but they can't sell or demolish the building). You can copy keys
> freely; there's only ever one landlord.

This is exactly the `std::unique_ptr` (owner) vs. raw pointer/`span` (view)
pattern from normal C++ — applied to GPU memory.

---

## 3. `DeviceView<T>` — the borrowed handle

```cpp
template<class T>
struct DeviceView {
    T*  ptr = nullptr;
    int n   = 0;

    __device__ T&       operator[](int i)       { return ptr[i]; }
    __device__ const T& operator[](int i) const { return ptr[i]; }

    __host__ __device__ int  size()  const { return n; }
    __host__ __device__ bool empty() const { return n == 0; }
};
```

Why it looks like this:

- **It's a `struct` with public data and no destructor** → it's a **POD**
  ("plain old data"): trivially copyable. That matters because CUDA passes kernel
  arguments **by value** (the struct is bit-copied to the GPU). A type with a
  destructor or owning semantics would be wrong/dangerous to copy that way. A
  `DeviceView` is safe to copy because copying it just duplicates a pointer +
  an int — no ownership implied.

- **`operator[]` is marked `__device__`** → it can *only* be called from GPU
  code. That's deliberate: `ptr` is GPU memory, so indexing it on the host would
  crash. The annotation makes the compiler enforce "use this inside kernels
  only."

- **`size()`/`empty()` are `__host__ __device__`** → those just read the `int n`
  (not the GPU memory), so they're safe to call from either side, e.g. for the
  bounds check `if (i < a.size())` inside a kernel.

Usage inside a kernel:

```cpp
__global__ void fill_iota(DeviceView<float> a) {   // taken BY VALUE
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) a[i] = 2.0f * i;             // a[i] -> a.ptr[i] on GPU
}
```

---

## 4. `DeviceArray<T>` — the owner

```cpp
template<class T>
class DeviceArray {
public:
    DeviceArray() = default;                        // empty: ptr=nullptr, n=0

    explicit DeviceArray(std::size_t n) : n_(n) {   // allocate
        if (n_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
    }

    ~DeviceArray() { reset(); }                     // free
    ...
};
```

It reuses **every pattern from Step 2's `CudaStream`**: RAII (allocate in ctor,
free in dtor), move-only (so ownership is unique), `CUDA_CHECK` on the calls. If
those words feel fuzzy, re-read `learning_step2_cuda_utils.md` §3 — it's the same
idea with a different resource.

### 4.1 Why move-only (again)

If you could **copy** a `DeviceArray`, two objects would hold the same `ptr_`,
and both destructors would `cudaFree` it → **double-free** (crash/corruption).
So copy is deleted, move is allowed:

```cpp
DeviceArray(const DeviceArray&)            = delete;   // no copy
DeviceArray& operator=(const DeviceArray&) = delete;

DeviceArray(DeviceArray&& other) noexcept             // move: steal
    : ptr_(std::exchange(other.ptr_, nullptr)),
      n_(std::exchange(other.n_, 0)) {}
```

`std::exchange(other.ptr_, nullptr)` returns `other`'s old pointer (we take it)
and sets `other.ptr_` to `nullptr`, so the moved-from object's destructor frees
nothing. **Exactly one owner at all times.**

### 4.2 The accessors

```cpp
T*       data();          // raw GPU pointer (for cudaMemcpy, etc.)
const T* data() const;
std::size_t size()  const;
std::size_t bytes() const;   // n * sizeof(T) — handy for cudaMemcpy/cudaMalloc
bool        empty() const;
```

`data()` is how you reach the raw pointer for host-side CUDA calls like
`cudaMemcpy(host, a.data(), a.bytes(), cudaMemcpyDeviceToHost)`.

### 4.3 `zero()`

```cpp
void zero(cudaStream_t stream = nullptr) {
    if (ptr_ && n_ > 0)
        CUDA_CHECK(cudaMemsetAsync(ptr_, 0, n_ * sizeof(T), stream));
}
```

- `cudaMemset` writes **bytes**, so this sets every byte to 0. For our types
  (`float`, `int`, `cufftComplex`) the all-zero bit pattern *is* the number 0
  (and `{0,0}` for complex), so byte-zero = numeric-zero. (This wouldn't be true
  for, say, a type whose "zero" isn't all-zero bits — not our case.)
- The **`Async`** variant + `stream` argument means the zeroing is queued on our
  pipeline stream rather than blocking the CPU — it runs in order with the
  kernels. This is why every PIC step can do `sources.zero(stream)` cheaply.

### 4.4 `view()` — making the borrowed handle

```cpp
DeviceView<T> view() {
    return DeviceView<T>{ptr_, static_cast<int>(n_)};
}
DeviceView<const T> view() const {       // const array -> read-only view
    return DeviceView<const T>{ptr_, static_cast<int>(n_)};
}
```

This is the bridge between the two types: the owner produces a cheap POD handle
to pass into kernels. Note the size is stored as `std::size_t` in the array but
narrowed to `int` in the view, because kernels index with `int` (our grids are
far below 2³¹ elements).

---

## 5. The "container resize" pattern (why the default ctor matters)

Later containers (`Particles`, `Sources`, `Fields`, the spectral workspace) hold
`DeviceArray` members, but the **size isn't known until runtime** (it depends on
the grid `nx*ny`, the particle count, etc.). The default constructor + move-
assign make this clean:

```cpp
struct Sources {
    DeviceArray<float> rho;            // starts empty (default ctor)

    void allocate(const Grid& g) {
        rho = DeviceArray<float>(g.real_size());   // construct sized, move-assign in
    }
};
```

`rho = DeviceArray<float>(...)` builds a fresh sized array and **move-assigns**
it into `rho`; the move-assign first frees whatever `rho` held, then steals the
new buffer. No copies, no leaks. This is *why* `DeviceArray` has both a default
ctor and a move-assignment operator.

---

## 6. How a real PIC kernel call looks (tying it together)

```cpp
// Host side: owners
DeviceArray<float> rho(grid.real_size());
rho.zero(stream);

// Launch: pass VIEWS (POD), never the arrays
deposit_charge<<<blocks, threads, 0, stream>>>(particles.views(), rho.view());

// Read a result back to the CPU
std::vector<float> h(rho.size());
cudaMemcpy(h.data(), rho.data(), rho.bytes(), cudaMemcpyDeviceToHost);
```

- Owners (`DeviceArray`) stay on the host and manage lifetime.
- `*.view()` produces the POD handles the kernel actually receives.
- `data()`/`bytes()` feed the host-side copy.

That separation — owners on the host, views in kernels — is the whole point of
Step 3, and it propagates up into every container (`ParticleViews`,
`SourceViews`, `FieldViews`) in later steps.

---

## 7. How we verified Step 3

Goal: *allocate / zero / copy-back correctness; `compute-sanitizer` clean.* We:

1. Allocated a `DeviceArray<float>(1000)`, called `zero()`, copied to host,
   checked every element is `0`.
2. Launched a kernel through `view()` writing `a[i] = 2*i`, copied back, checked.
3. Move-constructed into another array and confirmed the source is `empty()` /
   `data()==nullptr` (no double-free risk).
4. Did the default-ctor + `c = DeviceArray<float>(N)` move-assign pattern.
5. Ran `compute-sanitizer --tool memcheck --leak-check full`:
   **`0 bytes leaked in 0 allocations`, `0 errors`.**

---

## 8. Concepts you learned in Step 3 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Host vs. device memory | CPU RAM vs. GPU VRAM, separate address spaces | a GPU pointer can't be dereferenced on the CPU |
| Owning vs. non-owning | one owner frees; many borrowers just use | safe to copy a view, never a copy of the owner |
| POD / trivially copyable | plain struct, no dtor | required to pass a type into a kernel by value |
| `__device__` / `__host__ __device__` | where a function may be called | enforces "index GPU memory only on the GPU" |
| RAII + move-only (reprise) | ctor allocs, dtor frees, unique owner | no leaks, no double-free |
| `cudaMemsetAsync` | stream-ordered byte zero | cheap per-step `zero()` of sources/fields |
| `data()` / `bytes()` | raw pointer + size in bytes | feed `cudaMemcpy` and host checks |
| default ctor + move-assign | build-empty-then-replace | runtime-sized container members |

---

## 9. One-sentence summary

Step 3 splits GPU memory into an **owner** (`DeviceArray<T>`: RAII, move-only,
plain `cudaMalloc/cudaFree`, with `zero()`/`data()`/`view()`) and a **borrowed
POD handle** (`DeviceView<T>`: passed by value into kernels with a device-only
`operator[]`), so the host has clean unique ownership and kernels get zero-
overhead pointer access — verified correct and leak-free with `compute-sanitizer`.
