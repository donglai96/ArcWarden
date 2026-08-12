# ArcWarden — Steps 0–3 Summary & Key Learnings (English)

> Project: a 2D3V electrostatic spectral GPU PIC code, designed to grow into a
> Darwin / full-EM solver. Pure GPU, single card, C++20 + CUDA, namespace `arc`,
> headers in `include/pic/`.
> This note summarizes what we built in Steps 0–3 and the transferable knowledge:
> GPU programming, C++ data-structure design, the build system, and CUDA.

---

## Part 1 — What each step did

### Step 0: Scaffolding
- Created the directory tree: `include/pic/`, `src/`, `tests/`.
- Created **empty** header/source/test files and an empty `CMakeLists.txt`.
- Purpose: lock the file layout before writing any code.
- **Verification:** tree exists, files empty, nothing extra.

### Step 1: Build system (CMake)
- Wrote `CMakeLists.txt`: C++20 + CUDA, picks the right `nvcc` (cuda-13.3),
  targets the RTX 5090 (`sm_120`), links `CUDA::cudart` / `CUDA::cufft`.
- A runnable `main.cpp` that prints the GPU and exits 0; a CTest smoke test.
- **Key fix (found in Step 2):** `CMAKE_CUDA_ARCHITECTURES` was being cached as
  `75`. Cause: the `if(NOT DEFINED ...)` guard sat *after* `project(... CUDA)`,
  but `project()` already initializes that variable to a compiler default. Moving
  the `set(... native)` *before* `project()` fixed it → `sm_120`.
- **Verification:** `cmake -B build && cmake --build build` succeeds; the binary
  reports `RTX 5090 / sm_120 / 31.4 GB`; `ctest` passes.

### Step 2: CUDA infrastructure — `cuda_utils.hpp`
- **`CUDA_CHECK` / `CUFFT_CHECK`** macros: wrap any CUDA/cuFFT call, throw an
  `arc::CudaError` with `file:line`, the failing expression, and the reason.
- **`CudaStream`**: RAII, move-only wrapper for `cudaStream_t` (one stream drives
  the whole pipeline; enables later CUDA Graph capture).
- **`CudaEvent`**: RAII, move-only wrapper for `cudaEvent_t` (timing + sync).
- **Verification:** create/move/destroy + timing + error path; `compute-sanitizer
  --leak-check full` → 0 bytes leaked.

### Step 3: GPU memory ownership — `device_array.hpp`
- **`DeviceArray<T>`**: OWNS GPU memory. Host-side, RAII, move-only, plain
  `cudaMalloc`/`cudaFree`. Has `zero(stream)`, `data()`, `size()`, `bytes()`,
  `view()`. Never passed into a kernel.
- **`DeviceView<T>`**: NON-OWNING POD handle (`ptr` + `n`), trivially copyable,
  passed *by value* into kernels, with a device-only `operator[]`.
- The default-ctor + move-assign enable the "allocate later" container pattern:
  `rho = DeviceArray<float>(grid.real_size())`.
- **Verification:** zero/fill-via-kernel/copy-back correct; move leaves source
  empty; `compute-sanitizer` → 0 leaks / 0 errors.

---

## Part 2 — Key learnings: GPU programming

1. **Two processors, two memories.** Host (CPU+RAM) and Device (GPU+VRAM) have
   *separate address spaces*. A `cudaMalloc` pointer is GPU memory — the CPU
   cannot dereference it; only kernels can. Data crosses via `cudaMemcpy`.

2. **Kernels and launch configuration.** A `__global__` function is a kernel,
   launched as `kernel<<<blocks, threads, sharedBytes, stream>>>(args)`. Each
   thread computes its global index:
   `int i = blockIdx.x * blockDim.x + threadIdx.x;` and must bounds-check
   `if (i < n)`.

3. **Kernel arguments are passed by value (bit-copied to the GPU).** So you pass
   small POD handles (`DeviceView`), never owning objects. This is *why* the
   owning/non-owning split exists.

4. **Function-space qualifiers.** `__host__` = callable on CPU, `__device__` =
   callable on GPU, `__host__ __device__` = both. We mark `DeviceView::operator[]`
   as `__device__` so indexing GPU memory on the host is a compile error.

5. **Streams = ordered queues of GPU work.** Operations on one stream run in
   order; the CPU can keep going (async). Using a single pipeline stream keeps
   ordering simple and enables CUDA Graph capture later.

6. **Events = markers in a stream**, used for timing
   (`cudaEventElapsedTime`) and synchronization. You must synchronize the later
   event before reading elapsed time.

7. **Async + sync discipline.** `cudaMemsetAsync`, kernel launches, and
   `cudaMemcpyAsync` are non-blocking; you `cudaStreamSynchronize` /
   `cudaDeviceSynchronize` when you need the result on the CPU.

---

## Part 3 — Key learnings: C++ data structures & design

1. **RAII (Resource Acquisition Is Initialization).** Tie a resource's lifetime
   to an object: acquire in the constructor, release in the destructor. Cleanup
   then happens automatically — even when an exception unwinds the stack. We use
   it for streams, events, and GPU memory.

2. **Move-only types.** For a unique resource (one stream, one allocation),
   **delete copy** and **allow move**. Copying would create two owners → double
   free. Move *transfers* ownership; `std::exchange(x, nullptr)` returns the old
   value and nulls the source so its destructor does nothing. Same model as
   `std::unique_ptr`.

3. **Owning vs. non-owning split.** Separate the type that *manages lifetime*
   (`DeviceArray`, host-side, RAII) from the type that is *used* in hot code
   (`DeviceView`, POD, copyable). Landlord vs. key.

4. **POD / trivially copyable.** A plain struct with public data and no
   destructor can be safely bit-copied — a hard requirement for kernel arguments.

5. **`noexcept` cleanup in destructors.** Destructors swallow errors (best-effort
   `cudaFree`/`cudaStreamDestroy`) — throwing during stack unwinding can call
   `std::terminate`.

6. **Throw, don't abort.** `CUDA_CHECK` throws so that RAII destructors run on the
   way out and release GPU resources cleanly, instead of leaking on a hard abort.

7. **Compile-time vs. run-time polymorphism (design direction).** Hot per-particle
   paths use compile-time policy (templates + `if constexpr`) → zero overhead;
   "which solver" is a cheap host-side choice (`std::variant`/`std::visit`),
   never a device virtual call.

---

## Part 4 — Key learnings: the build system (CMake / make)

1. **CMake is a build-system *generator*; two phases.**
   `cmake -B build` (configure: read `CMakeLists.txt`, detect compilers/GPU,
   generate `build/`) then `cmake --build build` (build: run the compiler).

2. **Compile then link.** Each `.cpp`/`.cu` compiles independently to a `.o`
   object file; the linker stitches objects + libraries into one executable.
   Incremental builds only recompile changed files.

3. **`CMakeLists.txt` runs top-to-bottom; ordering matters.** Anything that
   affects *how a language is enabled* (compiler choice, GPU architecture) must
   be set **before** `project(... LANGUAGES ...)`. Target configuration comes
   after.

4. **The cache persists decisions.** `build/CMakeCache.txt` remembers everything
   from configure. A stale cached value can override your edited default — fix
   with `-DVAR=...` or `rm -rf build` (this is what caused/hid the `sm_75` bug).

5. **Target-centric, scoped commands.** Use `target_include_directories`,
   `target_link_libraries`, etc. `PRIVATE` = only this target; `PUBLIC` = this
   target + dependents; `INTERFACE` = dependents only.

6. **Imported targets** like `CUDA::cudart` / `CUDA::cufft` carry both their
   include paths and link flags, so `find_package(CUDAToolkit)` + linking the
   target is all you need.

7. **Clean/rebuild commands.**
   - changed source → `cmake --build build`
   - changed `CMakeLists.txt` → `rm -rf build && cmake -B build && cmake --build build`
   - light clean → `cmake --build build --target clean`

---

## Part 5 — Key learnings: CUDA specifics

1. **Error handling is manual.** CUDA returns status codes (`cudaError_t`,
   `cufftResult`), not exceptions. You must check every call — hence
   `CUDA_CHECK`/`CUFFT_CHECK`.

2. **cuFFT has no built-in error-string function** (the CUDA runtime does:
   `cudaGetErrorName`/`cudaGetErrorString`). We wrote our own lookup. Note cuFFT
   13.x removed some legacy enum values (`CUFFT_PARSE_ERROR`, etc.).

3. **GPU architecture flags.** `sm_120` is the RTX 5090's compute capability
   (12.0). `native` auto-detects; `120-real` = emit real SASS for sm_120.

4. **Toolkit version matters.** The system `nvcc` (12.0) is too old for sm_120;
   we explicitly select cuda-13.3.

5. **`compute-sanitizer`** is the CUDA correctness checker:
   `--tool memcheck --leak-check full` catches leaks and out-of-bounds. We use it
   as the verification gate for every memory/resource step.

6. **Byte-wise zeroing.** `cudaMemset` writes bytes; for `float`/`int`/
   `cufftComplex` the all-zero bit pattern equals numeric zero, so it's a valid
   `zero()`.

7. **R2C FFT layout (committed convention).** real `[ny][nx]`,
   `idx = j*nx + i`; R2C output `[ny][nx/2+1]`; `cufftPlan2d(&p, ny, nx)`. Lock
   this everywhere — Darwin/EM batched FFTs depend on it.

---

## Part 6 — Other important practices

1. **One subtitle = one step; verify before advancing.** Each step is only "done"
   after its verification passes (compile + run + sanitizer). No skipping.

2. **Verification gate = `compute-sanitizer` clean.** For infrastructure steps,
   "it compiles" is not enough; "0 bytes leaked / 0 errors" is the bar.

3. **Design for the future, allocate for the present.** Containers *reserve* the
   names for later physics (`J`, `B`, `E_k`, `B_k`) but don't allocate them in v1.
   This keeps the API stable from ES → Darwin → full EM without rewrites.

4. **Keep geometry in one place.** `Grid` is the single source of truth for
   `nx, ny, dx, dy, Lx, Ly, idx`; `RunParams` holds physics only. Avoids "drift"
   between two copies of the same number.

5. **Normalization contract up front.** `eps0=1, me=1, |e|=1, ωpe=1`; positions
   in cell units; pin it before coding so units don't shift mid-project.

6. **Learning notes per step.** `learning_step{1,2,3}_*.md` explain the *why*, so
   the knowledge transfers beyond this codebase.

---

## Status
Steps 0, 1, 2, 3 complete and verified. Next: **Step 4 — `config.hpp`**
(`SimConfig` compile-time policy, `RunParams`, deposit policy tags).
