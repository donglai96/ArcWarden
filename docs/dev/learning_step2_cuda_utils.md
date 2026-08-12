# Learning Note — Step 2: CUDA Infrastructure (`cuda_utils.hpp`)

> Goal: understand **what Step 2 built and why**, and learn the C++/CUDA concepts
> behind it — error checking, RAII, move-only ownership. By the end you should be
> able to read `include/pic/cuda_utils.hpp` and explain every line.

---

## 0. What Step 2 is, in one paragraph

Every later GPU file (FFT, deposition, pusher, solver) needs three boring-but-
critical things: **(1)** a way to check whether a CUDA call failed and report
*where*, **(2)** a CUDA **stream** (a queue of GPU work) that is safely created
and destroyed, and **(3)** a CUDA **event** for timing. Step 2 writes these once,
correctly, in one header so nobody re-invents them (badly) later. It's pure
infrastructure — no physics yet.

The whole step is a single header: `include/pic/cuda_utils.hpp`, namespace `arc`.

---

## 1. Background: how CUDA reports errors (and why we need a wrapper)

CUDA's C API does **not** throw exceptions. Almost every call returns a status
code instead:

```cpp
cudaError_t err = cudaStreamCreate(&s);   // returns cudaSuccess or an error code
```

If you ignore that return value (very easy to do), a failure stays silent and
your program corrupts later, far from the real cause. The standard fix is a
**check macro** that inspects the return code and reacts. We want it to:

- report the **file and line** where the failure happened,
- report **which call** failed and the human-readable reason,
- **stop execution** cleanly (we throw a C++ exception).

That's `CUDA_CHECK`. The same idea for the cuFFT library gives `CUFFT_CHECK`.

---

## 2. Utility #1 — the error-checking macros

### 2.1 The exception type

```cpp
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};
```

We define our own exception type so callers can `catch (const arc::CudaError&)`
specifically. It inherits from `std::runtime_error`, so `.what()` returns the
message string. Why throw instead of `abort()`? Because **RAII cleanup runs
during stack unwinding** — streams/events/memory get released on the way out
instead of leaking (more in §3).

### 2.2 Turning a cuFFT code into text

```cpp
inline const char* cufft_error_string(cufftResult r) {
    switch (r) {
        case CUFFT_SUCCESS:      return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        ...
    }
}
```

cuFFT (unlike the CUDA runtime) has **no built-in function** to convert its
error enum to a string, so we write our own lookup table. (The CUDA runtime
*does* provide `cudaGetErrorName` / `cudaGetErrorString`, so we just call those.)

> Note we had to delete `CUFFT_INCOMPLETE_PARAMETER_LIST`, `CUFFT_PARSE_ERROR`,
> `CUFFT_LICENSE_ERROR` — cuFFT 13.x removed those enum values, so referencing
> them no longer compiles.

### 2.3 The actual check functions

```cpp
namespace detail {
inline void cuda_check(cudaError_t err, const char* expr,
                       const char* file, int line) {
    if (err != cudaSuccess) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "%s:%d: CUDA error in `%s`: %s (%s)",
            file, line, expr,
            cudaGetErrorName(err), cudaGetErrorString(err));
        throw CudaError(buf);
    }
}
} // namespace detail
```

- It takes the **return code**, plus the *text* of the expression, the file, and
  the line.
- On failure it formats a message like
  `cuda_utils.cu:31: CUDA error in \`cudaSetDevice(99999)\`: cudaErrorInvalidDevice (invalid device ordinal)`
  and throws.
- On success it does nothing — zero overhead in the happy path.
- It lives in `namespace detail` to signal "internal helper, don't call directly".

### 2.4 The macros that capture file/line/expression

```cpp
#define CUDA_CHECK(expr)  ::arc::detail::cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUFFT_CHECK(expr) ::arc::detail::cufft_check((expr), #expr, __FILE__, __LINE__)
```

Why a **macro** and not a function? Only a macro can capture *where it was
written*:

- `__FILE__` and `__LINE__` are compiler built-ins that expand to the current
  file name and line number **at the call site**.
- `#expr` is the **stringizing operator**: it turns the code you passed into a
  string literal. So `CUDA_CHECK(cudaStreamCreate(&s))` makes `#expr` become
  `"cudaStreamCreate(&s)"` for the error message.

A normal function couldn't see the caller's file/line or the original source
text — they'd all report the line *inside* the helper.

**Usage everywhere from now on:**

```cpp
CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
CUFFT_CHECK(cufftExecR2C(plan, in, out));
```

Wrap every CUDA/cuFFT call. If it fails, you get an exact location instead of a
silent corruption.

---

## 3. The big idea behind #2 and #3: RAII

Before the stream/event classes, you need **RAII** (Resource Acquisition Is
Initialization). It's the central C++ resource-management pattern:

> Tie a resource's lifetime to an object's lifetime. **Acquire** it in the
> constructor; **release** it in the destructor. When the object goes out of
> scope, cleanup happens automatically — even if an exception is thrown.

The raw CUDA way is error-prone:

```cpp
cudaStream_t s;
cudaStreamCreate(&s);
... do work ...            // if this throws or returns early -> LEAK
cudaStreamDestroy(s);      // easy to forget; never runs on early exit
```

The RAII way can't leak:

```cpp
{
    arc::CudaStream s;     // constructor calls cudaStreamCreate
    ... do work ...        // throw/return is fine
}                          // destructor calls cudaStreamDestroy automatically
```

This is *why* `CUDA_CHECK` throws instead of aborting: a thrown exception unwinds
the stack, and every RAII object's destructor runs on the way out, releasing GPU
resources cleanly.

### 3.1 "Move-only" — the second half of correct ownership

A CUDA stream handle is a single underlying resource. If you **copied** the
wrapper, two objects would hold the *same* handle, and both destructors would
call `cudaStreamDestroy` on it — a **double free**. To prevent that, these
classes are **move-only**:

- **Copying is deleted** — you cannot accidentally duplicate ownership.
- **Moving is allowed** — you can *transfer* ownership from one object to
  another (the source is left empty, so only one object ever owns the handle).

```cpp
CudaStream(const CudaStream&)            = delete;   // no copy
CudaStream& operator=(const CudaStream&) = delete;   // no copy-assign

CudaStream(CudaStream&& other) noexcept              // move: steal the handle
    : stream_(std::exchange(other.stream_, nullptr)) {}
```

`std::exchange(other.stream_, nullptr)` does two things at once: returns
`other`'s old handle (which we keep) and sets `other.stream_` to `nullptr`
(so the moved-from object's destructor does nothing). Result: **exactly one
owner at all times.**

This is the same ownership model as `std::unique_ptr`.

---

## 4. Utility #2 — `CudaStream`

### 4.1 What a CUDA stream *is*

A **stream** is an ordered queue of GPU operations. Work pushed to the same
stream runs in order; the CPU can keep going while the GPU churns through it.
Our design uses **one stream for the whole pipeline** (deposit → FFT → solve →
push), which also makes later CUDA Graph capture possible (one capture target).

### 4.2 The class, piece by piece

```cpp
class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }      // acquire

    explicit CudaStream(unsigned int flags) {                    // non-blocking variant
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() { reset(); }                                   // release

    CudaStream(const CudaStream&) = delete;                      // no copy
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept                      // move-construct
        : stream_(std::exchange(other.stream_, nullptr)) {}

    CudaStream& operator=(CudaStream&& other) noexcept {         // move-assign
        if (this != &other) {
            reset();                                             // free what we hold
            stream_ = std::exchange(other.stream_, nullptr);     // take theirs
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }                 // raw handle for CUDA calls
    operator cudaStream_t() const { return stream_; }            // implicit convert too

    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    void reset() noexcept {
        if (stream_) {
            cudaStreamDestroy(stream_);   // best-effort: no throw in destructor
            stream_ = nullptr;
        }
    }
    cudaStream_t stream_ = nullptr;
};
```

Things worth noticing:

- **Two constructors:** default makes a normal stream; the `flags` one allows
  `cudaStreamNonBlocking`. `explicit` stops accidental conversions from an int.
- **`get()` and the implicit conversion** both hand the raw `cudaStream_t` to
  CUDA APIs, so you can write `kernel<<<g,b,0,stream>>>` or
  `cudaMemcpyAsync(..., stream)`.
- **`reset()` swallows errors** (`noexcept`, no `CUDA_CHECK`). Throwing from a
  destructor is dangerous in C++ (can call `std::terminate`), so cleanup is
  best-effort.
- **Move-assign frees first, then steals**, and guards against self-assignment
  (`this != &other`).

---

## 5. Utility #3 — `CudaEvent`

### 5.1 What a CUDA event *is*

An **event** is a marker you drop into a stream. The GPU "records" it when it
reaches that point. Two main uses:

1. **Timing**: record an event before and after some GPU work, then ask how many
   milliseconds elapsed between them.
2. **Synchronization**: wait until the GPU has passed a marker.

### 5.2 The interesting methods

The structure mirrors `CudaStream` (RAII, move-only). The new parts:

```cpp
void record(cudaStream_t stream = nullptr) const {
    CUDA_CHECK(cudaEventRecord(event_, stream));   // drop the marker into the stream
}

void synchronize() const { CUDA_CHECK(cudaEventSynchronize(event_)); } // CPU waits for it

// milliseconds from `start` to this event
float elapsed_ms_since(const CudaEvent& start) const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
}
```

### 5.3 The timing pattern

```cpp
arc::CudaEvent start, stop;
start.record(stream);
my_kernel<<<grid, block, 0, stream>>>(...);
stop.record(stream);
stop.synchronize();                       // wait until the GPU reaches `stop`
float ms = stop.elapsed_ms_since(start);  // how long the kernel took
```

You **must** `synchronize()` on the later event before reading the elapsed time,
otherwise the GPU may not have finished yet and the number is meaningless.

---

## 6. How we verified Step 2

The verification target was: *create/destroy streams and events with no leaks
(`compute-sanitizer` clean).* What we ran:

1. **Compile a small exercise** that creates a stream, moves it, runs a no-op
   kernel, times it with events, move-assigns an event, and deliberately
   triggers an error (`cudaSetDevice(99999)`) to prove the throw + `file:line`
   path works.
2. **Confirm the GPU arch**: the `.cu` compiled to `sm_120` (the RTX 5090).
3. **Run under the leak checker:**
   ```bash
   compute-sanitizer --tool memcheck --leak-check full ./test
   ```
   Result: **`0 bytes leaked in 0 allocations`** → RAII releases everything.
   (The 2 "errors" it reported were our *intentional* invalid-device probe, not
   leaks.)

---

## 7. Concepts you learned in Step 2 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Status-code error handling | CUDA returns codes, not exceptions | We must check every call ourselves |
| `CUDA_CHECK` macro | wraps a call, throws with file/line | pinpoints failures instantly |
| `#expr`, `__FILE__`, `__LINE__` | preprocessor capture of call site | only a macro can see *where* it's used |
| RAII | ctor acquires, dtor releases | no leaks, even on exceptions |
| Move-only type | copy deleted, move allowed | single owner → no double-free |
| `std::exchange` | return old value, set new | clean ownership transfer in moves |
| CUDA stream | ordered GPU work queue | one stream drives the whole pipeline |
| CUDA event | marker in a stream | timing + synchronization |
| `noexcept` destructor cleanup | never throw while unwinding | avoids `std::terminate` |

---

## 8. One-sentence summary

Step 2 gives every later GPU file a safe foundation: `CUDA_CHECK`/`CUFFT_CHECK`
turn silent error codes into located exceptions, and `CudaStream`/`CudaEvent` are
RAII, move-only owners so GPU resources are created once and always released —
verified leak-free with `compute-sanitizer`.
