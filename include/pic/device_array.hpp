// ArcWarden — GPU memory ownership (Step 3).
//
// Two types, deliberately separated (design §2.1):
//
//   DeviceArray<T>  — OWNING. Host-side, RAII, move-only. Wraps a plain
//                     cudaMalloc/cudaFree allocation. NEVER passed into a kernel.
//   DeviceView<T>   — NON-OWNING. POD, trivially copyable, no destructor. The
//                     lightweight handle that IS passed by value into kernels.
//
// Allocation policy: long-lived arrays (particles, Sources, Fields, the FFT
// workspace) use plain cudaMalloc/cudaFree — simple, no stream-lifetime traps,
// and friendly to later CUDA Graph capture. Per-step async scratch
// (cudaMallocAsync + mempool) is deferred; v1 has essentially none.

#ifndef ARC_PIC_DEVICE_ARRAY_HPP
#define ARC_PIC_DEVICE_ARRAY_HPP

#include "pic/cuda_utils.hpp"

#include <cstddef>
#include <utility>

namespace arc {

// ---- DeviceView<T> : non-owning POD handle for kernels -------------------

// Passed BY VALUE into __global__ functions. No ownership, no destructor.
// operator[] is device-only: the pointer is GPU memory and must not be
// dereferenced on the host.
template<class T>
struct DeviceView {
    T*  ptr = nullptr;
    int n   = 0;

    __device__ T&       operator[](int i)       { return ptr[i]; }
    __device__ const T& operator[](int i) const { return ptr[i]; }

    __host__ __device__ int  size()  const { return n; }
    __host__ __device__ bool empty() const { return n == 0; }
};

// ---- DeviceArray<T> : owning, RAII, move-only ---------------------------

template<class T>
class DeviceArray {
public:
    DeviceArray() = default;

    // Allocates n elements with plain cudaMalloc (uninitialized).
    explicit DeviceArray(std::size_t n) : n_(n) {
        if (n_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
        }
    }

    ~DeviceArray() { reset(); }

    DeviceArray(const DeviceArray&)            = delete;  // unique ownership
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          n_(std::exchange(other.n_, 0)) {}

    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
            n_   = std::exchange(other.n_, 0);
        }
        return *this;
    }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }

    std::size_t size()  const { return n_; }
    std::size_t bytes() const { return n_ * sizeof(T); }
    bool        empty() const { return n_ == 0; }

    // Stream-ordered zero-fill (byte-wise; valid because our T are PODs whose
    // all-zero bit pattern means 0 / {0,0} for cufftComplex).
    void zero(cudaStream_t stream = nullptr) {
        if (ptr_ && n_ > 0) {
            CUDA_CHECK(cudaMemsetAsync(ptr_, 0, n_ * sizeof(T), stream));
        }
    }

    DeviceView<T> view() {
        return DeviceView<T>{ptr_, static_cast<int>(n_)};
    }
    DeviceView<const T> view() const {
        return DeviceView<const T>{ptr_, static_cast<int>(n_)};
    }

private:
    void reset() noexcept {
        if (ptr_) {
            cudaFree(ptr_);  // best-effort in destructor: do not throw
            ptr_ = nullptr;
        }
        n_ = 0;
    }

    T*          ptr_ = nullptr;
    std::size_t n_   = 0;
};

} // namespace arc

#endif // ARC_PIC_DEVICE_ARRAY_HPP
