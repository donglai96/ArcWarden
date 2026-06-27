// ArcWarden — CUDA infrastructure (Step 2).
//
// Centralizes the three things every other GPU file needs:
//   - CUDA_CHECK / CUFFT_CHECK : error checking with file:line, throw on failure
//   - CudaStream               : RAII, move-only wrapper around cudaStream_t
//   - CudaEvent                : RAII, move-only wrapper around cudaEvent_t (timing)
//
// Ownership convention (design §2.1): long-lived resources are RAII, move-only,
// non-copyable. A single stream feeds the whole pipeline so later CUDA Graph
// capture (plan §12) has one capture target.

#ifndef ARC_PIC_CUDA_UTILS_HPP
#define ARC_PIC_CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace arc {

// ---- error reporting -------------------------------------------------------

// Thrown by CUDA_CHECK / CUFFT_CHECK so callers can unwind with RAII cleanup
// instead of leaking on a bare abort().
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};

inline const char* cufft_error_string(cufftResult r) {
    switch (r) {
        case CUFFT_SUCCESS:        return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:   return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:   return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:   return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:  return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:    return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:   return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:   return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE:   return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED:return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED:  return "CUFFT_NOT_SUPPORTED";
        default:                   return "CUFFT_UNKNOWN";
    }
}

namespace detail {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s:%d: CUDA error in `%s`: %s (%s)",
                      file, line, expr, cudaGetErrorName(err), cudaGetErrorString(err));
        throw CudaError(buf);
    }
}

inline void cufft_check(cufftResult res, const char* expr, const char* file, int line) {
    if (res != CUFFT_SUCCESS) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s:%d: cuFFT error in `%s`: %s",
                      file, line, expr, cufft_error_string(res));
        throw CudaError(buf);
    }
}

} // namespace detail

#define CUDA_CHECK(expr) ::arc::detail::cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUFFT_CHECK(expr) ::arc::detail::cufft_check((expr), #expr, __FILE__, __LINE__)

// ---- CudaStream ------------------------------------------------------------

// RAII, move-only. The single pipeline stream (design §2.2).
class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

    // Allow non-blocking streams (do not implicitly sync with the default stream).
    explicit CudaStream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() { reset(); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            reset();
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    void reset() noexcept {
        if (stream_) {
            // Best-effort in destructor: swallow errors rather than throw.
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
};

// ---- CudaEvent -------------------------------------------------------------

// RAII, move-only. Used for timing (cudaEventElapsedTime) and stream sync.
class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }

    explicit CudaEvent(unsigned int flags) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEvent() { reset(); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr)) {}

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            reset();
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }

    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }

    void record(cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() const { CUDA_CHECK(cudaEventSynchronize(event_)); }

    // Milliseconds elapsed from `start` to this event. Both must be recorded
    // and completed first (call synchronize() on the later event).
    float elapsed_ms_since(const CudaEvent& start) const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

private:
    void reset() noexcept {
        if (event_) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }

    cudaEvent_t event_ = nullptr;
};

} // namespace arc

#endif // ARC_PIC_CUDA_UTILS_HPP
