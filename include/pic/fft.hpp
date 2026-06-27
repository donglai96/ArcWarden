// ArcWarden — cuFFT plan ownership (Step 6).
//
// CufftPlan2D: an owning, RAII, move-only wrapper around the *pair* of cuFFT
// 2D plans needed for the spectral solver — one CUFFT_R2C (forward, real ->
// half-complex) and one CUFFT_C2R (inverse). Built once for a fixed (ny,nx)
// and reused every step; a cuFFT plan only depends on the transform size, not
// on the data, so it is pure overhead to rebuild it (design §7.1).
//
// Layout: the plan is created as cufftPlan2d(&p, ny, nx) — cuFFT takes the
// dimensions SLOWEST-first, matching our real [ny][nx] layout where x (the nx
// axis) is contiguous and gets halved to nkx=nx/2+1 by R2C (see grid.hpp R2C
// contract). cuFFT does NOT normalize; the 1/(nx*ny) factor is applied by the
// SpectralEngine on the inverse (spectral.hpp).

#ifndef ARC_PIC_FFT_HPP
#define ARC_PIC_FFT_HPP

#include "pic/cuda_utils.hpp"

#include <cufft.h>

#include <utility>

namespace arc {

class CufftPlan2D {
public:
    CufftPlan2D() = default;

    // Build both directions for an (nx,ny) real grid. cuFFT wants (ny,nx).
    CufftPlan2D(int nx, int ny) : nx_(nx), ny_(ny) {
        CUFFT_CHECK(cufftPlan2d(&r2c_, ny, nx, CUFFT_R2C));
        CUFFT_CHECK(cufftPlan2d(&c2r_, ny, nx, CUFFT_C2R));
        created_ = true;
    }

    ~CufftPlan2D() { reset(); }

    CufftPlan2D(const CufftPlan2D&)            = delete;  // move-only: a plan owns
    CufftPlan2D& operator=(const CufftPlan2D&) = delete;  // GPU-side resources

    CufftPlan2D(CufftPlan2D&& o) noexcept
        : r2c_(o.r2c_), c2r_(o.c2r_),
          nx_(std::exchange(o.nx_, 0)), ny_(std::exchange(o.ny_, 0)),
          created_(std::exchange(o.created_, false)) {
        o.r2c_ = o.c2r_ = 0;
    }

    CufftPlan2D& operator=(CufftPlan2D&& o) noexcept {
        if (this != &o) {
            reset();
            r2c_ = o.r2c_; c2r_ = o.c2r_;
            nx_ = std::exchange(o.nx_, 0);
            ny_ = std::exchange(o.ny_, 0);
            created_ = std::exchange(o.created_, false);
            o.r2c_ = o.c2r_ = 0;
        }
        return *this;
    }

    // Forward: real [ny][nx] -> half-complex [ny][nkx]. cuFFT is unnormalized.
    void exec_r2c(const float* in, cufftComplex* out, cudaStream_t s) const {
        CUFFT_CHECK(cufftSetStream(r2c_, s));
        // cuFFT's signature is non-const; the R2C input is read-only in practice.
        CUFFT_CHECK(cufftExecR2C(r2c_, const_cast<cufftReal*>(in), out));
    }

    // Inverse: half-complex [ny][nkx] -> real [ny][nx]. NOTE: cuFFT C2R is
    // destructive on its input buffer. Still unnormalized here.
    void exec_c2r(cufftComplex* in, float* out, cudaStream_t s) const {
        CUFFT_CHECK(cufftSetStream(c2r_, s));
        CUFFT_CHECK(cufftExecC2R(c2r_, in, out));
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }

private:
    void reset() noexcept {
        if (created_) {
            // Best-effort in destructor: swallow errors rather than throw.
            cufftDestroy(r2c_);
            cufftDestroy(c2r_);
        }
        r2c_ = c2r_ = 0;
        nx_ = ny_ = 0;
        created_ = false;
    }

    cufftHandle r2c_ = 0, c2r_ = 0;
    int  nx_ = 0, ny_ = 0;
    bool created_ = false;  // explicit validity flag (0 is not a safe handle sentinel)
};

} // namespace arc

#endif // ARC_PIC_FFT_HPP
