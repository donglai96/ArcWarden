// ArcWarden — spectral engine (Step 6).
//
// Concentrates everything FFT-related behind one host-side owning object so the
// field solver (Step 8) just borrows it and never touches a raw cuFFT call
// (design §7.1). Three pieces:
//
//   SpectralWorkspace  — long-lived k-space buffers (rho_k/phi_k/Ex_k/Ey_k),
//                        allocated once at construction.
//   SpectralFormFactor — Green's function g(k) and particle-shape smoothing
//                        s(k) in one place (UPIC ffc/ffe idea). v1: s(k)=1,
//                        g(k)=1/(eps0 k²). Keeping 1/k² here (not inlined into
//                        the solver) means smoothing/filtering is a one-spot
//                        change later.
//   SpectralEngine     — owns the CufftPlan2D + workspace + KGrid + form
//                        factor; exposes r2c() (raw forward) and c2r()
//                        (inverse + the 1/(nx*ny) normalization cuFFT omits).
//
// Normalization contract (grid.hpp / design §3): cuFFT does not scale, so a
// round trip c2r(r2c(f)) returns (nx*ny)*f. c2r() applies 1/(nx*ny) so callers
// get f back directly.

#ifndef ARC_PIC_SPECTRAL_HPP
#define ARC_PIC_SPECTRAL_HPP

#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/fft.hpp"
#include "pic/grid.hpp"

#include <cufft.h>

namespace arc {

// ---- k-space buffers (allocated once, reused every step) ------------------

struct SpectralWorkspace {
    DeviceArray<cufftComplex> rho_k;
    DeviceArray<cufftComplex> phi_k;
    DeviceArray<cufftComplex> Ex_k, Ey_k;
    // Darwin k-space (allocated only by allocate_em): current, magnetic field,
    // transverse E, and the dcu/amu transforms used to build E_T.
    DeviceArray<cufftComplex> Jx_k, Jy_k, Jz_k;
    DeviceArray<cufftComplex> Bx_k, By_k, Bz_k;
    DeviceArray<cufftComplex> ETx_k, ETy_k, ETz_k;
    DeviceArray<cufftComplex> dcux_k, dcuy_k, dcuz_k;
    DeviceArray<cufftComplex> amu0_k, amu1_k, amu2_k, amu3_k;

    SpectralWorkspace() = default;
    explicit SpectralWorkspace(int complex_size)
        : rho_k(complex_size), phi_k(complex_size),
          Ex_k(complex_size), Ey_k(complex_size) {}

    void allocate_em(int nk) {
        Jx_k = DeviceArray<cufftComplex>(nk); Jy_k = DeviceArray<cufftComplex>(nk); Jz_k = DeviceArray<cufftComplex>(nk);
        Bx_k = DeviceArray<cufftComplex>(nk); By_k = DeviceArray<cufftComplex>(nk); Bz_k = DeviceArray<cufftComplex>(nk);
        ETx_k = DeviceArray<cufftComplex>(nk); ETy_k = DeviceArray<cufftComplex>(nk); ETz_k = DeviceArray<cufftComplex>(nk);
        dcux_k = DeviceArray<cufftComplex>(nk); dcuy_k = DeviceArray<cufftComplex>(nk); dcuz_k = DeviceArray<cufftComplex>(nk);
        amu0_k = DeviceArray<cufftComplex>(nk); amu1_k = DeviceArray<cufftComplex>(nk);
        amu2_k = DeviceArray<cufftComplex>(nk); amu3_k = DeviceArray<cufftComplex>(nk);
    }
};

// ---- Green's function + shape smoothing (one place) -----------------------

// POD with __host__ __device__ accessors so the solver kernel can use it by
// value. v1: smoothing(k)=1, green(k)=1/(eps0 k²); DC (k2==0) returns 0 so the
// caller never divides by zero (the k=0 mode is set to 0 anyway, design §7.2).
struct SpectralFormFactor {
    double eps0 = 1.0;
    double c    = 1.0;   // speed of light (Darwin); μ₀ = 1/(eps0 c²)
    double n0   = 1.0;   // background density → ω_pe² (= n0 q²/ε₀m = n0 here)

    __host__ __device__ double smoothing(double /*k2*/) const { return 1.0; }

    // Poisson Green's function 1/(eps0 k²) (UPIC ffc). DC → 0.
    __host__ __device__ double green(double k2) const {
        return k2 > 0.0 ? 1.0 / (eps0 * k2) : 0.0;
    }

    // Darwin magnetic Green's function μ₀/k² (B_k = i μ₀ (k×J)/k²). DC → 0.
    __host__ __device__ double green_t(double k2) const {
        return k2 > 0.0 ? 1.0 / (eps0 * c * c * k2) : 0.0;
    }

    // Darwin transverse-E Green's function (UPIC ffe): E_T_k = -green_et·(∂J/∂t)_T
    // where ∂J/∂t EXCLUDES the (q/m)ρE_T self-term — that term is RESUMMED into the
    // denominator: green_et = μ₀/(k² + μ₀ω_pe²) = 1/(ε₀c²k² + n0). Bounded by 1/n0,
    // so the solve is stable (the naive μ₀/k² diverges at low k). DC → 0.
    __host__ __device__ double green_et(double k2) const {
        return k2 > 0.0 ? 1.0 / (eps0 * c * c * k2 + n0) : 0.0;
    }
};

// ---- in-place scale kernel (templated => vague linkage, ODR-safe in a header)

namespace detail {
template<class T>
__global__ void scale_inplace_kernel(T* a, int n, T s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}
} // namespace detail

// ---- SpectralEngine : owns plan + workspace + kgrid + form factor ----------

class SpectralEngine {
public:
    // batch is reserved for later Darwin multi-field batched FFTs; v1 = 1.
    explicit SpectralEngine(const Grid& g, int batch = 1)
        : grid_(g),
          kgrid_(g),
          plan_(g.nx, g.ny),
          ws_(kgrid_.complex_size()) {
        (void)batch;
    }

    // Forward: real [ny][nx] -> half-complex [ny][nkx]. Unnormalized (raw cuFFT).
    void r2c(const float* in, cufftComplex* out, cudaStream_t s) const {
        plan_.exec_r2c(in, out, s);
    }

    // Inverse: half-complex [ny][nkx] -> real [ny][nx], normalized by 1/(nx*ny).
    // NOTE: cuFFT C2R consumes (overwrites) the input buffer.
    void c2r(cufftComplex* in, float* out, cudaStream_t s) const {
        plan_.exec_c2r(in, out, s);
        const int   n     = grid_.real_size();
        const float scale = 1.0f / static_cast<float>(n);
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        detail::scale_inplace_kernel<float><<<blocks, threads, 0, s>>>(out, n, scale);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Allocate the Darwin k-space buffers (call once for FieldModel::Darwin).
    void enable_em() { ws_.allocate_em(kgrid_.complex_size()); }

    SpectralWorkspace&       ws()       { return ws_; }
    const SpectralWorkspace& ws() const { return ws_; }
    const Grid&              grid()  const { return grid_; }
    const KGrid&             kgrid() const { return kgrid_; }
    const SpectralFormFactor& form_factor() const { return ff_; }

private:
    Grid               grid_;
    KGrid              kgrid_;
    CufftPlan2D        plan_;   // owning: created once, reused
    SpectralWorkspace  ws_;     // owning: k-space buffers
    SpectralFormFactor ff_;     // v1 defaults (eps0=1)
};

} // namespace arc

#endif // ARC_PIC_SPECTRAL_HPP
