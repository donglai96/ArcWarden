// ArcWarden — external pump field + background B0 injection (EM/Darwin driver).
//
// Reproduces the An et al. (2019) whistler-pump setup: an external traveling-wave
// electric field drives the plasma for t < t_off, exciting a whistler whose
// magnetic field then forms self-consistently through the Darwin solve. Two helpers:
//
//   add_pump : E_total += Re{ Ẽ_α e^{i(k0 x − w0 t)} }·ramp(t)   (α = x,y,z)
//   add_b0   : B_total += background B0   (so the push/dcu gather see total B)
//
// Both run on the real-space grid each step AFTER the self-consistent field solve
// and BEFORE the dcu deposit / push, so the total (self + external) fields drive
// the particles — exactly the paper's prescription. Pump is 1D along x (depends on
// the x cell index only); the y spatial axis is the trivial periodic direction.

#ifndef ARC_PIC_PUMP_HPP
#define ARC_PIC_PUMP_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"

namespace arc {
namespace detail {

// Trapezoidal ramp (S2): linear up to t_rmp, flat, linear down to t_off, then 0.
__host__ __device__ inline double pump_ramp(double t, double trmp, double toff) {
    if (t >= toff) return 0.0;
    if (t < trmp)  return t / trmp;
    if (t > toff - trmp) return (toff - t) / trmp;
    return 1.0;
}

template<class Dummy = void>
__global__ void add_pump_kernel(FieldViews f, Grid g, RunParams rp, double t) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= g.nx * g.ny) return;
    const int i = c % g.nx;                 // x cell (pump is 1D along x)
    const double x  = i * g.dx;
    const double th = rp.pump_k0 * x - rp.pump_w0 * t;
    const double r  = pump_ramp(t, rp.pump_trmp, rp.pump_toff);
    const double ct = cos(th), st = sin(th);
    // Re{ Ẽx e^iθ } = Ex·cosθ ; Re{ i·Ey e^iθ } = −Ey·sinθ ; Re{ Ez e^iθ } = Ez·cosθ
    f.Ex[c] += static_cast<float>(rp.pump_ex * ct * r);
    f.Ey[c] += static_cast<float>(-rp.pump_ey * st * r);
    f.Ez[c] += static_cast<float>(rp.pump_ez * ct * r);
}

template<class Dummy = void>
__global__ void add_b0_kernel(FieldViews f, int n, float bx, float by, float bz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f.Bx[i] += bx; f.By[i] += by; f.Bz[i] += bz;
}

} // namespace detail

// Add the background B0 to the (self-consistent) magnetic field on the grid.
inline void add_background_b0(Fields& fld, const Grid& g, const RunParams& rp,
                              cudaStream_t s) {
    const int n = g.real_size();
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    detail::add_b0_kernel<><<<blocks, threads, 0, s>>>(
        fld.views(), n, rp.B0[0], rp.B0[1], rp.B0[2]);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Add the external pump E to the total E on the grid (no-op if rp.pump is off or
// t >= t_off).
inline void add_pump_field(Fields& fld, const Grid& g, const RunParams& rp,
                           double t, cudaStream_t s) {
    if (!rp.pump || t >= rp.pump_toff) return;
    const int n = g.real_size();
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    detail::add_pump_kernel<><<<blocks, threads, 0, s>>>(fld.views(), g, rp, t);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace arc

#endif // ARC_PIC_PUMP_HPP
