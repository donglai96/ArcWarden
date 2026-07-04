// ArcWarden — modular diagnostics: base interface + shared helpers.
//
// A diagnostic is a small object built from the deck that reads simulation state each
// step (via the concrete `const Fields&`/`const Particles&` that Simulation exposes) and
// writes its own output. Modules are NON-templated, so one DiagManager drives ES and
// Darwin runs alike. Each module owns its own cadence (it strides inside sample()).

#ifndef ARC_PIC_DIAG_MODULE_HPP
#define ARC_PIC_DIAG_MODULE_HPP

#include "pic/fields.hpp"        // Fields + DeviceArray + Grid
#include "pic/particles.hpp"
#include "pic/config.hpp"        // RunParams

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace arc { namespace diag {

// Live state handed to every module once per sampled step. The manager precomputes the
// B0-direction (∥) and v_th so modules don't each re-derive them.
struct DiagFrame {
    const Fields&    f;
    const Particles& p;
    const Grid&      g;
    const RunParams& rp;
    long   step;
    double t;
    double bmag;              // |B0|  (0 → unmagnetized: phase-space falls back to vx)
    double cost, sint;        // b̂_x, b̂_z  (B0 direction in the x-z plane)
    double vth;               // reference thermal velocity (species[0].uth[0])
    double vr;                // Landau resonant velocity / v_th  (0 if no pump)
};

struct IDiag {
    virtual void sample(const DiagFrame&) = 0;
    virtual void finalize() {}
    virtual ~IDiag() = default;
};

// ---- shared helpers (lifted from run_deck.cu / whistler_pump.cu) ----

// Σ v² over a device array, on the host.
inline double sum_sq(const DeviceArray<float>& a) {
    std::vector<float> h(a.size());
    cudaMemcpy(h.data(), a.data(), a.bytes(), cudaMemcpyDeviceToHost);
    double s = 0; for (float v : h) s += (double)v * v; return s;
}

// |E_x(mode m)|² from a host row of length nx (direct DFT, row j=0).
inline double mode_power(const std::vector<float>& ex, int nx, int m) {
    double re = 0, im = 0; const double w = 2.0 * M_PI * m / nx;
    for (int i = 0; i < nx; ++i) { re += ex[i]*std::cos(w*i); im -= ex[i]*std::sin(w*i); }
    return (re * re + im * im) / (double(nx) * nx);
}

// GPU phase-space histogram: bin (x, v_∥/v_th). v_∥ = v·b̂ with b̂=(bxn,0,bzn).
static __global__ void phase_hist_kernel(const float* px, const float* pux, const float* puz,
                                         long np, float* hist, int nxb, int nvb,
                                         int nx, float bxn, float bzn, float vth,
                                         float vlo, float vhi) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    float xx = px[i];
    if (xx < 0) xx += nx; else if (xx >= nx) xx -= nx;
    const float vp = (pux[i] * bxn + puz[i] * bzn) / vth;
    const int ix = (int)(xx / nx * nxb);
    const int iv = (int)((vp - vlo) / (vhi - vlo) * nvb);
    if (ix >= 0 && ix < nxb && iv >= 0 && iv < nvb)
        atomicAdd(&hist[iv * nxb + ix], 1.0f);
}

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_MODULE_HPP
