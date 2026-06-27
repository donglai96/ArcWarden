// ArcWarden — electrostatic spectral field solver (Step 8).
//
// ElectrostaticSpectralSolver does the full rho -> E in Fourier space, borrowing
// (not owning) the SpectralEngine (design §7.2). This is the host orchestration
// layer: called once per step, so a concrete class (no virtual, no variant yet)
// is cleanest; a DarwinSpectralSolver with the same solve(...) signature slots in
// later via std::variant with zero churn.
//
// Pipeline:
//   1. r2c:  rho (real [ny][nx]) -> rho_k (half-complex [ny][nkx])
//   2. k-space kernel (compile-time-fixed logic):
//        phi_k = rho_k / (eps0 k²)          via SpectralFormFactor::green
//          - k=0 (k²=0): green returns 0  -> removes DC = neutralizing background
//          - Nyquist (kx=nx/2 or ky=ny/2): force zeroed (the -i k first-derivative
//            operator is ill-posed on a pure-real Nyquist mode)
//        Ex_k = -i kx phi_k ;  Ey_k = -i ky phi_k       (complex: -i k (a+ib) = (k b, -k a))
//   3. c2r:  Ex_k -> Ex, Ey_k -> Ey   (SpectralEngine::c2r applies 1/(nx*ny))
//
// NORMALIZATION CONTRACT: see config.hpp / grid.hpp. eps0 comes from RunParams.

#ifndef ARC_PIC_SOLVER_ES_HPP
#define ARC_PIC_SOLVER_ES_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/sources.hpp"
#include "pic/spectral.hpp"

#include <cufft.h>

namespace arc {
namespace detail {

// Templated on the form-factor type for vague linkage (ODR-safe in a header).
// One complex element (i,j) of the half-spectrum per thread.
template<class FF>
__global__ void es_poisson_field_kernel(DeviceView<cufftComplex> rho_k,
                                        DeviceView<cufftComplex> phi_k,
                                        DeviceView<cufftComplex> Ex_k,
                                        DeviceView<cufftComplex> Ey_k,
                                        KGrid kg, FF ff, int nx, int ny) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= kg.complex_size()) return;

    const int i = t % kg.nkx;   // x: reduced axis [0, nkx)
    const int j = t / kg.nkx;   // y: full axis    [0, ny)

    const cufftComplex r  = rho_k[t];
    const double       k2 = kg.k2(i, j);
    const double       g  = ff.green(k2) * ff.smoothing(k2);  // 1/(eps0 k²); 0 at DC

    cufftComplex phi;
    phi.x = static_cast<float>(r.x * g);
    phi.y = static_cast<float>(r.y * g);
    phi_k[t] = phi;

    // Zero the FORCE on the Nyquist mode(s) (even-length axes only).
    const bool nyq = ((nx % 2 == 0) && i == nx / 2) ||
                     ((ny % 2 == 0) && j == ny / 2);

    cufftComplex ex, ey;
    if (nyq) {
        ex.x = ex.y = ey.x = ey.y = 0.0f;
    } else {
        const double kx = kg.kx(i);
        const double ky = kg.ky(j);
        // E_k = -i k phi  =>  (k * phi.y, -k * phi.x)
        ex.x = static_cast<float>( kx * phi.y);
        ex.y = static_cast<float>(-kx * phi.x);
        ey.x = static_cast<float>( ky * phi.y);
        ey.y = static_cast<float>(-ky * phi.x);
    }
    Ex_k[t] = ex;
    Ey_k[t] = ey;
}

} // namespace detail

class ElectrostaticSpectralSolver {
public:
    // Full rho -> E. Borrows the engine (its workspace holds rho_k/phi_k/Ex_k/Ey_k).
    void solve(const Sources& src, Fields& fld, SpectralEngine& eng,
               const RunParams& rp, cudaStream_t s) const {
        SpectralWorkspace& ws = eng.ws();

        // 1. forward transform rho -> rho_k
        eng.r2c(src.rho.data(), ws.rho_k.data(), s);

        // 2. k-space Poisson + field operator
        const KGrid        kg = eng.kgrid();
        SpectralFormFactor ff{ rp.eps0 };
        const int          Nk = kg.complex_size();
        constexpr int      threads = 256;
        const int          blocks  = (Nk + threads - 1) / threads;
        detail::es_poisson_field_kernel<SpectralFormFactor><<<blocks, threads, 0, s>>>(
            ws.rho_k.view(), ws.phi_k.view(), ws.Ex_k.view(), ws.Ey_k.view(),
            kg, ff, eng.grid().nx, eng.grid().ny);
        CUDA_CHECK(cudaPeekAtLastError());

        // 3. inverse transforms (normalized); c2r consumes Ex_k / Ey_k
        eng.c2r(ws.Ex_k.data(), fld.Ex.data(), s);
        eng.c2r(ws.Ey_k.data(), fld.Ey.data(), s);
    }
};

} // namespace arc

#endif // ARC_PIC_SOLVER_ES_HPP
