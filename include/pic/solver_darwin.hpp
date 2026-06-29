// ArcWarden — spectral Darwin field solver (EM stage).
//
// DarwinSpectralSolver does the full Darwin field solve in Fourier space,
// borrowing (not owning) the SpectralEngine — the EM sibling of
// ElectrostaticSpectralSolver (solver_es.hpp), same solve(...) shape so the
// Simulation loop swaps it in via Cfg::field_model with no churn.
//
// Darwin model (no radiation; UPIC mpdbeps2 lineage). Three field pieces:
//   E_L  longitudinal electric  : E_L_k = -i k ρ_k/(ε₀ k²)   (reuses the ES kernel)
//   B    magnetic               : B_k   = i μ₀ (k × J)/k²     (Coulomb gauge; k×J_T=k×J)
//   E_T  transverse inductive   : solved from dcu/amu in the ndc iteration (Phase E)
// The push gathers E_total = E_L + E_T and B.
//
// PHASE C scope: E_L + B only (E_T ≡ 0). Phase E adds the transverse iteration.
//
// k×J with k=(kx,ky,0), J=(Jx,Jy,Jz):  (ky Jz, -kx Jz, kx Jy - ky Jx).
// Multiplying a complex c by i:  i(a+ib) = (-b + i a). μ₀ = 1/(ε₀ c²) = green_t·k².

#ifndef ARC_PIC_SOLVER_DARWIN_HPP
#define ARC_PIC_SOLVER_DARWIN_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/device_array.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/solver_es.hpp"   // es_poisson_field_kernel (E_L)
#include "pic/sources.hpp"
#include "pic/spectral.hpp"

#include <cufft.h>

namespace arc {
namespace detail {

// B_k = i μ₀ (k × J_k)/k². One half-spectrum element per thread. DC and Nyquist
// modes are zeroed (the i·k operator is ill-posed there, as in the ES solver).
template<class FF>
__global__ void darwin_b_field_kernel(DeviceView<cufftComplex> Jx_k,
                                      DeviceView<cufftComplex> Jy_k,
                                      DeviceView<cufftComplex> Jz_k,
                                      DeviceView<cufftComplex> Bx_k,
                                      DeviceView<cufftComplex> By_k,
                                      DeviceView<cufftComplex> Bz_k,
                                      KGrid kg, FF ff, int nx, int ny) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= kg.complex_size()) return;
    const int i = t % kg.nkx;
    const int j = t / kg.nkx;

    const double k2  = kg.k2(i, j);
    const double G   = ff.green_t(k2);   // μ₀/k²; 0 at DC
    const bool   nyq = ((nx % 2 == 0) && i == nx / 2) ||
                       ((ny % 2 == 0) && j == ny / 2);

    cufftComplex bx, by, bz;
    if (nyq || k2 == 0.0) {
        bx.x = bx.y = by.x = by.y = bz.x = bz.y = 0.0f;
    } else {
        const double kx = kg.kx(i), ky = kg.ky(j);
        const cufftComplex Jx = Jx_k[t], Jy = Jy_k[t], Jz = Jz_k[t];
        // Bx = i G ky Jz
        bx.x = static_cast<float>(-G * ky * Jz.y);
        bx.y = static_cast<float>( G * ky * Jz.x);
        // By = -i G kx Jz
        by.x = static_cast<float>( G * kx * Jz.y);
        by.y = static_cast<float>(-G * kx * Jz.x);
        // Bz = i G (kx Jy - ky Jx)
        const double Cx = kx * Jy.x - ky * Jx.x;
        const double Cy = kx * Jy.y - ky * Jx.y;
        bz.x = static_cast<float>(-G * Cy);
        bz.y = static_cast<float>( G * Cx);
    }
    Bx_k[t] = bx; By_k[t] = by; Bz_k[t] = bz;
}

// Transverse electric field (Phase E):  E_T_k = -μ₀ (∂J/∂t)_T / k², with
//   ∂J/∂t = dcu - ∇·Π,   ∇·Π reconstructed from the 4 deviatoric amu comps:
//     (∇·Π)_x = i(kx amu0 + ky amu1), (∇·Π)_y = i(kx amu1 - ky amu0),
//     (∇·Π)_z = i(kx amu2 + ky amu3),
// then transverse-projected  V_T = V - k(k·V)/k²  (k=(kx,ky,0); z already transverse).
// DC and Nyquist zeroed. (i·(a+ib) = (-b + i a).)
template<class FF>
__global__ void darwin_et_kernel(DeviceView<cufftComplex> dcux_k,
                                 DeviceView<cufftComplex> dcuy_k,
                                 DeviceView<cufftComplex> dcuz_k,
                                 DeviceView<cufftComplex> amu0_k,
                                 DeviceView<cufftComplex> amu1_k,
                                 DeviceView<cufftComplex> amu2_k,
                                 DeviceView<cufftComplex> amu3_k,
                                 DeviceView<cufftComplex> ETx_k,
                                 DeviceView<cufftComplex> ETy_k,
                                 DeviceView<cufftComplex> ETz_k,
                                 KGrid kg, FF ff, int nx, int ny) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= kg.complex_size()) return;
    const int i = t % kg.nkx, j = t / kg.nkx;
    const double k2 = kg.k2(i, j);
    const double G  = ff.green_et(k2);   // resummed transverse-E Green's fn (ffe)
    const bool nyq = ((nx % 2 == 0) && i == nx / 2) || ((ny % 2 == 0) && j == ny / 2);

    cufftComplex ex, ey, ez;
    if (nyq || k2 == 0.0) {
        ex.x = ex.y = ey.x = ey.y = ez.x = ez.y = 0.0f;
    } else {
        const double kx = kg.kx(i), ky = kg.ky(j);
        const cufftComplex a0 = amu0_k[t], a1 = amu1_k[t], a2 = amu2_k[t], a3 = amu3_k[t];
        // S = (∇·Π)/i  (so ∇·Π = i S)
        const double Sx_r = kx*a0.x + ky*a1.x, Sx_i = kx*a0.y + ky*a1.y;
        const double Sy_r = kx*a1.x - ky*a0.x, Sy_i = kx*a1.y - ky*a0.y;
        const double Sz_r = kx*a2.x + ky*a3.x, Sz_i = kx*a2.y + ky*a3.y;
        const cufftComplex dcx = dcux_k[t], dcy = dcuy_k[t], dcz = dcuz_k[t];
        // dJdt = dcu - i S      (i S = (-S_i, S_r))
        double Jx_r = dcx.x + Sx_i, Jx_i = dcx.y - Sx_r;
        double Jy_r = dcy.x + Sy_i, Jy_i = dcy.y - Sy_r;
        double Jz_r = dcz.x + Sz_i, Jz_i = dcz.y - Sz_r;
        // transverse projection (only x,y; z already ⟂ since kz=0)
        const double inv = 1.0 / k2;
        const double kd_r = (kx*Jx_r + ky*Jy_r) * inv;
        const double kd_i = (kx*Jx_i + ky*Jy_i) * inv;
        Jx_r -= kx*kd_r; Jx_i -= kx*kd_i;
        Jy_r -= ky*kd_r; Jy_i -= ky*kd_i;
        // E_T = -G dJdt_T
        ex.x = static_cast<float>(-G*Jx_r); ex.y = static_cast<float>(-G*Jx_i);
        ey.x = static_cast<float>(-G*Jy_r); ey.y = static_cast<float>(-G*Jy_i);
        ez.x = static_cast<float>(-G*Jz_r); ez.y = static_cast<float>(-G*Jz_i);
    }
    ETx_k[t] = ex; ETy_k[t] = ey; ETz_k[t] = ez;
}

// E = E_L + et_scale·E_T (real space). E_L has no z component. et_scale=0 gives
// the field for the dcu deposit (E_L + pump, no E_T — the self-term is resummed in
// green_et); et_scale=1 gives the total E for the push.
template<class Dummy = void>
__global__ void combine_etotal_kernel(float* Ex, float* Ey, float* Ez,
                                      const float* ELx, const float* ELy,
                                      const float* ETx, const float* ETy, const float* ETz,
                                      float et_scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Ex[i] = ELx[i] + et_scale * ETx[i];
    Ey[i] = ELy[i] + et_scale * ETy[i];
    Ez[i] = et_scale * ETz[i];
}

} // namespace detail

class DarwinSpectralSolver {
public:
    // PHASE C: solve E_L (ρ → E, reusing the ES kernel) and B (J → B). E_total =
    // E_L here (E_T added in Phase E). Ez is forced 0 (longitudinal E has no z in
    // 2D and E_T is not yet active). Borrows the engine's k-space workspace.
    void solve(const Sources& src, Fields& fld, SpectralEngine& eng,
               const RunParams& rp, cudaStream_t s) const {
        SpectralWorkspace& ws = eng.ws();
        const KGrid kg = eng.kgrid();
        SpectralFormFactor ff{ rp.eps0, rp.c };
        const int Nk = kg.complex_size();
        constexpr int threads = 256;
        const int blocks = (Nk + threads - 1) / threads;

        // --- longitudinal E_L from ρ (reuse the ES Poisson/field kernel) ---
        eng.r2c(src.rho.data(), ws.rho_k.data(), s);
        detail::es_poisson_field_kernel<SpectralFormFactor><<<blocks, threads, 0, s>>>(
            ws.rho_k.view(), ws.phi_k.view(), ws.Ex_k.view(), ws.Ey_k.view(),
            kg, ff, eng.grid().nx, eng.grid().ny);
        CUDA_CHECK(cudaPeekAtLastError());
        eng.c2r(ws.Ex_k.data(), fld.Ex.data(), s);   // E_total x  (= E_L, E_T=0)
        eng.c2r(ws.Ey_k.data(), fld.Ey.data(), s);   // E_total y
        fld.Ez.zero(s);

        // --- magnetic field B from current J ---
        eng.r2c(src.Jx.data(), ws.Jx_k.data(), s);
        eng.r2c(src.Jy.data(), ws.Jy_k.data(), s);
        eng.r2c(src.Jz.data(), ws.Jz_k.data(), s);
        detail::darwin_b_field_kernel<SpectralFormFactor><<<blocks, threads, 0, s>>>(
            ws.Jx_k.view(), ws.Jy_k.view(), ws.Jz_k.view(),
            ws.Bx_k.view(), ws.By_k.view(), ws.Bz_k.view(),
            kg, ff, eng.grid().nx, eng.grid().ny);
        CUDA_CHECK(cudaPeekAtLastError());
        eng.c2r(ws.Bx_k.data(), fld.Bx.data(), s);
        eng.c2r(ws.By_k.data(), fld.By.data(), s);
        eng.c2r(ws.Bz_k.data(), fld.Bz.data(), s);
    }

    // Full-Darwin step pieces (Phase E). solve_el_b writes E_L into fld.ELx/ELy
    // (kept so E_total can be re-formed each ndc iteration) and B into fld.Bx/By/Bz.
    void solve_el_b(const Sources& src, Fields& fld, SpectralEngine& eng,
                    const RunParams& rp, cudaStream_t s) const {
        SpectralWorkspace& ws = eng.ws();
        const KGrid kg = eng.kgrid();
        SpectralFormFactor ff{ rp.eps0, rp.c };
        const int Nk = kg.complex_size();
        constexpr int threads = 256;
        const int blocks = (Nk + threads - 1) / threads;

        eng.r2c(src.rho.data(), ws.rho_k.data(), s);
        detail::es_poisson_field_kernel<SpectralFormFactor><<<blocks, threads, 0, s>>>(
            ws.rho_k.view(), ws.phi_k.view(), ws.Ex_k.view(), ws.Ey_k.view(),
            kg, ff, eng.grid().nx, eng.grid().ny);
        CUDA_CHECK(cudaPeekAtLastError());
        eng.c2r(ws.Ex_k.data(), fld.ELx.data(), s);
        eng.c2r(ws.Ey_k.data(), fld.ELy.data(), s);

        eng.r2c(src.Jx.data(), ws.Jx_k.data(), s);
        eng.r2c(src.Jy.data(), ws.Jy_k.data(), s);
        eng.r2c(src.Jz.data(), ws.Jz_k.data(), s);
        detail::darwin_b_field_kernel<SpectralFormFactor><<<blocks, threads, 0, s>>>(
            ws.Jx_k.view(), ws.Jy_k.view(), ws.Jz_k.view(),
            ws.Bx_k.view(), ws.By_k.view(), ws.Bz_k.view(),
            kg, ff, eng.grid().nx, eng.grid().ny);
        CUDA_CHECK(cudaPeekAtLastError());
        eng.c2r(ws.Bx_k.data(), fld.Bx.data(), s);
        eng.c2r(ws.By_k.data(), fld.By.data(), s);
        eng.c2r(ws.Bz_k.data(), fld.Bz.data(), s);
    }

    // Form fld.E = E_L + et_scale·E_T. et_scale=0 → E_L only (dcu gather field,
    // before adding pump); et_scale=1 → total E (push field).
    void form_e(Fields& fld, float et_scale, cudaStream_t s) const {
        const int n = static_cast<int>(fld.Ex.size());
        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        detail::combine_etotal_kernel<><<<blocks, threads, 0, s>>>(
            fld.Ex.data(), fld.Ey.data(), fld.Ez.data(),
            fld.ELx.data(), fld.ELy.data(), fld.ETx.data(), fld.ETy.data(), fld.ETz.data(),
            et_scale, n);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    void form_el(Fields& fld, cudaStream_t s)     const { form_e(fld, 0.0f, s); }
    void form_etotal(Fields& fld, cudaStream_t s) const { form_e(fld, 1.0f, s); }

    // One-shot transverse-field solve: dcu,amu (deposited with E_L+pump, NOT E_T) →
    // E_T into fld.ETx/ETy/ETz via the resummed green_et. Caller forms E_total after.
    void solve_et(const Sources& src, Fields& fld, SpectralEngine& eng,
                  const RunParams& rp, cudaStream_t s) const {
        SpectralWorkspace& ws = eng.ws();
        const KGrid kg = eng.kgrid();
        SpectralFormFactor ff{ rp.eps0, rp.c, rp.n0 };
        const int Nk = kg.complex_size();
        constexpr int threads = 256;
        const int blocks = (Nk + threads - 1) / threads;

        eng.r2c(src.dcux.data(), ws.dcux_k.data(), s);
        eng.r2c(src.dcuy.data(), ws.dcuy_k.data(), s);
        eng.r2c(src.dcuz.data(), ws.dcuz_k.data(), s);
        eng.r2c(src.amu0.data(), ws.amu0_k.data(), s);
        eng.r2c(src.amu1.data(), ws.amu1_k.data(), s);
        eng.r2c(src.amu2.data(), ws.amu2_k.data(), s);
        eng.r2c(src.amu3.data(), ws.amu3_k.data(), s);

        detail::darwin_et_kernel<SpectralFormFactor><<<blocks, threads, 0, s>>>(
            ws.dcux_k.view(), ws.dcuy_k.view(), ws.dcuz_k.view(),
            ws.amu0_k.view(), ws.amu1_k.view(), ws.amu2_k.view(), ws.amu3_k.view(),
            ws.ETx_k.view(), ws.ETy_k.view(), ws.ETz_k.view(),
            kg, ff, eng.grid().nx, eng.grid().ny);
        CUDA_CHECK(cudaPeekAtLastError());

        eng.c2r(ws.ETx_k.data(), fld.ETx.data(), s);
        eng.c2r(ws.ETy_k.data(), fld.ETy.data(), s);
        eng.c2r(ws.ETz_k.data(), fld.ETz.data(), s);
        // caller forms E_total = E_L + E_T (+pump) for the push
    }
};

} // namespace arc

#endif // ARC_PIC_SOLVER_DARWIN_HPP
