// ArcWarden — M3 §6.4: the FullF deposition policy seam.
//
// FullF (total-f markers + perturbation-current deposit) must subtract the
// ANALYTIC equilibrium current from the deposited total:
//     δJ_a = q_a Σ_p W_p v_p S_p  -  J0_a^analytic(x)
// so that the equilibrium (drift-ring) current never pollutes the field
// equations, while the markers carry the full distribution. FullF is an
// independent policy, NOT a delta-f special case (in the two-weight form
// f = (p+w) g, forcing p = 1, w = 1 gives f = 2g — contradiction).
//
// M3 scope: uniform drift-free bi-Maxwellian equilibria carry NO equilibrium
// current, so J0 ≡ 0 and the FullF policy coincides with the plain deposit
// (rp.deltaf = 0). This header pins the seam where the M6 prescribed-drift
// work adds the finite J0 profile (magnetization + curvature/gradient-drift
// ring current), applied as a per-column subtraction from J after the
// deposit and BEFORE the Ampere update.

#ifndef ARC_PIC_J0_ANALYTIC_HPP
#define ARC_PIC_J0_ANALYTIC_HPP

namespace arc {

// M6 will replace this with the drift-ring profile evaluated on Yee J sites.
struct J0Analytic {
    __host__ __device__ static float jx(int, int) { return 0.f; }
    __host__ __device__ static float jy(int, int) { return 0.f; }
    __host__ __device__ static float jz(int, int) { return 0.f; }
};

} // namespace arc

#endif // ARC_PIC_J0_ANALYTIC_HPP
