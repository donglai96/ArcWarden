// ArcWarden — PLAN_2D P0-1 gate: χ_r shell split on the (E,μ) mirror loader.
//
// The shell species multiplies marker weights by a raised-cosine window
// χ_r(|v∥eq|); the invert species carries (1−χ_r). Checks, for dist=bimax
// and dist=prodkappa on a 2D slab mirror (b0_prof=3):
//   1. EXACT complement: same seed ⇒ identical draws ⇒ per-marker
//      w_shell + w_invert == w_full to float rounding (χ_r + (1−χ_r) = 1).
//   2. Window support: markers with |v∥eq| ≥ v2 + dv/2 have w_shell = 0;
//      markers strictly inside [v1+dv/2, v2−dv/2] have w_shell == w_full.
//   3. Shell fraction 0 < ⟨χ⟩ < 1 and matches an independent host-side
//      recomputation of χ from the marker velocities (invariant formula).
//
// Exits non-zero on failure.

#include "pic/grid.hpp"
#include "pic/particles.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace arc;

namespace {

struct HostP {
    std::vector<float> x, ux, uy, uz, w;
    std::size_t n;
};

HostP download(Particles& p) {
    HostP h;
    h.n = p.n;
    h.x.resize(p.n); h.ux.resize(p.n); h.uy.resize(p.n);
    h.uz.resize(p.n); h.w.resize(p.n);
    CUDA_CHECK(cudaMemcpy(h.x.data(),  p.x.data(),  p.n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.ux.data(), p.ux.data(), p.n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.uy.data(), p.uy.data(), p.n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.uz.data(), p.uz.data(), p.n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.w.data(),  p.w.data(),  p.n * 4, cudaMemcpyDeviceToHost));
    return h;
}

double chi_host(double v, double v1, double v2, double dv) {
    const double h = 0.5 * dv;
    double chi = 1.0;
    if (v1 > 0.0) {
        if (v <= v1 - h) chi = 0.0;
        else if (v < v1 + h) chi = 0.5 * (1.0 - std::cos(M_PI * (v - v1 + h) / dv));
    }
    if (v >= v2 + h) chi = 0.0;
    else if (v > v2 - h) chi *= 0.5 * (1.0 + std::cos(M_PI * (v - v2 + h) / dv));
    return chi;
}

int run_case(int dist, const char* name) {
    const int nx = 128, ny = 8;
    const double dx = 0.5;
    Grid g(nx, ny, nx * dx, ny * dx);
    RunParams rp;
    rp.qm = -1.0; rp.B0[0] = 0.2f; rp.wce = 0.2;
    rp.b0_prof = 3; rp.b0_a = 2.0e-4;             // slab mirror, ratio ~1.8 at ends
    rp.b0_xc = 0.5 * nx * dx; rp.b0_yc = 0.5 * ny * dx;
    rp.rng_seed = 20260809UL;

    const double V1 = 0.04, V2 = 0.30, DV = 0.02;

    Species base{"pool", 0.01, 200, {0.12, 0.24, 0.24}, {0, 0, 0}, false};
    if (dist == 3) { base.dist = 3; base.kappa_par = 4.0; base.cone_b = 5.0; }

    Species s_full = base;
    Species s_shell = base;
    s_shell.shell_v1 = V1; s_shell.shell_v2 = V2; s_shell.shell_dv = DV;
    Species s_inv = s_shell;
    s_inv.shell_invert = 1;

    CudaStream st;
    Particles pf, ps, pi;
    pf.initialize_mirror(s_full, g, rp, st.get());
    ps.initialize_mirror(s_shell, g, rp, st.get());
    pi.initialize_mirror(s_inv, g, rp, st.get());
    st.synchronize();

    if (pf.n != ps.n || pf.n != pi.n || pf.n == 0) {
        std::printf("FAIL[%s]: marker counts differ (%zu %zu %zu)\n",
                    name, pf.n, ps.n, pi.n);
        return 1;
    }
    HostP F = download(pf), S = download(ps), I = download(pi);

    int bad_sum = 0, bad_supp = 0, bad_chi = 0;
    double wsum_f = 0, wsum_s = 0;
    for (std::size_t k = 0; k < F.n; ++k) {
        wsum_f += F.w[k]; wsum_s += S.w[k];
        // 1. complement
        if (std::fabs((double)S.w[k] + (double)I.w[k] - (double)F.w[k])
            > 1e-6 * (double)F.w[k] + 1e-12) ++bad_sum;
        // recompute invariant from the SHELL marker's own velocities
        const double xph = S.x[k] * dx;
        const double b = (double)bg::b0x(rp, (float)xph) / (double)rp.B0[0];
        const double up2 = (double)S.uy[k] * S.uy[k] + (double)S.uz[k] * S.uz[k];
        const double veq = std::sqrt((double)S.ux[k] * S.ux[k]
                                     + up2 * (1.0 - 1.0 / b));
        // 2. support
        if (veq >= V2 + 0.5 * DV + 1e-4 && S.w[k] != 0.0f) ++bad_supp;
        if (veq > V1 + 0.5 * DV + 1e-4 && veq < V2 - 0.5 * DV - 1e-4
            && S.w[k] != F.w[k]) ++bad_supp;
        // 3. χ recomputation (float b in kernel vs here: same call — tight tol)
        const double chi = chi_host(veq, V1, V2, DV);
        if (std::fabs((double)S.w[k] - chi * (double)F.w[k])
            > 2e-3 * (double)F.w[k] + 1e-10) ++bad_chi;
    }
    const double fshell = wsum_s / wsum_f;
    std::printf("[%s] n=%zu  shell fraction <chi> = %.4f  "
                "bad: sum %d  support %d  chi %d\n",
                name, F.n, fshell, bad_sum, bad_supp, bad_chi);
    if (bad_sum || bad_supp || bad_chi) { std::printf("FAIL[%s]\n", name); return 1; }
    if (!(fshell > 0.02 && fshell < 0.98)) {
        std::printf("FAIL[%s]: degenerate shell fraction\n", name); return 1;
    }
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= run_case(0, "bimax");
    rc |= run_case(3, "prodkappa");
    if (!rc) std::printf("PASS test_shell_split\n");
    return rc;
}
