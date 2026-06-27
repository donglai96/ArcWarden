// ArcWarden — diagnostics (Step 12).
//
// Diagnostics computes the conserved/derived scalars each dump step and appends a
// CSV row: kinetic energy, field energy, total energy, total charge, max|E|.
// Decoupled from Simulation (design §9): it takes ParticleViews + Sources + Fields
// + RunParams, NOT a Simulation&, so the two never depend on each other.
//
// Reductions are accumulated in DOUBLE (plan §17) — energy conservation is judged
// at the level of small drifts, so float accumulation would mask the signal. v1
// copies the (modest) arrays to the host and reduces there: simple and provably
// correct; a fused GPU reduction is a later optimization (diagnostics are
// infrequent, gated by dump_every).
//
// UNITS (config.hpp contract, ωpe=1): a macro-particle represents `weight`
// physical particles of mass me=1, so kinetic energy = weight·Σ ½(ux²+uy²+uz²).
// Field energy = ½·eps0·Σ(Ex²+Ey²)·dx·dy. Total charge = Σ rho·dx·dy.

#ifndef ARC_PIC_DIAGNOSTICS_HPP
#define ARC_PIC_DIAGNOSTICS_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/fields.hpp"
#include "pic/particles.hpp"
#include "pic/sources.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace arc {

struct DiagSample {
    long   step   = 0;
    double time   = 0.0;
    double ke     = 0.0;   // kinetic energy
    double ee     = 0.0;   // electric field energy
    double total  = 0.0;   // ke + ee
    double charge = 0.0;   // total charge
    double max_e  = 0.0;   // max |E|
};

class Diagnostics {
public:
    Diagnostics() = default;
    // Empty path => no CSV file (e.g. tests); history is still kept in memory.
    explicit Diagnostics(const std::string& csv_path) {
        if (!csv_path.empty()) {
            csv_.open(csv_path);
            csv_ << "step,time,ke,ee,total,charge,max_e\n";
        }
    }

    // Always compute + record (used by tests and by maybe_compute when due).
    DiagSample compute(long step, ParticleViews p, const Sources& src,
                       const Fields& f, const RunParams& rp, cudaStream_t s) {
        CUDA_CHECK(cudaStreamSynchronize(s));   // results must be ready before copy

        // ---- kinetic energy (particles) ----
        const int n = p.n;
        std::vector<float> h(n);
        double sum_u2 = 0.0;
        auto add_sq = [&](const float* dptr) {
            CUDA_CHECK(cudaMemcpy(h.data(), dptr, std::size_t(n) * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double acc = 0.0;
            for (int i = 0; i < n; ++i) acc += double(h[i]) * h[i];
            sum_u2 += acc;
        };
        add_sq(p.ux.ptr); add_sq(p.uy.ptr); add_sq(p.uz.ptr);
        const double ke = rp.weight * 0.5 * sum_u2;

        // ---- field energy + max|E| (grid) ----
        const int ncell = src.rho.size() ? int(src.rho.size()) : 0;
        std::vector<float> ex(ncell), ey(ncell), rho(ncell);
        CUDA_CHECK(cudaMemcpy(ex.data(),  f.Ex.data(),  std::size_t(ncell) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ey.data(),  f.Ey.data(),  std::size_t(ncell) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rho.data(), src.rho.data(), std::size_t(ncell) * sizeof(float), cudaMemcpyDeviceToHost));

        double sum_e2 = 0.0, sum_rho = 0.0, max_e2 = 0.0;
        for (int c = 0; c < ncell; ++c) {
            const double e2 = double(ex[c]) * ex[c] + double(ey[c]) * ey[c];
            sum_e2 += e2;
            if (e2 > max_e2) max_e2 = e2;
            sum_rho += rho[c];
        }

        DiagSample smp;
        smp.step   = step;
        smp.time   = step * rp.dt;
        smp.ke     = ke;
        smp.ee     = 0.5 * rp.eps0 * sum_e2 * g_dx_ * g_dy_;
        smp.charge = sum_rho * g_dx_ * g_dy_;
        smp.max_e  = std::sqrt(max_e2);
        smp.total  = smp.ke + smp.ee;

        hist_.push_back(smp);
        return smp;
    }

    // Compute only on dump steps (dump_every<=0 => every step), and write CSV.
    void maybe_compute(long step, ParticleViews p, const Sources& src,
                       const Fields& f, const RunParams& rp, cudaStream_t s) {
        if (rp.dump_every > 0 && (step % rp.dump_every) != 0) return;
        const DiagSample smp = compute(step, p, src, f, rp, s);
        if (csv_.is_open()) {
            csv_ << smp.step << ',' << smp.time << ',' << smp.ke << ',' << smp.ee
                 << ',' << smp.total << ',' << smp.charge << ',' << smp.max_e << '\n';
            csv_.flush();
        }
    }

    // Diagnostics needs the cell area (dx,dy); set once from the Grid.
    void set_geometry(double dx, double dy) { g_dx_ = dx; g_dy_ = dy; }

    const std::vector<DiagSample>& history() const { return hist_; }
    const DiagSample& last() const { return hist_.back(); }

private:
    std::ofstream           csv_;
    std::vector<DiagSample> hist_;
    double g_dx_ = 1.0, g_dy_ = 1.0;
};

} // namespace arc

#endif // ARC_PIC_DIAGNOSTICS_HPP
