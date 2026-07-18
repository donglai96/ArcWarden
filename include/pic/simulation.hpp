// ArcWarden — simulation orchestration / main loop (Step 13).
//
// Simulation owns every component and runs the unified PIC cycle (design §10):
//
//   step:  zero rho -> deposit -> solve (rho->E) -> push -> migrate -> diagnose
//
// Ownership: grid, particles, sources, fields, the SpectralEngine (which owns the
// FFT plan + k-space workspace, §7.1), a concrete ElectrostaticSpectralSolver
// (v1; Darwin would swap this for a std::variant, §7.2), Diagnostics, and the
// single pipeline CudaStream. The whole rho->E lives inside solver_.solve — no
// stray FFT calls in the loop.
//
// Leapfrog start: init() loads particles, computes the t=0 field, and rolls the
// velocities back half a step (Pusher::half_step_back) so u is at t=-dt/2 while x
// is at t=0 — otherwise the energy curve has a systematic first-step offset.

#ifndef ARC_PIC_SIMULATION_HPP
#define ARC_PIC_SIMULATION_HPP

#include "pic/config.hpp"
#include "pic/cuda_utils.hpp"
#include "pic/depositor.hpp"
#include "pic/diagnostics.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/pump.hpp"
#include "pic/pusher.hpp"
#include "pic/solver_darwin.hpp"
#include "pic/solver_es.hpp"
#include "pic/sources.hpp"
#include "pic/spectral.hpp"

#include <string>

namespace arc {

template<class Cfg = arc::Cfg>
class Simulation {
public:
    Simulation(const Grid& g, const RunParams& rp, const std::string& csv = "")
        : p_(rp), grid_(g), spectral_(g), diag_(csv) {
        diag_.set_geometry(g.dx, g.dy);
    }

    // Multi-species overload (OSIRIS/UPIC style): init() loads from the species list
    // instead of the single-Maxwellian RunParams path.
    Simulation(const Grid& g, const RunParams& rp, const SpeciesList& species,
               const std::string& csv = "")
        : p_(rp), grid_(g), species_(species), spectral_(g), diag_(csv) {
        diag_.set_geometry(g.dx, g.dy);
    }

    // Load particles, allocate grid containers, compute the t=0 field, and roll
    // velocities back half a step (leapfrog start).
    void init() {
        if (!species_.empty()) particles_.initialize(species_, grid_, p_, stream_);
        else                   particles_.initialize(p_, grid_, stream_);
        sources_.allocate(grid_);
        fields_.allocate(grid_);

        if constexpr (Cfg::field_model == FieldModel::Darwin) {
            sources_.allocate_em(grid_);
            fields_.allocate_em(grid_, stream_);
            spectral_.enable_em();
            // t=0 fields so the first push has E_total,B (retained E_T starts 0).
            darwin_fields(0.0, stream_);
            // (leapfrog half-step rollback for the magnetized push is a small first-
            // step transient; deferred — does not affect Weibel growth rates.)
        } else {
            sources_.zero(stream_);
            Depositor<Cfg>::charge(particles_, sources_, grid_, p_, stream_);
            solver_.solve(sources_, fields_, spectral_, p_, stream_);
            Pusher<Cfg>::half_step_back(particles_, fields_, grid_, p_, stream_);
        }
        stream_.synchronize();
    }

    // One PIC step (design §10).
    void step(long n) {
        if constexpr (Cfg::field_model == FieldModel::Darwin) {
            darwin_fields(n * p_.dt, stream_);
            Pusher<Cfg>::boris_em(particles_, fields_, grid_, p_, stream_);
            particles_.migrate(grid_, stream_);
        } else {
            sources_.zero(stream_);
            Depositor<Cfg>::charge(particles_, sources_, grid_, p_, stream_);
            solver_.solve(sources_, fields_, spectral_, p_, stream_);
            Pusher<Cfg>::boris(particles_, fields_, grid_, p_, stream_);
            particles_.migrate(grid_, stream_);
        }
        diag_.maybe_compute(n, particles_.views(), sources_, fields_, p_, stream_);
    }

    void run() {
        for (long n = 0; n < p_.nsteps; ++n) step(n);
        stream_.synchronize();
    }

    // Darwin field solve for the current particle state: deposit ρ,J,amu → E_L,B →
    // ndc fixed-point sweeps for the transverse E_T (E_total = E_L + E_T). Leaves
    // fields_.Ex/Ey/Ez = E_total and fields_.Bx/By/Bz = B for the push.
    void darwin_fields(double t, cudaStream_t s) {
        sources_.zero_rho_j(s);
        sources_.zero_dcu_amu(s);
        Depositor<Cfg>::charge_current_sorted(particles_, sources_, grid_, p_, s); // sorts + ρ,J
        Depositor<Cfg>::deposit_amu(particles_, sources_, grid_, p_, s);           // amu (once)
        dsolver_.solve_el_b(sources_, fields_, spectral_, p_, s);                  // E_L, B (self)
        add_background_b0(fields_, grid_, p_, s);                                  // total B = B_self + B0
        // dcu gathers E_L + pump (NOT E_T — the ρE_T self-term is resummed in green_et)
        dsolver_.form_el(fields_, s);
        add_pump_field(fields_, grid_, p_, t, s);
        sources_.zero_dcu(s);
        Depositor<Cfg>::deposit_dcu(particles_, sources_, fields_, grid_, p_, s);
        dsolver_.solve_et(sources_, fields_, spectral_, p_, s);                    // → E_T (one shot)
        // total E for the push = E_L + E_T + pump
        dsolver_.form_etotal(fields_, s);
        add_pump_field(fields_, grid_, p_, t, s);
    }

    const Diagnostics& diagnostics() const { return diag_; }
    const Fields&      fields() const { return fields_; }
    const RunParams&   params() const { return p_; }
    const Particles&   particles() const { return particles_; }
    Particles&         particles()       { return particles_; }   // tests/tools: perturb after init
    const Grid&        grid() const { return grid_; }

private:
    RunParams      p_;
    Grid           grid_;
    SpeciesList    species_;
    Particles      particles_;
    Sources        sources_;
    Fields         fields_;
    SpectralEngine spectral_;
    ElectrostaticSpectralSolver solver_;
    DarwinSpectralSolver        dsolver_;
    Diagnostics    diag_;
    CudaStream     stream_;
};

} // namespace arc

#endif // ARC_PIC_SIMULATION_HPP
