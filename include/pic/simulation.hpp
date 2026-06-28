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
#include "pic/pusher.hpp"
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

        sources_.zero(stream_);
        Depositor<Cfg>::charge(particles_, sources_, grid_, p_, stream_);
        solver_.solve(sources_, fields_, spectral_, p_, stream_);
        Pusher<Cfg>::half_step_back(particles_, fields_, grid_, p_, stream_);
        stream_.synchronize();
    }

    // One PIC step (design §10).
    void step(long n) {
        sources_.zero(stream_);
        Depositor<Cfg>::charge(particles_, sources_, grid_, p_, stream_);
        solver_.solve(sources_, fields_, spectral_, p_, stream_);
        Pusher<Cfg>::boris(particles_, fields_, grid_, p_, stream_);
        particles_.migrate(grid_, stream_);
        diag_.maybe_compute(n, particles_.views(), sources_, fields_, p_, stream_);
    }

    void run() {
        for (long n = 0; n < p_.nsteps; ++n) step(n);
        stream_.synchronize();
    }

    const Diagnostics& diagnostics() const { return diag_; }
    const RunParams&   params() const { return p_; }
    const Particles&   particles() const { return particles_; }
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
    Diagnostics    diag_;
    CudaStream     stream_;
};

} // namespace arc

#endif // ARC_PIC_SIMULATION_HPP
