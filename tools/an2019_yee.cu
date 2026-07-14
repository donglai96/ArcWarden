// ArcWarden — An et al. (2019) whistler-pump runs on the FULL-MAXWELL (Yee)
// branch: the M1 cross-validation against the spectral Darwin reproduction.
//
// Reads the SAME decks (decks/an2019_sim*.ini). Differences vs the Darwin
// runner handled here:
//   - CFL: full Maxwell keeps light waves; with ny = 1 the y-curls vanish so
//     the bound is the 1D c·dt < dx. dt = cfl·dx/c (deck dt is Darwin's 0.2).
//   - pump is applied at gather time inside the push (yee2d.hpp), so the
//     evolved Yee E stays purely self-consistent.
// Diagnostics (ny = 1, row dumps every `stride` steps, float32):
//   <prefix>_ex.bin  Ex(x) rows      (longitudinal — Langmuir band)
//   <prefix>_bt.bin  By(x),Bz(x) rows (whistler amplitude)
//   <prefix>_phase.bin  (x, vx) subsample at snapshot times + end
//   <prefix>_meta.txt
//
// Usage: an2019_yee <deck.ini> [outdir] [--ppc=N] [--cfl=X] [--tend=T]

#include "pic/deck.hpp"
#include "pic/diag/manager.hpp"
#include "pic/pump.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation_maxwell.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace arc;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: an2019_yee <deck.ini> [outdir] [--ppc=N] [--cfl=X] [--tend=T]\n");
        return 1;
    }
    try {
        Deck d = load_deck(argv[1]);
        std::string outdir = ".";
        double cfl = 0.4, tend = d.rp.nsteps * d.rp.dt;
        int ppc_override = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if      (a.rfind("--ppc=", 0) == 0)  ppc_override = std::stoi(a.substr(6));
            else if (a.rfind("--cfl=", 0) == 0)  cfl = std::stod(a.substr(6));
            else if (a.rfind("--tend=", 0) == 0) tend = std::stod(a.substr(7));
            else if (a.rfind("--", 0) != 0)      outdir = a;
        }
        ::mkdir(outdir.c_str(), 0755);
        write_run_meta(outdir, argv[1], argc, argv);
        if (ppc_override > 0)
            for (auto& sp : d.species) sp.ppc = ppc_override;

        Grid g(d.nx, d.ny, d.Lx, d.Ly);
        RunParams rp = d.rp;
        const double dt_darwin = rp.dt;
        rp.dt = cfl * g.dx / rp.c;                     // ny=1: 1D light CFL
        const long nsteps = (long)(tend / rp.dt);
        const int  stride = (int)std::lround(2.0 / rp.dt);   // sample every 2/wpe

        std::string pref = outdir + "/" + (d.prefix.empty() ? "an2019_yee" : d.prefix) + "_";

        std::printf("an2019_yee — full-Maxwell branch (M1 cross-check)\n");
        std::printf("  grid %dx%d dx=%g  c=%g  B0=(%g,%g,%g)\n", d.nx, d.ny, g.dx,
                    rp.c, rp.B0[0], rp.B0[1], rp.B0[2]);
        std::printf("  dt=%.4f (Darwin deck: %.2f)  nsteps=%ld (tend=%.0f)\n",
                    rp.dt, dt_darwin, nsteps, tend);

        // cadence rescale: deck strides are in DARWIN steps; keep the same
        // physical sampling intervals on the finer Maxwell step.
        rp.nsteps = nsteps;
        d.rp = rp;
        d.kt_stride = std::max(1, (int)std::lround(d.kt_stride * dt_darwin / rp.dt));

        MaxwellSimulation sim(g, rp);
        sim.particles().initialize(d.species, g, rp, sim.stream());
        sim.stream().synchronize();
        std::printf("  particles: %zu\n", sim.particles().n);
        std::fflush(stdout);

        // mirror Fields: the diag modules consume the spectral Fields type with
        // Darwin conventions (B includes B0; E includes the pump). Copy the Yee
        // arrays in and add B0/pump before each sample.
        Fields mirror(g);
        mirror.allocate_em(g, sim.stream());
        mirror.zero(sim.stream());
        diag::DiagManager mgr(d, rp, pref, outdir);
        const size_t nbytes = (size_t)g.real_size() * sizeof(float);
        auto sync_mirror = [&](double t) {
            YeeFields& yf = sim.fields();
            cudaStream_t st = sim.stream();
            CUDA_CHECK(cudaMemcpyAsync(mirror.Ex.data(), yf.ex_.data(), nbytes, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(mirror.Ey.data(), yf.ey_.data(), nbytes, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(mirror.Ez.data(), yf.ez_.data(), nbytes, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(mirror.Bx.data(), yf.bx_.data(), nbytes, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(mirror.By.data(), yf.by_.data(), nbytes, cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(mirror.Bz.data(), yf.bz_.data(), nbytes, cudaMemcpyDeviceToDevice, st));
            // Yee has no separate E_L solve; with ny=1 all resolved k are along
            // x, so the longitudinal field the kt/dispersion diagnostics want
            // is just Ex (self-consistent part, no pump).
            if (mirror.ELx.size() > 0)
                CUDA_CHECK(cudaMemcpyAsync(mirror.ELx.data(), yf.ex_.data(), nbytes,
                                           cudaMemcpyDeviceToDevice, st));
            add_background_b0(mirror, g, rp, st);
            add_pump_field(mirror, g, rp, t, st);
        };

        const auto t0 = std::chrono::steady_clock::now();
        for (long n = 0; n <= nsteps; ++n) {
            if (n > 0) sim.step();
            const double t = n * rp.dt;
            sync_mirror(t);
            mgr.sample(mirror, sim.particles(), g, rp, n, t);
            if (n > 0 && n % (nsteps / 10) == 0) {
                const auto e = sim.field_energy();
                std::printf("t=%8.1f  WE=%.3e  WB=%.3e\n", t, e.we, e.wb);
                std::fflush(stdout);
            }
        }
        mgr.finalize();
        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("done: %.1f s (%.3g particle-steps/s)\n", sec,
                    (double)sim.particles().n * nsteps / sec);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "an2019_yee: %s\n", e.what());
        return 1;
    }
}
