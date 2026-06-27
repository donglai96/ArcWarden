// ArcWarden — two-stream phase-space movie dumper (visualization tool).
//
// Runs the two-stream instability (same setup as tests/test_two_stream.cu) and
// dumps ALL particles' (x, vx) phase space to one CSV per frame, plus a manifest
// (frame,step,time,file). Feed the output folder to scripts/plot_phase_movie.py
// to render per-frame images, a begin-vs-end comparison, and an mp4.
//
//   ./two_stream_movie [outdir] [frame_every]
//   defaults: outdir=phase_frames, frame_every=10 steps
//
// Not a CTest test — a demo/visualization executable.

#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace arc;

// Copy all particles to the host and write (x_physical, vx). Synchronizes the
// device first so the just-completed step's writes are visible.
static void dump_phase(const Particles& P, const Grid& g, const std::string& path) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const int N = static_cast<int>(P.n);
    std::vector<float> x(N), vx(N);
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost));
    std::ofstream f(path);
    f << "x,vx\n";
    for (int i = 0; i < N; ++i) f << x[i] * g.dx << ',' << vx[i] << '\n';
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const std::string outdir = (argc > 1) ? argv[1] : "phase_frames";
    const int frame_every    = (argc > 2) ? std::atoi(argv[2]) : 10;
    fs::create_directories(outdir);

    // ---- two-stream setup (matches test_two_stream) ----
    const double v0 = 1.0;
    const double kv = std::sqrt(3.0 / 8.0);     // most-unstable k*v0 (ωpe=1)
    const double Lx = 2.0 * M_PI / kv;
    Grid g(64, 4, Lx, 1.0);

    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.vth = 0.0; rp.vd = v0; rp.two_stream = true;
    rp.ppc = 64;
    rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;
    rp.dt = 0.05;
    rp.nsteps = 600;
    rp.dump_every = 1 << 30;      // effectively disable per-step diagnostics (we dump frames ourselves)
    rp.perturb_amp = 0.005;
    rp.perturb_kx  = 1;

    Simulation<> sim(g, rp);
    sim.init();

    std::ofstream man(outdir + "/manifest.csv");
    man << "frame,step,time,file\n";

    int  frame = 0;
    char name[64];
    auto write_frame = [&](long step) {
        std::snprintf(name, sizeof(name), "frame_%04d.csv", frame);
        dump_phase(sim.particles(), g, outdir + "/" + name);
        man << frame << ',' << step << ',' << step * rp.dt << ',' << name << '\n';
        man.flush();
        ++frame;
    };

    write_frame(0);                              // initial state (t=0)
    for (long n = 0; n < rp.nsteps; ++n) {
        sim.step(n);
        if ((n + 1) % frame_every == 0) write_frame(n + 1);
    }

    std::printf("wrote %d frames to %s/  (N=%d particles each, Lx=%.4f, dt=%.3f)\n",
                frame, outdir.c_str(), static_cast<int>(sim.particles().n), g.Lx, rp.dt);
    return 0;
}
