// ArcWarden — two-stream phase-space movie dumper + GPU benchmark.
//
// Runs the two-stream instability (same setup as tests/test_two_stream.cu) with a
// large particle count, dumps (x, vx) phase-space frames for a movie, then runs a
// no-dump benchmark loop and reports throughput + GPU utilization (via NVML).
//
//   ./two_stream_movie [outdir] [ppc] [frame_every]
//   defaults: outdir=phase_frames, ppc=4096 (~2.1M particles), frame_every=10
//
// Frames subsample to ~DUMP_CAP points (plotting/disk-friendly) while the
// simulation uses ALL particles; the GPU report reflects the full count.

#include "pic/grid.hpp"
#include "pic/simulation.hpp"

#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace arc;

static constexpr int DUMP_CAP = 100000;   // max points written per frame

// ---- NVML background sampler: polls GPU/memory utilization while running ----
struct GpuMonitor {
    nvmlDevice_t dev{};
    std::thread  th;
    std::atomic<bool>      running{false};
    std::atomic<long>      samples{0};
    std::atomic<long long> sum_gpu{0}, sum_mem{0};
    unsigned peak_gpu = 0;                 // single-writer (the thread); read after join
    bool ok = false;

    void start() {
        if (nvmlInit() != NVML_SUCCESS) return;
        if (nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS) { nvmlShutdown(); return; }
        ok = true;
        running = true;
        th = std::thread([this] {
            while (running.load()) {
                nvmlUtilization_t u;
                if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) {
                    sum_gpu += u.gpu; sum_mem += u.memory; ++samples;
                    if (u.gpu > peak_gpu) peak_gpu = u.gpu;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    void stop() {
        if (!ok) return;
        running = false;
        if (th.joinable()) th.join();
    }
    double mean_gpu() const { long n = samples.load(); return n ? double(sum_gpu.load()) / n : 0.0; }
    double mean_mem() const { long n = samples.load(); return n ? double(sum_mem.load()) / n : 0.0; }
    unsigned long long mem_used_mib() const {
        nvmlMemory_t m{};
        if (ok && nvmlDeviceGetMemoryInfo(dev, &m) == NVML_SUCCESS) return m.used / (1024ull * 1024ull);
        return 0;
    }
    void shutdown() { if (ok) nvmlShutdown(); }
};

// Deterministic per-particle display jitter in [-0.5, 0.5) (cell units), constant
// across frames (hash of index) so the movie doesn't jiggle.
__host__ static inline float disp_jitter(int i) {
    unsigned h = static_cast<unsigned>(i) * 2654435761u + 1013904223u;
    h ^= h >> 15;
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f) - 0.5f;
}

// Copy particles to the host and write (x_physical, vx) for the given indices.
// Notes:
//  - a RANDOM index subset (not a stride) is used: the quiet start lays positions
//    on a base-2 bit-reversal grid, so any fixed stride aliases into stripes.
//  - x gets a DISPLAY-ONLY +/-0.5-cell jitter. The deposited density is uniform,
//    but the deterministic sub-cell quiet-start pattern is identical in every cell
//    and scatter-plots as a faint comb; the jitter smears that sub-cell lattice so
//    the plot shows the true (uniform-x, warm-beam) distribution. It does NOT touch
//    the simulation — only the dumped coordinate.
static void dump_phase(const Particles& P, const Grid& g, const std::string& path,
                       const std::vector<int>& idx) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const int N = static_cast<int>(P.n);
    std::vector<float> x(N), vx(N);
    CUDA_CHECK(cudaMemcpy(x.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost));
    std::ofstream f(path);
    f << "x,vx\n";
    for (int i : idx) {
        float xd = x[i] + disp_jitter(i);
        if (xd < 0.0f) xd += g.nx; else if (xd >= g.nx) xd -= g.nx;  // keep in [0,nx)
        f << xd * g.dx << ',' << vx[i] << '\n';
    }
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const std::string outdir = (argc > 1) ? argv[1] : "phase_frames";
    const int    ppc         = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const int    frame_every = (argc > 3) ? std::atoi(argv[3]) : 10;
    const double vth         = (argc > 4) ? std::atof(argv[4]) : 0.1;  // beam thermal width
    fs::create_directories(outdir);

    // ---- two-stream setup (matches test_two_stream; larger ppc) ----
    const double v0 = 1.0;
    const double kv = std::sqrt(3.0 / 8.0);     // most-unstable k*v0 (ωpe=1)
    const double Lx = 2.0 * M_PI / kv;
    Grid g(64, 8, Lx, 1.0);

    RunParams rp;
    rp.n0 = 1.0; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.vth = vth; rp.vd = v0; rp.two_stream = true;   // warm beams: each a Maxwellian of width vth
    rp.ppc = ppc;
    rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;
    rp.dt = 0.05;
    rp.nsteps = 600;
    rp.dump_every = 1 << 30;     // disable per-step diagnostics
    rp.perturb_amp = 0.005;
    rp.perturb_kx  = 1;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    Simulation<> sim(g, rp);
    sim.init();
    const long long N = static_cast<long long>(sim.particles().n);
    std::printf("ArcWarden two-stream: %lld particles (ppc=%d, %dx%d grid)  "
                "beams v0=%.2f, vth=%.2f  on %s\n",
                N, ppc, g.nx, g.ny, v0, vth, prop.name);

    // ---- fixed random sample of particle indices (same set every frame) ----
    std::vector<int> sample(N);
    std::iota(sample.begin(), sample.end(), 0);
    if (N > DUMP_CAP) {
        std::mt19937 rng(12345);
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(DUMP_CAP);
        std::sort(sample.begin(), sample.end());
    }

    // ---- movie frames ----
    std::ofstream man(outdir + "/manifest.csv");
    man << "frame,step,time,file\n";
    int  frame = 0;
    char name[64];
    auto write_frame = [&](long step) {
        std::snprintf(name, sizeof(name), "frame_%04d.csv", frame);
        dump_phase(sim.particles(), g, outdir + "/" + name, sample);
        man << frame << ',' << step << ',' << step * rp.dt << ',' << name << '\n';
        man.flush();
        ++frame;
    };
    write_frame(0);
    for (long n = 0; n < rp.nsteps; ++n) {
        sim.step(n);
        if ((n + 1) % frame_every == 0) write_frame(n + 1);
    }
    std::printf("wrote %d movie frames to %s/ (subsampled to <= %d pts/frame)\n",
                frame, outdir.c_str(), DUMP_CAP);

    // ---- GPU benchmark: tight step loop, no dumps, NVML-sampled ----
    const long bench_steps = 1000;   // ~1+ s so the NVML mean is stable
    GpuMonitor mon; mon.start();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (long n = 0; n < bench_steps; ++n) sim.step(rp.nsteps + n);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    mon.stop();

    const double secs   = std::chrono::duration<double>(t1 - t0).count();
    const double per_step_ms = 1e3 * secs / bench_steps;
    const double throughput  = double(N) * bench_steps / secs;   // particle-updates/sec

    std::printf("\n==================== GPU report ====================\n");
    std::printf("device              : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("particles           : %lld   (%d steps benchmarked)\n", N, (int)bench_steps);
    std::printf("wall time           : %.3f s   (%.3f ms / step)\n", secs, per_step_ms);
    std::printf("throughput          : %.1f M particle-updates / s\n", throughput / 1e6);
    if (mon.ok && mon.samples.load() > 0) {
        std::printf("GPU utilization     : %.1f%% mean, %u%% peak   (%ld NVML samples)\n",
                    mon.mean_gpu(), mon.peak_gpu, mon.samples.load());
        std::printf("mem-bw utilization  : %.1f%% mean\n", mon.mean_mem());
        std::printf("GPU memory in use   : %llu MiB\n", mon.mem_used_mib());
    } else {
        std::printf("GPU utilization     : (NVML unavailable)\n");
    }
    std::printf("====================================================\n");
    mon.shutdown();
    return 0;
}
