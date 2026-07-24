// G2.1 cross-check — the Li/Bortnik/An 2019 two-band setup on OUR SPECTRAL
// DARWIN branch (same UPIC lineage as their actual code). Twin of
// tools/liband_yee.cu with identical outputs/meta so the same plot scripts
// run on both arms; the Yee-vs-Darwin pair is the grid-heating control AND
// a methods-paper section (momentum-conserving Yee at dx/lambda_D ~ 5 vs
// smoothed spectral Darwin).
//
// Usage: ./liband_darwin <deck.ini> [outdir] [--nsteps=N]
// Outputs: bline_XXXXXX.bin (dBy,dBz,Epar float32), probe.bin, fhist_*.bin,
//          energy.csv, meta.txt — formats identical to liband_yee.
#include "pic/config.hpp"
#include "pic/deck.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace arc;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <deck.ini> [outdir] [--nsteps=N]\n", argv[0]); return 1; }
    Deck d = load_deck(argv[1]);
    std::string outdir = (argc > 2 && argv[2][0] != '-') ? argv[2] : "liband_darwin_out";
    for (int i = 2; i < argc; ++i)
        if (!std::strncmp(argv[i], "--nsteps=", 9)) d.rp.nsteps = atol(argv[i] + 9);

    RunParams rp = d.rp;
    Grid g(d.nx, d.ny, d.Lx, d.Ly);
    if (!d.darwin || d.species.empty()) {
        std::fprintf(stderr, "liband_darwin: needs [field] model=darwin and >=1 species\n");
        return 1;
    }
    const double th = std::atan2((double)rp.B0[2], (double)rp.B0[0]);
    const float  cx = (float)std::cos(th), cz = (float)std::sin(th);

    std::filesystem::create_directories(outdir);
    write_run_meta(outdir, argv[1], argc, argv);

    Simulation<CfgDarwin> sim(g, rp, d.species);
    sim.init();

    const int nx = g.nx;
    // cadences in PHYSICAL time (match the Yee arm), scaled by dt so a
    // dt-scan doesn't quadruple the I/O load
    const int bline_every = std::max(1, (int)std::lround(4.0 / rp.dt));
    const int probe_every = std::max(1, (int)std::lround(0.2 / rp.dt));
    const int fhist_every = std::max(1, (int)std::lround(40.0 / rp.dt));
    constexpr int NB = 240;
    constexpr double VMAX = 0.6;

    std::vector<double> poff = d.probes;
    if (poff.empty()) poff = {0.25 * d.Lx, 0.5 * d.Lx, 0.75 * d.Lx};
    const int nprobe = (int)poff.size();
    std::vector<int> probe_ix(nprobe);
    for (int p = 0; p < nprobe; ++p)
        probe_ix[p] = ((int)(poff[p] / g.dx) % nx + nx) % nx;

    std::FILE* fpb = std::fopen((outdir + "/probe.bin").c_str(), "wb");
    std::FILE* fen = std::fopen((outdir + "/energy.csv").c_str(), "w");
    std::fprintf(fen, "step,time,dB2\n");
    {   std::FILE* fm = std::fopen((outdir + "/meta.txt").c_str(), "w");
        std::fprintf(fm, "nx %d\ndx %.9g\ndt %.9g\nnsteps %ld\nbline_every %d\n"
                         "probe_every %d\nfhist_every %d\nnb %d\nvmax %.9g\n"
                         "nprobe %d\nnsp %d\nwce %.9g\ntheta_deg %.9g\nnmarkers %lld\n",
                     nx, g.dx, rp.dt, rp.nsteps, bline_every, probe_every,
                     fhist_every, NB, VMAX, nprobe, (int)d.species.size(),
                     std::hypot(rp.B0[0], rp.B0[2]), th * 180.0 / M_PI,
                     (long long)sim.particles().n);
        for (int p = 0; p < nprobe; ++p) std::fprintf(fm, "probe_ix %d\n", probe_ix[p]);
        std::fclose(fm); }

    std::vector<float> by(g.real_size()), bz(g.real_size()),
                       ex(g.real_size()), ez(g.real_size());
    std::vector<float> hux, huz, hw;
    const long nsteps = rp.nsteps;
    for (long n = 1; n <= nsteps; ++n) {
        sim.step(n - 1);
        const bool want_line  = n % bline_every == 0;
        const bool want_probe = n % probe_every == 0;
        const bool want_hist  = n % fhist_every == 0;
        if (want_line || want_probe) {
            cudaDeviceSynchronize();
            const Fields& f = sim.fields();
            CUDA_CHECK(cudaMemcpy(by.data(), f.By.data(), by.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bz.data(), f.Bz.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < nx; ++i) { by[i] -= rp.B0[1]; bz[i] -= rp.B0[2]; }
        }
        if (want_probe) {
            std::vector<float> pb(2 * nprobe);
            for (int p = 0; p < nprobe; ++p) { pb[2*p] = by[probe_ix[p]]; pb[2*p+1] = bz[probe_ix[p]]; }
            std::fwrite(pb.data(), 4, 2 * nprobe, fpb);
        }
        if (want_line) {
            const Fields& f = sim.fields();
            CUDA_CHECK(cudaMemcpy(ex.data(), f.Ex.data(), ex.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ez.data(), f.Ez.data(), ez.size() * 4, cudaMemcpyDeviceToHost));
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/bline_%06ld.bin", outdir.c_str(), n / bline_every);
            std::FILE* fo = std::fopen(fn, "wb");
            std::fwrite(by.data(), 4, nx, fo);
            std::fwrite(bz.data(), 4, nx, fo);
            std::vector<float> ep(nx);
            for (int i = 0; i < nx; ++i) ep[i] = cx * ex[i] + cz * ez[i];
            std::fwrite(ep.data(), 4, nx, fo);
            std::fclose(fo);
        }
        if (want_hist) {
            cudaDeviceSynchronize();
            const auto& P = sim.particles();
            const size_t N = P.n;
            hux.resize(N); huz.resize(N); hw.resize(N);
            CUDA_CHECK(cudaMemcpy(hux.data(), P.ux.data(), N * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(huz.data(), P.uz.data(), N * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hw.data(),  P.w.data(),  N * 4, cudaMemcpyDeviceToHost));
            std::vector<double> hist(NB, 0.0);
            for (size_t t = 0; t < N; ++t) {
                const double vp = cx * (double)hux[t] + cz * (double)huz[t];
                const int b = (int)((vp + VMAX) / (2 * VMAX) * NB);
                if (b >= 0 && b < NB) hist[b] += hw[t];
            }
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s/fhist_%06ld.bin", outdir.c_str(), n / fhist_every);
            std::FILE* fo = std::fopen(fn, "wb");
            std::fwrite(hist.data(), 8, NB, fo);
            std::fclose(fo);
        }
        if (n % 200 == 0) {
            double db2 = 0.0;
            for (int i = 0; i < nx; ++i) db2 += by[i]*by[i] + bz[i]*bz[i];
            std::fprintf(fen, "%ld,%.6g,%.9e\n", n, n * rp.dt, db2 / nx);
            std::fflush(fen);
            if (n % 20000 == 0)
                std::printf("t=%8.0f  <dB2>=%.3e\n", n * rp.dt, db2 / nx);
        }
    }
    std::fclose(fpb); std::fclose(fen);
    std::printf("done: %s\n", outdir.c_str());
    return 0;
}
