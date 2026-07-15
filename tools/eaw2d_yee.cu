// ArcWarden — Ma et al. (PoP 2024) EAW/whistler anisotropy runs on the 2D
// full-Maxwell (Yee) branch: reproduce case 7 of donglai96/EAW_ECH_whistler
// (OSIRIS deck eaw_ech_search_case7 / paper Fig. 1).
//
// Physics: uniform anisotropic electrons (T_perp/T_par = 5) in B0 = wce x̂
// (in-plane). The whistler anisotropy instability grows OBLIQUE whistlers
// (theta ~ 48 deg at saturation t ~ 1100 wpe^-1) whose parallel E component
// Landau-resonates with thermal electrons at v_ph ~ 3.3 v_th_par, producing
// electrostatic time-domain structures in Ex. First genuinely-2D exercise of
// MaxwellSimulation (the An 2019 cross-check ran ny = 1).
//
// Diagnostics (float32 binary unless noted):
//   <p>_linex.bin  rows [6][nx]  Ex,Ey,Ez,Bx,By,Bz along x at j=jline, every t_line
//   <p>_liney.bin  rows [6][ny]  same along y at i=iline
//   <p>_hist.txt   t  WEx WEy WEz WBx WBy WBz  Sux2 Suy2 Suz2   (raw sums)
//   <p>_snap.bin   [6][ny][nx] full fields every t_snap
//   <p>_part.bin   [M][5] (x,y,ux,uy,uz) marker subsample every t_part
//   <p>_meta.txt
//
// Usage: eaw2d_yee <deck.ini> [outdir] [--ppc=N] [--tend=T] [--tsnap=T] [--tline=T]

#include "pic/deck.hpp"
#include "pic/run_meta.hpp"
#include "pic/simulation_maxwell.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace arc;

// per-component field energy sums (no dA factor; post-processing applies it)
static __global__ void k_energy6(YeeViews v, double* out) {
    __shared__ double s[6];
    if (threadIdx.x < 6) s[threadIdx.x] = 0;
    __syncthreads();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < v.nx * v.ny) {
        atomicAdd(&s[0], (double)v.ex[n] * v.ex[n]);
        atomicAdd(&s[1], (double)v.ey[n] * v.ey[n]);
        atomicAdd(&s[2], (double)v.ez[n] * v.ez[n]);
        atomicAdd(&s[3], (double)v.bx[n] * v.bx[n]);
        atomicAdd(&s[4], (double)v.by[n] * v.by[n]);
        atomicAdd(&s[5], (double)v.bz[n] * v.bz[n]);
    }
    __syncthreads();
    if (threadIdx.x < 6) atomicAdd(&out[threadIdx.x], s[threadIdx.x]);
}

// velocity second moments (temperature/anisotropy history)
static __global__ void k_moments(ParticleViews p, double* out) {
    __shared__ double s[3];
    if (threadIdx.x < 3) s[threadIdx.x] = 0;
    __syncthreads();
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < p.n) {
        atomicAdd(&s[0], (double)p.ux[t] * p.ux[t]);
        atomicAdd(&s[1], (double)p.uy[t] * p.uy[t]);
        atomicAdd(&s[2], (double)p.uz[t] * p.uz[t]);
    }
    __syncthreads();
    if (threadIdx.x < 3) atomicAdd(&out[threadIdx.x], s[threadIdx.x]);
}

// strided marker subsample -> [M][5] buffer
static __global__ void k_subsample(ParticleViews p, float* out, long stride, long m) {
    const long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    const long t = k * stride;
    out[k * 5 + 0] = p.x[t];  out[k * 5 + 1] = p.y[t];
    out[k * 5 + 2] = p.ux[t]; out[k * 5 + 3] = p.uy[t]; out[k * 5 + 4] = p.uz[t];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: eaw2d_yee <deck.ini> [outdir] [--ppc=N] [--tend=T] [--tsnap=T] [--tline=T]\n");
        return 1;
    }
    try {
        Deck d = load_deck(argv[1]);
        std::string outdir = ".";
        double tend = d.rp.nsteps * d.rp.dt, tsnap = 50.0, tline = 1.0, tpart = 250.0;
        int ppc_override = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if      (a.rfind("--ppc=", 0) == 0)   ppc_override = std::stoi(a.substr(6));
            else if (a.rfind("--tend=", 0) == 0)  tend  = std::stod(a.substr(7));
            else if (a.rfind("--tsnap=", 0) == 0) tsnap = std::stod(a.substr(8));
            else if (a.rfind("--tline=", 0) == 0) tline = std::stod(a.substr(8));
            else if (a.rfind("--", 0) != 0)       outdir = a;
        }
        ::mkdir(outdir.c_str(), 0755);
        write_run_meta(outdir, argv[1], argc, argv);
        if (ppc_override > 0)
            for (auto& sp : d.species) sp.ppc = ppc_override;

        Grid g(d.nx, d.ny, d.Lx, d.Ly);
        RunParams rp = d.rp;
        const long nsteps      = (long)std::lround(tend / rp.dt);
        const int  line_stride = std::max(1, (int)std::lround(tline / rp.dt));
        const int  snap_stride = std::max(1, (int)std::lround(tsnap / rp.dt));
        const int  part_stride = std::max(1, (int)std::lround(tpart / rp.dt));
        const int  jline = std::min(270, d.ny - 1), iline = std::min(270, d.nx - 1);
        rp.nsteps = nsteps;

        std::string pref = outdir + "/" + (d.prefix.empty() ? "eaw2d" : d.prefix) + "_";
        std::printf("eaw2d_yee — 2D anisotropy-driven oblique whistler (full Maxwell)\n");
        std::printf("  grid %dx%d  dx=%g  c=%g  B0=(%g,%g,%g)  jfilter=%d\n",
                    d.nx, d.ny, g.dx, rp.c, rp.B0[0], rp.B0[1], rp.B0[2], rp.jfilter);
        std::printf("  dt=%g  nsteps=%ld (tend=%g)  line/snap/part strides %d/%d/%d\n",
                    rp.dt, nsteps, tend, line_stride, snap_stride, part_stride);

        MaxwellSimulation sim(g, rp);
        sim.particles().initialize(d.species, g, rp, sim.stream());
        sim.stream().synchronize();
        const long np = (long)sim.particles().n;
        std::printf("  particles: %ld (uth = %g %g %g)\n", np,
                    d.species[0].uth[0], d.species[0].uth[1], d.species[0].uth[2]);
        std::fflush(stdout);

        // marker subsample size / stride
        const long msub    = std::min<long>(np, 1000000);
        const long sstride = std::max<long>(1, np / msub);
        const long msamp   = np / sstride;

        FILE* flx = std::fopen((pref + "linex.bin").c_str(), "wb");
        FILE* fly = std::fopen((pref + "liney.bin").c_str(), "wb");
        FILE* fsn = std::fopen((pref + "snap.bin").c_str(), "wb");
        FILE* fpt = std::fopen((pref + "part.bin").c_str(), "wb");
        FILE* fh  = std::fopen((pref + "hist.txt").c_str(), "w");
        std::fprintf(fh, "# t WEx WEy WEz WBx WBy WBz Sux2 Suy2 Suz2  (raw grid/particle sums)\n");

        DeviceArray<double> dsum(9);
        DeviceArray<float>  dcol(d.ny), dsub(msamp * 5);
        std::vector<float>  hrow(d.nx), hcol(d.ny), hfull((size_t)g.real_size()),
                            hsub((size_t)msamp * 5);
        double hsum[9];
        int nline = 0, nsnap = 0, npart = 0;

        cudaStream_t st = sim.stream();
        auto dump_linex = [&](float* arr) {   // row j=jline is contiguous
            CUDA_CHECK(cudaMemcpyAsync(hrow.data(), arr + (size_t)jline * d.nx,
                                       d.nx * sizeof(float), cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::fwrite(hrow.data(), sizeof(float), d.nx, flx);
        };
        auto dump_liney = [&](float* arr) {   // column i=iline via 2D copy
            CUDA_CHECK(cudaMemcpy2DAsync(hcol.data(), sizeof(float),
                                         arr + iline, (size_t)d.nx * sizeof(float),
                                         sizeof(float), d.ny, cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::fwrite(hcol.data(), sizeof(float), d.ny, fly);
        };
        auto dump_snap = [&](float* arr) {
            CUDA_CHECK(cudaMemcpyAsync(hfull.data(), arr, hfull.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::fwrite(hfull.data(), sizeof(float), hfull.size(), fsn);
        };

        const auto t0 = std::chrono::steady_clock::now();
        for (long n = 0; n <= nsteps; ++n) {
            if (n > 0) sim.step();
            const double t = n * rp.dt;
            YeeFields& yf = sim.fields();
            float* comps[6] = {yf.ex_.data(), yf.ey_.data(), yf.ez_.data(),
                               yf.bx_.data(), yf.by_.data(), yf.bz_.data()};

            if (n % line_stride == 0) {
                for (float* a : comps) dump_linex(a);
                for (float* a : comps) dump_liney(a);
                ++nline;
                CUDA_CHECK(cudaMemsetAsync(dsum.data(), 0, dsum.bytes(), st));
                const int nc = g.real_size();
                k_energy6<<<(nc + 255) / 256, 256, 0, st>>>(sim.fields().views(), dsum.data());
                k_moments<<<(int)((np + 255) / 256), 256, 0, st>>>(sim.particles().views(),
                                                                   dsum.data() + 6);
                CUDA_CHECK(cudaMemcpyAsync(hsum, dsum.data(), sizeof(hsum),
                                           cudaMemcpyDeviceToHost, st));
                CUDA_CHECK(cudaStreamSynchronize(st));
                std::fprintf(fh, "%.4f %.6e %.6e %.6e %.6e %.6e %.6e %.10e %.10e %.10e\n",
                             t, hsum[0], hsum[1], hsum[2], hsum[3], hsum[4], hsum[5],
                             hsum[6], hsum[7], hsum[8]);
            }
            if (n % snap_stride == 0) {
                for (float* a : comps) dump_snap(a);
                ++nsnap;
            }
            if (n % part_stride == 0) {
                k_subsample<<<(int)((msamp + 255) / 256), 256, 0, st>>>(
                    sim.particles().views(), dsub.data(), sstride, msamp);
                CUDA_CHECK(cudaMemcpyAsync(hsub.data(), dsub.data(),
                                           hsub.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost, st));
                CUDA_CHECK(cudaStreamSynchronize(st));
                std::fwrite(hsub.data(), sizeof(float), hsub.size(), fpt);
                ++npart;
            }
            if (n > 0 && nsteps >= 10 && n % (nsteps / 10) == 0) {
                const auto e = sim.field_energy();
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                std::printf("t=%8.1f  WE=%.3e  WB=%.3e  [%.0f s]\n", t, e.we, e.wb, el);
                std::fflush(stdout);
            }
        }
        std::fclose(flx); std::fclose(fly); std::fclose(fsn);
        std::fclose(fpt); std::fclose(fh);

        FILE* fm = std::fopen((pref + "meta.txt").c_str(), "w");
        std::fprintf(fm, "nx %d\nny %d\nLx %g\nLy %g\ndt %.10g\nc %g\nwce %g\n"
                         "jline %d\niline %d\nt_line %.10g\nt_snap %.10g\nt_part %.10g\n"
                         "nline %d\nnsnap %d\nnpart %d\nmsub %ld\nnp %ld\nppc %d\n"
                         "uthx %g\nuthy %g\nuthz %g\njfilter %d\n",
                     d.nx, d.ny, d.Lx, d.Ly, rp.dt, rp.c, rp.wce,
                     jline, iline, line_stride * rp.dt, snap_stride * rp.dt,
                     part_stride * rp.dt, nline, nsnap, npart, msamp, np,
                     d.species[0].ppc, d.species[0].uth[0], d.species[0].uth[1],
                     d.species[0].uth[2], rp.jfilter);
        std::fclose(fm);

        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("done: %.1f s (%.3g particle-steps/s)\n", sec,
                    (double)np * nsteps / sec);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "eaw2d_yee: %s\n", e.what());
        return 1;
    }
}
