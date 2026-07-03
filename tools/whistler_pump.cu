// ArcWarden — whistler-pump nonlinear trapping (An et al. 2019), DECK-DRIVEN.
//
// 1D(x)+3V spectral Darwin, isotropic Maxwellian electrons, uniform background B0,
// driven by an external oblique-whistler pump (mode M, frequency ω0) ramped off at
// t_off. The whistler *parallel* E Landau-traps electrons at v_r = ω0/k_∥; the ratio
// v_r/v_th (set by the electron temperature / Δx·ω_pe/c) selects the nonlinear regime:
//
//   v_r/v_th = 3.2  (tail)    -> beam-mode LANGMUIR waves          (decks/an2019_sim1.ini)
//   v_r/v_th = 2.1  (mid)     -> electron-acoustic + UNIPOLAR       (decks/an2019_sim2.ini)
//   v_r/v_th = 1.0  (thermal) -> phase-space holes + BIPOLAR fields (decks/an2019_sim3.ini)
//
//   ./whistler_pump <deck.ini> [ppc] [amp] [nsteps]
//     ppc/amp/nsteps > 0 override the deck (for smoke-tests / half-GPU sweeps).
//
// All physics is in the deck ([grid]/[time]/[field]/[background]/[pump]/[species]); this
// tool only adds the whistler-specific diagnostics. Outputs prefixed by [diagnostics]
// prefix (default = deck filename stem):  _kt.bin (δE_L(x,t) for k-t + ω-k),
// _vid_phase.bin/_vid_field.bin (video frames), _fields.csv + _fhist.bin (paper snapshot
// at t_snap: fields ∥/⊥ B0 + f(x, v_∥/v_th)). All velocities in v_th units.

#include "pic/config.hpp"
#include "pic/deck.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>

using namespace arc;

// GPU phase-space histogram: bin (x, v_∥/v_th) over all particles. v_∥ = v·b̂ with
// b̂=(bxn,0,bzn) the B0 direction; result is in thermal-velocity units.
__global__ void phase_hist_kernel(const float* px, const float* pux, const float* puz,
                                  long np, float* hist, int nxb, int nvb,
                                  int nx, float bxn, float bzn, float vth,
                                  float vlo, float vhi) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    float xx = px[i];
    if (xx < 0) xx += nx; else if (xx >= nx) xx -= nx;
    const float vp = (pux[i] * bxn + puz[i] * bzn) / vth;   // v_∥ / v_th
    const int ix = (int)(xx / nx * nxb);
    const int iv = (int)((vp - vlo) / (vhi - vlo) * nvb);
    if (ix >= 0 && ix < nxb && iv >= 0 && iv < nvb)
        atomicAdd(&hist[iv * nxb + ix], 1.0f);
}

// |E_x(mode m)|^2 from row j=0 (1D along x), direct DFT.
static double mode_power(const std::vector<float>& ex, int nx, int m) {
    double re = 0, im = 0;
    const double w = 2.0 * M_PI * m / nx;
    for (int i = 0; i < nx; ++i) { re += ex[i] * std::cos(w * i); im -= ex[i] * std::sin(w * i); }
    return (re * re + im * im) / (double(nx) * nx);
}

// deck filename stem (strip directory + extension) for the default output prefix.
static std::string deck_stem(const std::string& path) {
    std::string p = path;
    const auto slash = p.find_last_of("/\\");
    if (slash != std::string::npos) p = p.substr(slash + 1);
    const auto dot = p.find_last_of('.');
    if (dot != std::string::npos) p = p.substr(0, dot);
    return p;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered so progress survives
    if (argc < 2) { std::fprintf(stderr, "usage: %s <deck.ini> [ppc] [amp] [nsteps]\n", argv[0]); return 2; }

    Deck d = load_deck(argv[1]);
    if (!d.rp.pump) { std::fprintf(stderr, "whistler_pump: deck has no [pump] enable=true\n"); return 2; }
    if (!d.darwin)  std::fprintf(stderr, "whistler_pump: warning — [field] model is not darwin\n");

    const int nx = d.nx, ny = d.ny;
    const double dx = d.Lx / d.nx, Lx = d.Lx;
    const double vth = d.species.empty() ? 1.0 : d.species[0].uth[0];

    // CLI overrides (0/absent = keep deck value)
    const int    ppc = (argc > 2 && std::atoi(argv[2]) > 0) ? std::atoi(argv[2])
                     : (d.species.empty() ? 2000 : d.species[0].ppc);
    if (!d.species.empty()) d.species[0].ppc = ppc;
    const double amp = (argc > 3 && std::atof(argv[3]) > 0) ? std::atof(argv[3]) : d.pump_amp;
    const long   nsteps = (argc > 4 && std::atol(argv[4]) > 0) ? std::atol(argv[4]) : d.rp.nsteps;
    d.rp.nsteps = nsteps;
    // re-derive pump amplitudes for the effective amp (deck applied d.pump_amp)
    { const double s = amp * dx / 1e4;
      d.rp.pump_ex = d.pump_ex0 * s; d.rp.pump_ey = d.pump_ey0 * s; d.rp.pump_ez = d.pump_ez0 * s; }

    RunParams& rp = d.rp;
    Grid g(nx, ny, d.Lx, d.Ly);

    // B0 direction (parallel = ∥ B0) recovered from the field, not a hardcoded angle.
    const double bmag = std::sqrt((double)rp.B0[0]*rp.B0[0] + (double)rp.B0[1]*rp.B0[1]
                                + (double)rp.B0[2]*rp.B0[2]);
    const double cost = bmag > 0 ? rp.B0[0]/bmag : 1.0;   // b̂_x
    const double sint = bmag > 0 ? rp.B0[2]/bmag : 0.0;   // b̂_z
    const double k0 = rp.pump_k0, w0 = rp.pump_w0;
    const double vr = w0 / (k0 * cost);                    // Landau resonant velocity
    const int    M = d.pump_M, blo = d.band_lo, bhi = d.band_hi;
    const double toff = rp.pump_toff, tsnap = d.tsnap;

    std::string base = d.prefix.empty() ? deck_stem(argv[1]) : d.prefix;
    const std::string pref = base + "_";
    auto fname = [&](const char* suf) { return pref + suf; };

    std::printf("Whistler-pump [%s]: nx=%d ppc=%d N=%lld | v_th=%.4f dx=%.4f Lx=%.1f c=%.1f "
                "|B0|=%.4g | M=%d k0=%.5g w0=%.5g v_r=%.3f (v_r/v_th=%.3f) t_off=%.0f t_snap=%.0f "
                "amp=%.1f nsteps=%ld\n",
                base.c_str(), nx, ppc, (long long)nx * ny * ppc, vth, dx, Lx, rp.c, bmag,
                M, k0, w0, vr, vr / vth, toff, tsnap, amp, nsteps);

    Simulation<CfgDarwin> sim_(g, rp, d.species);
    sim_.init();

    std::vector<float> ex(nx), bx(g.real_size()), by(g.real_size()), bz(g.real_size());
    auto sample = [&](long n) {
        cudaDeviceSynchronize();
        const Fields& f = sim_.fields();
        cudaMemcpy(ex.data(), f.Ex.data(), nx * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bx.data(), f.Bx.data(), f.Bx.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(by.data(), f.By.data(), f.By.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(bz.data(), f.Bz.data(), f.Bz.bytes(), cudaMemcpyDeviceToHost);
        const double pM = mode_power(ex, nx, M), p2M = mode_power(ex, nx, 2*M),
                     p3M = mode_power(ex, nx, 3*M);
        double pB = 0; for (int m = blo; m <= bhi; ++m) pB += mode_power(ex, nx, m);
        double db = 0; const int N = g.real_size();
        for (int i = 0; i < N; ++i) {
            const double dbx = bx[i]-rp.B0[0], dby = by[i]-rp.B0[1], dbz = bz[i]-rp.B0[2];
            db += dbx*dbx + dby*dby + dbz*dbz;
        }
        const double dB_over_B0 = std::sqrt(db / N) / bmag;
        std::printf("t=%7.1f  |E_M|2=%.3e |E_2M|2=%.3e |E_3M|2=%.3e  band[%d-%d]=%.3e  dB/B0=%.4f\n",
                    n * rp.dt, pM, p2M, p3M, blo, bhi, pB, dB_over_B0);
    };

    // δE_L(x,t) time series for the k-t (mode vs time) and ω-k dispersion plots.
    // Dump the PURE LONGITUDINAL field (fld.ELx, no pump) every kt_stride steps.
    const int dstride = d.kt_stride > 0 ? d.kt_stride : 5;
    FILE* ktf = std::fopen(fname("kt.bin").c_str(), "wb");
    int nsamp = 0;
    std::vector<float> elx(nx);
    auto dump_el = [&]() {
        cudaMemcpy(elx.data(), sim_.fields().ELx.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(elx.data(), sizeof(float), nx, ktf); ++nsamp;
    };

    // ---- per-frame dump for the time-evolution video (phase space + fields) ----
    const int FNXB = 256, FNVB = 200; const float FVLO = -6.0f, FVHI = 6.0f;  // v_∥/v_th
    const int nframes_target = d.n_frames > 0 ? d.n_frames : 150;
    const int fstride = nsteps / nframes_target > 0 ? (int)(nsteps / nframes_target) : 1;
    DeviceArray<float> dhist((std::size_t)FNXB * FNVB);
    std::vector<float> hh((std::size_t)FNXB * FNVB), frex(nx), frey(nx), frez(nx), frby(nx);
    FILE* vph = std::fopen(fname("vid_phase.bin").c_str(), "wb");
    FILE* vfl = std::fopen(fname("vid_field.bin").c_str(), "wb");
    int nframe = 0;
    auto dump_frame = [&]() {
        const Fields& f = sim_.fields();
        const Particles& Pp = sim_.particles();
        dhist.zero();
        const int thr = 256, blk = (int)((Pp.n + thr - 1) / thr);
        phase_hist_kernel<<<blk, thr>>>(Pp.x.data(), Pp.ux.data(), Pp.uz.data(), (long)Pp.n,
                                        dhist.data(), FNXB, FNVB, nx,
                                        (float)cost, (float)sint, (float)vth, FVLO, FVHI);
        CUDA_CHECK(cudaPeekAtLastError());
        cudaMemcpy(hh.data(), dhist.data(), dhist.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(frex.data(), f.Ex.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frey.data(), f.Ey.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frez.data(), f.Ez.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frby.data(), f.By.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(hh.data(), sizeof(float), hh.size(), vph);
        std::vector<float> row(3*nx);
        for (int i = 0; i < nx; ++i) {
            row[i]      = (float)(frex[i]*cost + frez[i]*sint);  // E_∥
            row[nx+i]   = frey[i];                                // E_⊥ (pure transverse)
            row[2*nx+i] = frby[i] - rp.B0[1];                     // δB_y
        }
        std::fwrite(row.data(), sizeof(float), row.size(), vfl);
        ++nframe;
    };

    // ---- paper snapshot (fields + phase space) dumped when t crosses t_snap ----
    bool snapped = false;
    auto dump_snapshot = [&](double t) {
        cudaDeviceSynchronize();
        const Fields& f = sim_.fields();
        std::vector<float> fex(nx), fey(nx), fez(nx), fbx(nx), fby(nx), fbz(nx);
        cudaMemcpy(fex.data(), f.Ex.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fey.data(), f.Ey.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fez.data(), f.Ez.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fbx.data(), f.Bx.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fby.data(), f.By.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fbz.data(), f.Bz.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        { std::ofstream o(fname("fields.csv")); o << "x,Epar,Eperp,Ey,dBy,dBmag\n";
          for (int i = 0; i < nx; ++i) {
              const double Epar = fex[i]*cost + fez[i]*sint;   // ∥ B0
              const double Eperp = fey[i];                      // ⊥ (pure transverse, clean)
              const double dbx = fbx[i]-rp.B0[0], dby = fby[i]-rp.B0[1], dbz = fbz[i]-rp.B0[2];
              o << i*dx << ',' << Epar << ',' << Eperp << ',' << fey[i] << ','
                << dby << ',' << std::sqrt(dbx*dbx+dby*dby+dbz*dbz) << '\n';
          } }
        // f(x, v_∥/v_th) over ALL particles
        const Particles& P = sim_.particles();
        const int Np = (int)P.n;
        std::vector<float> px(Np), pvx(Np), pvz(Np);
        cudaMemcpy(px.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost);
        cudaMemcpy(pvx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(pvz.data(), P.uz.data(), P.uz.bytes(), cudaMemcpyDeviceToHost);
        const int nxb = 512, nvb = 480; const double vlo = -6.0, vhi = 6.0;  // v_∥/v_th
        std::vector<float> hist((size_t)nxb*nvb, 0.0f);
        for (int i = 0; i < Np; ++i) {
            double xx = px[i]*dx; if (xx < 0) xx += Lx; else if (xx >= Lx) xx -= Lx;
            const double vp = (pvx[i]*cost + pvz[i]*sint) / vth;
            int ix = (int)(xx / Lx * nxb);
            int iv = (int)((vp - vlo) / (vhi - vlo) * nvb);
            if (ix>=0 && ix<nxb && iv>=0 && iv<nvb) hist[(size_t)iv*nxb + ix] += 1.0f;
        }
        std::ofstream hb(fname("fhist.bin"), std::ios::binary);
        hb.write((char*)hist.data(), hist.size()*sizeof(float));
        { std::ofstream m(fname("fhist.meta"));
          m << "nxb "<<nxb<<"\nnvb "<<nvb<<"\nLx "<<Lx<<"\nvlo "<<vlo<<"\nvhi "<<vhi
            <<"\nvr "<<vr/vth<<"\ntsnap "<<t<<"\n"; }
        std::printf("  [snapshot] wrote %sfields.csv + %sfhist.bin at t=%.1f (%dx%d over %d particles)\n",
                    pref.c_str(), pref.c_str(), t, nxb, nvb, Np);
    };

    const int every = nsteps / 40 > 0 ? (int)(nsteps / 40) : 1;
    const long snap_step = (long)std::lround(tsnap / rp.dt);
    sample(0);
    cudaDeviceSynchronize(); dump_el(); dump_frame();
    for (long n = 0; n < nsteps; ++n) {
        sim_.step(n);
        if ((n + 1) % every == 0) sample(n + 1);
        if ((n + 1) % dstride == 0) { cudaDeviceSynchronize(); dump_el(); }
        if ((n + 1) % fstride == 0) { cudaDeviceSynchronize(); dump_frame(); }
        if (!snapped && (n + 1) >= snap_step) { dump_snapshot((n + 1) * rp.dt); snapped = true; }
    }
    if (!snapped) dump_snapshot(nsteps * rp.dt);   // fallback if run < t_snap
    std::fclose(ktf); std::fclose(vph); std::fclose(vfl);

    { std::ofstream o(fname("vid.meta"));
      o << "nframe "<<nframe<<"\nnxb "<<FNXB<<"\nnvb "<<FNVB<<"\nnx "<<nx
        <<"\nLx "<<Lx<<"\nvlo "<<FVLO<<"\nvhi "<<FVHI<<"\ntoff "<<toff
        <<"\ndt_frame "<<fstride*rp.dt<<"\nvr "<<vr/vth<<"\n"; }
    { std::ofstream o(fname("kt.meta"));
      o << "nx "<<nx<<"\nnsamp "<<nsamp<<"\ndt_sample "<<dstride*rp.dt
        <<"\ntoff "<<toff<<"\nLx "<<Lx<<"\nM "<<M<<"\nw0 "<<w0
        <<"\nband_lo "<<blo<<"\nband_hi "<<bhi<<"\nvphx "<<w0/k0<<"\n"; }
    std::printf("wrote %skt.bin (%d samples x %d), video (%d frames)\n", pref.c_str(), nsamp, nx, nframe);
    return 0;
}
