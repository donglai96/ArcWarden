// ArcWarden — whistler-pump nonlinear trapping (An et al. 2019, Simulation 1).
//
// Reproduces the setup: 1D(x)+3V spectral Darwin, isotropic Maxwellian electrons,
// background B0 at 30° (ω_ce=0.1 ω_pe), driven by an external whistler pump
// (mode M=10, ω0=0.0215 ω_pe) ramped off at t_off=1100. The whistler parallel E
// Landau-traps tail electrons at v_r≈3.2 v_th → spatially-modulated bump-on-tail →
// Langmuir waves (modes ~300-400).
//
//   ./whistler_pump [ppc] [nsteps] [nx] [ny]      defaults 2000 6000 4096 1
//
// Units: ω_pe=1, m_e=1, e=1, v_th=1 → λ_D=1, Δx=2λ_D=2. Pump amps from Table I,
// converted via Ẽ_α0 = 1e4·E_α0/(ω_pe²·Δx) → E_α0 = Ẽ_α0·Δx/1e4.
//
// Diagnostics each sample: whistler mode-10 power, harmonics (20,30), Langmuir-band
// power (modes 300-400), and δB_rms/B0. A final parallel phase-space dump.

#include "pic/config.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace arc;

// GPU phase-space histogram: bin (x, v_parallel) over all particles. v_par = v·b̂
// with b̂=(bxn,0,bzn) the B0 direction. Cheap to call each video frame (only the
// small histogram is copied back, not the 268M particles).
__global__ void phase_hist_kernel(const float* px, const float* pux, const float* puz,
                                  long np, float* hist, int nxb, int nvb,
                                  int nx, float bxn, float bzn, float vlo, float vhi) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    float xx = px[i];
    if (xx < 0) xx += nx; else if (xx >= nx) xx -= nx;
    const float vp = pux[i] * bxn + puz[i] * bzn;
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

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered so progress survives
    const int ppc    = (argc > 1) ? std::atoi(argv[1]) : 2000;
    const int nsteps = (argc > 2) ? std::atoi(argv[2]) : 6000;
    const int nx     = (argc > 3) ? std::atoi(argv[3]) : 4096;
    const int ny     = (argc > 4) ? std::atoi(argv[4]) : 1;
    const double amp = (argc > 5) ? std::atof(argv[5]) : 1.0;   // pump amplitude scale

    const double dx = 2.0;                 // Δx = 2 λ_D, v_th = 1
    Grid g(nx, ny, nx * dx, ny * dx);
    const double Lx = nx * dx;
    const double TwoPi = 6.28318530717958648;

    RunParams rp;
    rp.dt = 0.2; rp.qm = -1.0; rp.eps0 = 1.0; rp.c = dx / 0.0267;   // c from Δx ω_pe/c=0.0267
    rp.ndc = 2; rp.noisy_load = true; rp.rng_seed = 20260628UL; rp.dump_every = (1L<<30);
    // background B0 at 30° in x-z plane, ω_ce = 0.1 ω_pe (|qm|=1 → |B0|=0.1)
    const double wce = 0.1;
    rp.B0[0] = (float)(wce * std::cos(TwoPi/12)); rp.B0[1] = 0.0f; rp.B0[2] = (float)(wce * std::sin(TwoPi/12));
    // pump (Table I, Sim 1) in code units
    const double conv = amp * dx / 1e4;
    rp.pump = true;
    rp.pump_ex = 1.39 * conv; rp.pump_ey = 1.81 * conv; rp.pump_ez = -1.81 * conv;
    rp.pump_k0 = TwoPi * 10.0 / Lx; rp.pump_w0 = 0.0215;
    rp.pump_trmp = 62.83; rp.pump_toff = 1100.0;

    const double kpar = rp.pump_k0 * std::cos(TwoPi/12);
    std::printf("Whistler-pump Sim1: nx=%d ny=%d ppc=%d N=%lld | c=%.1f B0=%.3g "
                "k0=%.4g w0=%.4g v_ph_par=%.2f v_th (t_off=%.0f)\n",
                nx, ny, ppc, (long long)nx*ny*ppc, rp.c, wce,
                rp.pump_k0, rp.pump_w0, rp.pump_w0/kpar, rp.pump_toff);

    SpeciesList sp = { Species{ "electrons", 1.0, ppc, {1.0,1.0,1.0}, {0.0,0.0,0.0} } };
    Simulation<CfgDarwin> sim(g, rp, sp);
    sim.init();

    std::vector<float> ex(nx), bx(g.real_size()), by(g.real_size()), bz(g.real_size());
    auto sample = [&](long n) {
        cudaDeviceSynchronize();
        const Fields& f = sim.fields();
        // row j=0 of Ex
        cudaMemcpy(ex.data(), f.Ex.data(), nx * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bx.data(), f.Bx.data(), f.Bx.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(by.data(), f.By.data(), f.By.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(bz.data(), f.Bz.data(), f.Bz.bytes(), cudaMemcpyDeviceToHost);
        const double p10 = mode_power(ex, nx, 10), p20 = mode_power(ex, nx, 20), p30 = mode_power(ex, nx, 30);
        double pL = 0; for (int m = 300; m <= 400; ++m) pL += mode_power(ex, nx, m);
        double db = 0; const int N = g.real_size();
        for (int i = 0; i < N; ++i) {
            const double dbx = bx[i]-rp.B0[0], dby = by[i]-rp.B0[1], dbz = bz[i]-rp.B0[2];
            db += dbx*dbx + dby*dby + dbz*dbz;
        }
        const double dB_over_B0 = std::sqrt(db / N) / wce;
        std::printf("t=%7.1f  |E10|2=%.3e  |E20|2=%.3e |E30|2=%.3e  Langmuir[300-400]=%.3e  dB/B0=%.4f\n",
                    n * rp.dt, p10, p20, p30, pL, dB_over_B0);
    };

    // δE_L(x,t) time series for the k-t (mode vs time) and ω-k dispersion plots.
    // Dump the PURE LONGITUDINAL field (fld.ELx, no pump) every dstride steps.
    // dt_sample = 5*0.2 = 1.0 → Nyquist ω = π ≈ 3.14 ω_pe (resolves Langmuir ω≈1).
    const int dstride = 5;
    FILE* ktf = std::fopen("whistler_kt.bin", "wb");
    int nsamp = 0;
    std::vector<float> elx(nx);
    auto dump_el = [&]() {
        cudaMemcpy(elx.data(), sim.fields().ELx.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(elx.data(), sizeof(float), nx, ktf); ++nsamp;
    };

    // ---- per-frame dump for the time-evolution video (phase space + fields) ----
    const double bxn0 = std::cos(TwoPi/12), bzn0 = std::sin(TwoPi/12);
    const int FNXB = 256, FNVB = 200; const float FVLO = -6.0f, FVHI = 6.0f;
    const int fstride = nsteps / 150 > 0 ? nsteps / 150 : 1;   // ~150 frames
    DeviceArray<float> dhist((std::size_t)FNXB * FNVB);
    std::vector<float> hh((std::size_t)FNXB * FNVB), frex(nx), frey(nx), frez(nx), frby(nx);
    FILE* vph = std::fopen("whistler_vid_phase.bin", "wb");
    FILE* vfl = std::fopen("whistler_vid_field.bin", "wb");
    int nframe = 0;
    auto dump_frame = [&](long step) {
        const Fields& f = sim.fields();
        const Particles& Pp = sim.particles();
        dhist.zero();
        const int thr = 256, blk = (int)((Pp.n + thr - 1) / thr);
        phase_hist_kernel<<<blk, thr>>>(Pp.x.data(), Pp.ux.data(), Pp.uz.data(), (long)Pp.n,
                                        dhist.data(), FNXB, FNVB, nx,
                                        (float)bxn0, (float)bzn0, FVLO, FVHI);
        CUDA_CHECK(cudaPeekAtLastError());
        cudaMemcpy(hh.data(), dhist.data(), dhist.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(frex.data(), f.Ex.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frey.data(), f.Ey.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frez.data(), f.Ez.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frby.data(), f.By.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(hh.data(), sizeof(float), hh.size(), vph);
        // store E_par (=Ex cos+Ez sin), E_perp(=Ey clean), dBy per frame
        std::vector<float> row(3*nx);
        for (int i = 0; i < nx; ++i) {
            row[i]        = (float)(frex[i]*bxn0 + frez[i]*bzn0);  // E_par
            row[nx+i]     = frey[i];                                // E_perp (transverse)
            row[2*nx+i]   = frby[i] - rp.B0[1];                     // dBy
        }
        std::fwrite(row.data(), sizeof(float), row.size(), vfl);
        ++nframe; (void)step;
    };

    const int every = nsteps / 40 > 0 ? nsteps / 40 : 1;
    sample(0);
    cudaDeviceSynchronize(); dump_el(); dump_frame(0);
    for (long n = 0; n < nsteps; ++n) {
        sim.step(n);
        if ((n + 1) % every == 0) sample(n + 1);
        if ((n + 1) % dstride == 0) { cudaDeviceSynchronize(); dump_el(); }
        if ((n + 1) % fstride == 0) { cudaDeviceSynchronize(); dump_frame(n + 1); }
    }
    std::fclose(ktf); std::fclose(vph); std::fclose(vfl);
    { std::ofstream o("whistler_vid.meta");
      o << "nframe "<<nframe<<"\nnxb "<<FNXB<<"\nnvb "<<FNVB<<"\nnx "<<nx
        <<"\nLx "<<Lx<<"\nvlo "<<FVLO<<"\nvhi "<<FVHI<<"\ntoff "<<rp.pump_toff
        <<"\ndt_frame "<<fstride*rp.dt<<"\nvr "<<rp.pump_w0/(rp.pump_k0*bxn0)<<"\n"; }
    std::printf("wrote whistler_vid_phase.bin + whistler_vid_field.bin (%d frames)\n", nframe);
    { std::ofstream o("whistler_kt.meta");
      o << "nx "<<nx<<"\nnsamp "<<nsamp<<"\ndt_sample "<<dstride*rp.dt
        <<"\ntoff "<<rp.pump_toff<<"\nLx "<<Lx<<"\n"; }
    std::printf("wrote whistler_kt.bin (%d samples x %d, δE_L)\n", nsamp, nx);

    // ---- snapshot for the paper-style figure (fields + phase space) ----
    cudaDeviceSynchronize();
    const double bxn = std::cos(TwoPi/12), bzn = std::sin(TwoPi/12);   // B0 dir (x-z)
    const double sxn = std::sin(TwoPi/12), szn = std::cos(TwoPi/12);   // in-plane ⟂

    // (1) field profiles along x (row 0): E_∥, E_⊥(in-plane), E_y, δB_y, |δB|
    const Fields& f = sim.fields();
    std::vector<float> fex(nx), fey(nx), fez(nx), fbx(nx), fby(nx), fbz(nx);
    cudaMemcpy(fex.data(), f.Ex.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fey.data(), f.Ey.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fez.data(), f.Ez.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fbx.data(), f.Bx.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fby.data(), f.By.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fbz.data(), f.Bz.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
    { std::ofstream o("whistler_fields.csv"); o << "x,Epar,Eperp,Ey,dBy,dBmag\n";
      for (int i = 0; i < nx; ++i) {
          const double Epar = fex[i]*bxn + fez[i]*bzn, Eperp = fex[i]*sxn - fez[i]*szn;
          const double dbx = fbx[i]-rp.B0[0], dby = fby[i]-rp.B0[1], dbz = fbz[i]-rp.B0[2];
          o << i*g.dx << ',' << Epar << ',' << Eperp << ',' << fey[i] << ','
            << dby << ',' << std::sqrt(dbx*dbx+dby*dby+dbz*dbz) << '\n';
      } }

    // (2) f(x, v_∥) 2D histogram over ALL particles (smooth phase-space density)
    const Particles& P = sim.particles();
    const int Np = (int)P.n;
    std::vector<float> px(Np), pvx(Np), pvz(Np);
    cudaMemcpy(px.data(),  P.x.data(),  P.x.bytes(),  cudaMemcpyDeviceToHost);
    cudaMemcpy(pvx.data(), P.ux.data(), P.ux.bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(pvz.data(), P.uz.data(), P.uz.bytes(), cudaMemcpyDeviceToHost);
    const int nxb = 512, nvb = 480; const double vlo = -6.0, vhi = 6.0;
    std::vector<float> hist((size_t)nxb*nvb, 0.0f);
    for (int i = 0; i < Np; ++i) {
        double xx = px[i]*g.dx; if (xx < 0) xx += nx*g.dx; else if (xx >= nx*g.dx) xx -= nx*g.dx;
        const double vp = pvx[i]*bxn + pvz[i]*bzn;
        int ix = (int)(xx / (nx*g.dx) * nxb);
        int iv = (int)((vp - vlo) / (vhi - vlo) * nvb);
        if (ix>=0 && ix<nxb && iv>=0 && iv<nvb) hist[(size_t)iv*nxb + ix] += 1.0f;
    }
    std::ofstream hb("whistler_fhist.bin", std::ios::binary);
    hb.write((char*)hist.data(), hist.size()*sizeof(float));
    { std::ofstream m("whistler_fhist.meta");
      m << "nxb "<<nxb<<"\nnvb "<<nvb<<"\nLx "<<nx*g.dx<<"\nvlo "<<vlo<<"\nvhi "<<vhi
        <<"\nvr "<<rp.pump_w0/(rp.pump_k0*bxn)<<"\n"; }
    std::printf("wrote whistler_fields.csv + whistler_fhist.bin (%dx%d over %d particles)\n", nxb, nvb, Np);
    return 0;
}
