// ArcWarden — whistler-pump nonlinear trapping (An et al. 2019, Sims 1/2/3).
//
// 1D(x)+3V spectral Darwin, isotropic Maxwellian electrons, background B0 at 30°
// (ω_ce = 0.1 ω_pe), driven by an external oblique-whistler pump (mode M, cold-
// whistler frequency ω0) ramped off at t_off. The whistler *parallel* E Landau-
// traps electrons at v_r = ω0/k_∥. The controlling parameter (An et al., ref [37])
// is v_r/v_th: fixing v_r ≈ 0.04c and lowering v_th (raising the electron temp.)
// sweeps three regimes of nonlinear electrostatic structure:
//
//   Sim 1  v_r/v_th = 3.2  (tail trapping)   -> beam-mode LANGMUIR waves
//   Sim 2  v_r/v_th = 2.1  (mid-distribution) -> electron-acoustic + UNIPOLAR (double layer)
//   Sim 3  v_r/v_th = 1.0  (thermal trapping) -> phase-space holes + BIPOLAR fields
//
//   ./whistler_pump [sim] [ppc] [nsteps] [amp]        defaults: 1 2000 6000 auto
//
// Units: ω_pe=1, m_e=1, e=1, c=74.9 (fixed), λ_D = v_th, Δx = 2λ_D. Only v_th (and
// hence Δx, the box, and via the cold-whistler dispersion ω0) changes between sims.
// Pump amps from Table I (polarization 1.39 : 1.81 : −1.81), overall scale = amp.
//
// Outputs (prefixed whistler_s{N}_):  _kt.bin (δE_L(x,t) for k-t + ω-k),
// _vid_phase.bin/_vid_field.bin (video frames), _fields.csv + _fhist.bin (paper
// snapshot at t_snap: fields ∥/⊥ B0 + f(x,v_∥/v_th)).  All velocities in v_th units.

#include "pic/config.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/simulation.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Per-simulation configuration (An et al. 2019, Table I + Figs 2/3/5).
//
// Paper normalization: v_th = 1 (λ_D = 1), Δx = 2 λ_D. Only Δx·ω_pe/c (→ the speed of
// light c = Δx/dxwc) and the box change between sims. (nx, M, Δx·ω_pe/c, ω0) are the
// paper's own mutually-consistent values — ω0 already sits on the whistler dispersion,
// so NO dispersion solve is needed here. v_r/v_th = ω0/(k0 cosθ) comes out of them.
struct SimCfg {
    int    nx;        // grid points  (Table I, Nx)
    int    M;         // whistler mode number (wavelengths in the box)
    double dxwc;      // Δx·ω_pe/c  → c = Δx/dxwc  (per-sim; sets v_r/c ≈ 0.04)
    double w0;        // pump frequency ω0 [ω_pe]  (Table I, on the dispersion)
    double ex0;       // pump amplitude Ẽx0 (Table I); Ẽy0=1.81i, Ẽz0=−1.81 for all sims
    double toff;      // pump turn-off time  [ω_pe^-1]  (Table I)
    double tsnap;     // snapshot time for the paper field+phase figure
    int    nsteps;    // default step count  (tend = 4000 → 20000 steps at dt=0.2)
    double amp;       // pump amplitude scale on top of Table I (tuned to δB/B0 ~ 0.1)
    int    band_lo;   // diagnostic mode band (nonlinear structure) lo..hi
    int    band_hi;
    const char* tag;  // one-line description
};

static SimCfg make_cfg(int sim) {
    // Values are An et al. (2019) Table I verbatim: {Nx, M, Δx·ω_pe/c, ω0, Ẽx0, t_off}.
    switch (sim) {
        case 2:  return { 2048, 7, 0.04, 0.0194, 1.35, 750.0, 2500.0, 20000, 6.0, 20, 250,
                          "v_r/v_th=2.1  electron-acoustic + UNIPOLAR (double layer)" };
        case 3:  return { 1024, 7, 0.08, 0.0194, 1.35, 500.0, 1200.0, 20000, 5.0, 20, 100,
                          "v_r/v_th=1.0  phase-space holes + BIPOLAR fields" };
        default: return { 4096, 10, 0.0267, 0.0215, 1.39, 1100.0, 1540.0, 20000, 6.0, 300, 400,
                          "v_r/v_th=3.2  tail trapping -> LANGMUIR waves" };
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered so progress survives
    const int sim    = (argc > 1) ? std::atoi(argv[1]) : 1;
    SimCfg   S       = make_cfg(sim);
    const int ppc    = (argc > 2) ? std::atoi(argv[2]) : 2000;
    const int nsteps = (argc > 3 && std::atoi(argv[3]) > 0) ? std::atoi(argv[3]) : S.nsteps;
    const double amp = (argc > 4 && std::atof(argv[4]) > 0) ? std::atof(argv[4]) : S.amp;

    const int nx = S.nx, ny = 1;
    const double TwoPi = 6.28318530717958648;
    const double theta = TwoPi / 12.0;                  // 30°
    const double cost = std::cos(theta), sint = std::sin(theta);
    const double wce = 0.1;                              // ω_ce = 0.1 ω_pe

    // Paper normalization (An et al., ref [37]): v_th = 1, Δx = 2 λ_D = 2. The speed of
    // light is set PER SIM by Δx·ω_pe/c (c decreases with v_r so v_r/c stays ≈ 0.04).
    const double vth = 1.0;
    const double dx  = 2.0;                              // Δx = 2 λ_D
    const double c0  = dx / S.dxwc;                      // c from Δx·ω_pe/c
    Grid g(nx, ny, nx * dx, ny * dx);
    const double Lx = nx * dx;

    RunParams rp;
    rp.dt = 0.2; rp.qm = -1.0; rp.eps0 = 1.0; rp.c = c0;
    rp.ndc = 2; rp.noisy_load = true; rp.rng_seed = 20260628UL; rp.dump_every = (1L << 30);
    // background B0 at 30° in x-z plane, |B0| = ω_ce (|qm|=1)
    rp.B0[0] = (float)(wce * cost); rp.B0[1] = 0.0f; rp.B0[2] = (float)(wce * sint);
    // whistler pump: k0 from geometry, ω0 taken directly from the paper (on the dispersion)
    const double k0   = TwoPi * S.M / Lx;
    const double w0   = S.w0;
    const double conv = amp * dx / 1e4;                  // Ẽ_α0 = 1e4·E_α0/(ω_pe²·Δx)
    rp.pump = true;
    rp.pump_ex = S.ex0 * conv; rp.pump_ey = 1.81 * conv; rp.pump_ez = -1.81 * conv;
    rp.pump_k0 = k0; rp.pump_w0 = w0;
    rp.pump_trmp = 62.83; rp.pump_toff = S.toff;

    const double kpar = k0 * cost, vr = w0 / kpar;       // Landau resonant velocity
    std::printf("Whistler-pump Sim%d: %s\n", sim, S.tag);
    std::printf("  nx=%d ppc=%d N=%lld | v_th=%.4f dx=%.4f Lx=%.1f c=%.1f | M=%d k0=%.5g "
                "w0=%.5g v_r=%.3f (v_r/v_th=%.3f) t_off=%.0f t_snap=%.0f amp=%.1f nsteps=%d\n",
                nx, ppc, (long long)nx * ny * ppc, vth, dx, Lx, c0, S.M, k0, w0, vr,
                vr / vth, S.toff, S.tsnap, amp, nsteps);

    SpeciesList sp = { Species{ "electrons", 1.0, ppc,
                                {(float)vth, (float)vth, (float)vth}, {0.0, 0.0, 0.0} } };
    Simulation<CfgDarwin> sim_(g, rp, sp);
    sim_.init();

    char pref[32]; std::snprintf(pref, sizeof pref, "whistler_s%d_", sim);
    auto fname = [&](const char* suf) {
        static char buf[64]; std::snprintf(buf, sizeof buf, "%s%s", pref, suf); return buf;
    };

    std::vector<float> ex(nx), bx(g.real_size()), by(g.real_size()), bz(g.real_size());
    auto sample = [&](long n) {
        cudaDeviceSynchronize();
        const Fields& f = sim_.fields();
        cudaMemcpy(ex.data(), f.Ex.data(), nx * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bx.data(), f.Bx.data(), f.Bx.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(by.data(), f.By.data(), f.By.bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(bz.data(), f.Bz.data(), f.Bz.bytes(), cudaMemcpyDeviceToHost);
        const double pM = mode_power(ex, nx, S.M), p2M = mode_power(ex, nx, 2*S.M),
                     p3M = mode_power(ex, nx, 3*S.M);
        double pB = 0; for (int m = S.band_lo; m <= S.band_hi; ++m) pB += mode_power(ex, nx, m);
        double db = 0; const int N = g.real_size();
        for (int i = 0; i < N; ++i) {
            const double dbx = bx[i]-rp.B0[0], dby = by[i]-rp.B0[1], dbz = bz[i]-rp.B0[2];
            db += dbx*dbx + dby*dby + dbz*dbz;
        }
        const double dB_over_B0 = std::sqrt(db / N) / wce;
        std::printf("t=%7.1f  |E_M|2=%.3e |E_2M|2=%.3e |E_3M|2=%.3e  band[%d-%d]=%.3e  dB/B0=%.4f\n",
                    n * rp.dt, pM, p2M, p3M, S.band_lo, S.band_hi, pB, dB_over_B0);
    };

    // δE_L(x,t) time series for the k-t (mode vs time) and ω-k dispersion plots.
    // Dump the PURE LONGITUDINAL field (fld.ELx, no pump) every dstride steps.
    const int dstride = 5;
    FILE* ktf = std::fopen(fname("kt.bin"), "wb");
    int nsamp = 0;
    std::vector<float> elx(nx);
    auto dump_el = [&]() {
        cudaMemcpy(elx.data(), sim_.fields().ELx.data(), nx*sizeof(float), cudaMemcpyDeviceToHost);
        std::fwrite(elx.data(), sizeof(float), nx, ktf); ++nsamp;
    };

    // ---- per-frame dump for the time-evolution video (phase space + fields) ----
    const int FNXB = 256, FNVB = 200; const float FVLO = -6.0f, FVHI = 6.0f;  // v_∥/v_th
    const int fstride = nsteps / 150 > 0 ? nsteps / 150 : 1;   // ~150 frames
    DeviceArray<float> dhist((std::size_t)FNXB * FNVB);
    std::vector<float> hh((std::size_t)FNXB * FNVB), frex(nx), frey(nx), frez(nx), frby(nx);
    FILE* vph = std::fopen(fname("vid_phase.bin"), "wb");
    FILE* vfl = std::fopen(fname("vid_field.bin"), "wb");
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
                    pref, pref, t, nxb, nvb, Np);
    };

    const int every = nsteps / 40 > 0 ? nsteps / 40 : 1;
    const long snap_step = (long)std::lround(S.tsnap / rp.dt);
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
        <<"\nLx "<<Lx<<"\nvlo "<<FVLO<<"\nvhi "<<FVHI<<"\ntoff "<<S.toff
        <<"\ndt_frame "<<fstride*rp.dt<<"\nvr "<<vr/vth<<"\n"; }
    { std::ofstream o(fname("kt.meta"));
      o << "nx "<<nx<<"\nnsamp "<<nsamp<<"\ndt_sample "<<dstride*rp.dt
        <<"\ntoff "<<S.toff<<"\nLx "<<Lx<<"\nM "<<S.M<<"\nw0 "<<w0
        <<"\nband_lo "<<S.band_lo<<"\nband_hi "<<S.band_hi<<"\nvphx "<<w0/k0<<"\n"; }
    std::printf("wrote %skt.bin (%d samples x %d), video (%d frames)\n", pref, nsamp, nx, nframe);
    return 0;
}
