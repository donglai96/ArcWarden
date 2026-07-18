// M3 weight-precision study [docs/WEIGHT_PRECISION.md]: does FP32 accumulation
// of the delta-f weight wd drift over production-length runs (~1e5 steps)?
//
// Three accumulators (RunParams::df_wprec, flat deposit path only):
//   0  FP32               wd += inc                     (production today)
//   1  FP32 + Kahan       compensated sum, __fadd_rn    (aux float wc)
//   2  FP64 reference     double wdd, mirrored to wd    (aux double wdd)
//
// Two physical cases, identical initial conditions across precisions (same
// quiet load, same mt19937 grid-noise seed), run back to back:
//   equil   iso Maxwellian + 1e-6 B noise — no instability, wd stays at its
//           noise floor; the cleanest window on pure accumulation roundoff.
//   growth  test_deltaf_growth config (bi-Max, A=4, wce=0.25, mode m=2 at
//           k=1.636, gamma_theory=2.872e-3) — wd grows exponentially; gamma
//           fit per precision + per-particle divergence in the linear phase.
//
// Divergence metric: reldiff(p) = rms(wd_p - wd_ref64) / rms(wd_ref64) at each
// snapshot (particle order is stable on the flat path — no sort, migrate wraps
// in place — so per-particle comparison is valid). CAVEAT for the growth case:
// wd feeds back through the deposit, so any roundoff difference also seeds
// trajectory divergence; past a few e-folds the per-particle diff measures
// marker chaos, not accumulator quality — the statistical numbers (gamma, wd
// rms trajectory) are the honest metric there.
//
// Usage: ./weight_precision [nsteps=100000] [snap=5000]
// Output: table on stdout + build/weight_precision_{equil,growth}.csv

#include "pic/simulation_maxwell.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace arc;

struct Series {
    std::vector<double> t;
    std::vector<MaxwellSimulation::WdStats> st;
    std::vector<std::vector<double>> wd;   // per-particle snapshot (as double)
    std::vector<double> lp;                // 0.5*log mode-2 |B_k|^2 (growth)
};

static Series run_case(bool growth, int prec, long nsteps, int snap) {
    const int nx = 256, ny = 1, ppc = 1600;
    const double c = 1.0, wce = 0.25, dx = 0.03, dt = 0.02;
    const double apar  = growth ? 0.01678 : 0.03752;
    const double aperp = 0.03752;
    Grid g(nx, ny, nx * dx, dx);

    RunParams rp;
    rp.dt = dt; rp.c = c; rp.qm = -1.0; rp.eps0 = 1.0;
    rp.B0[0] = (float)wce; rp.wce = wce;
    rp.noisy_load = false; rp.dump_every = 0;
    rp.rng_seed = 20260718UL;
    rp.deltaf = 1; rp.df_tpar = apar * apar; rp.df_tperp = aperp * aperp;
    rp.df_wprec = prec;

    SpeciesList sp = { Species{"e", 1.0, ppc, {apar, aperp, aperp}, {0, 0, 0}, true} };
    MaxwellSimulation sim(g, rp);
    sim.particles().initialize(sp, g, rp, sim.stream());
    sim.particles().enable_deltaf(sim.stream(), prec);
    sim.stream().synchronize();

    // identical grid-noise seed for every precision
    std::mt19937 gen(20260718u);
    std::uniform_real_distribution<float> un(-1e-6f, 1e-6f);
    std::vector<float> by(g.real_size()), bz(g.real_size());
    for (auto& v : by) v = un(gen);
    for (auto& v : bz) v = un(gen);
    CUDA_CHECK(cudaMemcpy(sim.fields().by_.data(), by.data(), by.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.fields().bz_.data(), bz.data(), bz.size() * 4, cudaMemcpyHostToDevice));

    const std::size_t n = sim.particles().n;
    std::vector<float> wf(n);
    const double k2 = 2.0 * M_PI * 2.0 / (nx * dx);

    Series se;
    for (long s = 1; s <= nsteps; ++s) {
        sim.step();
        if (s % snap) continue;
        se.t.push_back(s * dt);
        se.st.push_back(sim.wd_stats());
        std::vector<double> w(n);
        if (prec == 2) {
            CUDA_CHECK(cudaMemcpy(w.data(), sim.particles().wdd.data(), n * 8, cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(wf.data(), sim.particles().wd.data(), n * 4, cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < n; ++i) w[i] = wf[i];
        }
        se.wd.push_back(std::move(w));
        if (growth) {
            CUDA_CHECK(cudaMemcpy(by.data(), sim.fields().by_.data(), by.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bz.data(), sim.fields().bz_.data(), bz.size() * 4, cudaMemcpyDeviceToHost));
            double cr = 0, ci = 0, dr = 0, di = 0;
            for (int i = 0; i < nx; ++i) {
                const double ph = k2 * (i + 0.5) * dx;
                cr += by[i] * std::cos(ph); ci -= by[i] * std::sin(ph);
                dr += bz[i] * std::cos(ph); di -= bz[i] * std::sin(ph);
            }
            se.lp.push_back(0.5 * std::log(cr * cr + ci * ci + dr * dr + di * di + 1e-300));
        }
    }
    return se;
}

static double fit_slope(const std::vector<double>& t, const std::vector<double>& y,
                        std::size_t i0, std::size_t i1) {
    double st = 0, sy = 0, stt = 0, sty = 0; const double m = (double)(i1 - i0);
    for (std::size_t i = i0; i < i1; ++i) {
        st += t[i]; sy += y[i]; stt += t[i] * t[i]; sty += t[i] * y[i];
    }
    return (m * sty - st * sy) / (m * stt - st * st);
}

int main(int argc, char** argv) {
    long nsteps = argc > 1 ? atol(argv[1]) : 100000;
    int  snap   = argc > 2 ? atoi(argv[2]) : 5000;
    const char* pname[3] = {"fp32", "kahan", "fp64"};

    for (int gc = 0; gc < 2; ++gc) {
        const bool growth = gc == 1;
        std::printf("==== case %s: nsteps=%ld snap=%d ====\n",
                    growth ? "growth" : "equil", nsteps, snap);
        Series se[3];
        for (int p = 0; p < 3; ++p) se[p] = run_case(growth, p, nsteps, snap);

        const std::string csvn = std::string("weight_precision_") +
                                 (growth ? "growth" : "equil") + ".csv";
        FILE* csv = fopen(csvn.c_str(), "w");
        std::fprintf(csv, "t,rms_ref,max_ref,sum_ref,sum_fp32,sum_kahan,"
                          "reldiff_fp32,reldiff_kahan\n");
        std::printf("%8s %11s %11s %12s %12s %14s %14s\n", "t", "rms_ref",
                    "max_ref", "sum_ref", "sum_fp32", "reldiff_fp32", "reldiff_kahan");
        for (std::size_t k = 0; k < se[2].t.size(); ++k) {
            const auto& ref = se[2].wd[k];
            const std::size_t n = ref.size();
            double s2 = 0, d0 = 0, d1 = 0;
            for (std::size_t i = 0; i < n; ++i) {
                s2 += ref[i] * ref[i];
                const double e0 = se[0].wd[k][i] - ref[i];
                const double e1 = se[1].wd[k][i] - ref[i];
                d0 += e0 * e0; d1 += e1 * e1;
            }
            const double rref = std::sqrt(s2 / n);
            const double r0 = std::sqrt(d0 / n) / (rref + 1e-300);
            const double r1 = std::sqrt(d1 / n) / (rref + 1e-300);
            std::fprintf(csv, "%.6g,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                         se[2].t[k], rref, se[2].st[k].max, se[2].st[k].sum,
                         se[0].st[k].sum, se[1].st[k].sum, r0, r1);
            std::printf("%8.0f %11.4e %11.4e %12.4e %12.4e %14.4e %14.4e\n",
                        se[2].t[k], rref, se[2].st[k].max, se[2].st[k].sum,
                        se[0].st[k].sum, r0, r1);
        }
        fclose(csv);
        if (growth) {
            // gamma over the same last-third window as the gate test
            const std::size_t i0 = 2 * se[2].t.size() / 3, i1 = se[2].t.size();
            std::printf("gamma (last third, theory 2.872e-3):");
            for (int p = 0; p < 3; ++p)
                std::printf("  %s %.4e", pname[p], fit_slope(se[p].t, se[p].lp, i0, i1));
            std::printf("\n");
        }
        std::printf("csv: %s\n\n", csvn.c_str());
    }
    return 0;
}
