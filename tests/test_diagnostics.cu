// ArcWarden — Step 12 test: diagnostic reductions against known states.
//
// Builds states with closed-form energy/charge and checks the reductions:
//   - Fields Ex=a, Ey=b uniform => EE = ½ eps0 (a²+b²) Ncell dx dy, max|E|=√(a²+b²).
//   - rho = c uniform           => charge = c Ncell dx dy.
//   - particles loaded Maxwellian => KE ≈ weight · ½ · N · 3 vth².
// Also checks the CSV file gets a row.

#include "pic/diagnostics.hpp"
#include "pic/fields.hpp"
#include "pic/grid.hpp"
#include "pic/particles.hpp"
#include "pic/sources.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace arc;

int main() {
    const int nx = 16, ny = 8;
    Grid g(nx, ny, 4.0, 2.0);       // dx=0.25, dy=0.25
    const int Ncell = g.real_size();
    RunParams rp;
    rp.eps0 = 1.0; rp.dt = 0.1; rp.qm = -1.0; rp.vth = 1.0; rp.ppc = 64;
    rp.n0 = 1.0; rp.weight = rp.n0 * g.dx * g.dy / rp.ppc;

    CudaStream stream;
    bool ok = true;

    // ---- grid state: uniform Ex=a, Ey=b, rho=c ----
    const float a = 0.3f, b = -0.4f, c = 2.0f;
    Fields fld(g);
    Sources src(g);
    {
        std::vector<float> ex(Ncell, a), ey(Ncell, b), rho(Ncell, c);
        CUDA_CHECK(cudaMemcpy(fld.Ex.data(), ex.data(), fld.Ex.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fld.Ey.data(), ey.data(), fld.Ey.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(src.rho.data(), rho.data(), src.rho.bytes(), cudaMemcpyHostToDevice));
    }

    Particles parts;
    parts.initialize(rp, g, stream);
    stream.synchronize();
    const int N = int(parts.n);

    const std::string csv = "diag_test.csv";
    rp.dump_every = 1;                       // write every step
    Diagnostics diag(csv);
    diag.set_geometry(g.dx, g.dy);
    diag.maybe_compute(7, parts.views(), src, fld, rp, stream);
    DiagSample s = diag.last();

    // ---- expected ----
    const double ee_expect    = 0.5 * rp.eps0 * (double(a)*a + double(b)*b) * Ncell * g.dx * g.dy;
    const double maxe_expect  = std::sqrt(double(a)*a + double(b)*b);
    const double chg_expect   = double(c) * Ncell * g.dx * g.dy;
    const double ke_expect    = rp.weight * 0.5 * N * 3.0 * rp.vth * rp.vth;  // quiet std≈vth

    std::printf("EE=%.6f (exp %.6f)  max|E|=%.6f (exp %.6f)\n", s.ee, ee_expect, s.max_e, maxe_expect);
    std::printf("charge=%.6f (exp %.6f)  KE=%.4f (exp %.4f)  time=%.2f\n",
                s.charge, chg_expect, s.ke, ke_expect, s.time);

    if (std::fabs(s.ee     - ee_expect)   > 1e-5)               { std::printf("FAIL EE\n");     ok = false; }
    if (std::fabs(s.max_e  - maxe_expect) > 1e-6)               { std::printf("FAIL max|E|\n"); ok = false; }
    if (std::fabs(s.charge - chg_expect)  > 1e-4)               { std::printf("FAIL charge\n"); ok = false; }
    if (std::fabs(s.ke     - ke_expect)   > 0.02 * ke_expect)   { std::printf("FAIL KE\n");     ok = false; }
    if (std::fabs(s.total  - (s.ke + s.ee)) > 1e-9)             { std::printf("FAIL total\n");  ok = false; }
    if (std::fabs(s.time   - 0.7)         > 1e-9)               { std::printf("FAIL time\n");   ok = false; }

    // ---- CSV row written ----
    {
        std::ifstream f(csv);
        std::string header, row;
        std::getline(f, header); std::getline(f, row);
        if (header.rfind("step,time,ke", 0) != 0 || row.rfind("7,", 0) != 0) {
            std::printf("FAIL: CSV not written correctly (row='%s')\n", row.c_str());
            ok = false;
        }
    }

    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
