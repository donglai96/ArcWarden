// pic2d — Sim2D: deck-driven multi-species orchestrator (P3a).
//
// Owns the field engine + N kinetic species and runs the validated step
// sequence (the exact order the V1/V3 gates ran inline):
//     faraday ½ → zero J → per-species push+deposit → reduce → jfilter →
//     cold fluid (if nc > 0) → faraday ½ → ampère → masks
// Species are deck-defined ([species <name>] blocks); each carries its own
// KineticCfg (f₀, δf/full-f, dist) and MarkerStore — multi-δf is the
// architecture, not a special case (PLAN_2D_REBORN §2.2).
//
// P3a scope: build-from-deck, step, energy/wd monitors, decimated field
// snapshots. The §2.5 ledger pack (J·E by species × channel, fv monitors,
// WNA/k⊥ products) lands P3b; checkpoint P3c.

#ifndef ARC_PIC2D_SIM2D_HPP
#define ARC_PIC2D_SIM2D_HPP

#include "pic2d/deck2d.hpp"
#include "pic2d/diag2d.hpp"
#include "pic2d/fields2d.hpp"
#include "pic2d/kinetic2d.hpp"
#include "pic2d/sort2d.hpp"

#include <memory>
#include <string>
#include <vector>

namespace arc2d {

struct Sim2D {
    Fields2D F;
    struct Sp {
        std::string name;
        KineticCfg C;
        std::unique_ptr<MarkerStore> mk;
    };
    std::vector<Sp> sp;
    Diag2D diag;
    Diag2D diag2;                      // optional off-L0 probe line
    Sorter2D sorter;
    arc::DeviceArray<unsigned long long> runaway;   // ucap clamp counter
    bool diag_on = false;
    bool diag2_on = false;
    long sort_every = 25;              // 0 = off (markers drift ~0.06 cells/step)
    int deposit_tiled = 1;             // shared-window deposit (needs sort)
    bool sorted_once = false;
    double time = 0;
    long nstep = 0;
    arc::DeviceArray<double> acc;
    arc::DeviceArray<unsigned int> wdmax_dev;      // small reduction scratch

#ifdef __CUDACC__
    void build(const Deck2D& d) {
        F.nx = d.nx; F.nz = d.nz;
        F.dx = d.dx; F.dz = d.dz; F.dt = d.dt;
        F.cspeed = d.cspeed; F.nc = d.nc;
        F.x0 = d.x0; F.z0 = d.z0;
        F.bg = d.bg;
        if (d.active_Lmax > d.active_Lmin && d.active_Lmin > 0)
            F.build_tiles(d.active_Lmin, d.active_Lmax);  // BEFORE allocate
        F.allocate(d.nx, d.nz);
        F.build_masks(d.absorber_cells, 0.05);   // uses the band if set
        // replica heuristic: only worthwhile on small, contended grids
        const double ppc_tot = [&] {
            double s = 0;
            for (const auto& q : d.species) s += q.ppc;
            return s;
        }();
        if (size_t(d.nx) * d.nz < 1u << 20 && ppc_tot > 256) F.allocate_replicas(16);
        acc = arc::DeviceArray<double>(4);
        wdmax_dev = arc::DeviceArray<unsigned int>(1);
        runaway = arc::DeviceArray<unsigned long long>(3);  // [0] ucap clamps,
        runaway.zero();               // [1] dropped deposits, [2] load exhausts

        const bool dipole = has_lines(d.bg);
        for (const auto& q : d.species) {
            Sp s;
            s.name = q.name;
            KineticCfg& C = s.C;
            C.qm = -1.f;
            C.deltaf = q.deltaf ? 1 : 0;
            C.rel = q.rel;
            C.dist = q.dist;
            C.kappa = float(q.kappa);
            C.tpar = float(q.uthpar * q.uthpar);
            C.tperp = float(q.uthperp * q.uthperp);
            C.n0 = float(q.n0);
            C.taud = float(q.taud);
            C.L0 = float(q.shell_L0);
            C.dL = float(q.shell_dL);
            C.edge = float(q.edge_dL);
            C.wdnoise = float(q.wdnoise);
            // walls at the high-|λ| line ends only (P2 ruling): inner-x and
            // ±z at the mask interior edge; NO outer-radial wall
            const double nd = d.absorber_cells;
            C.wx0 = float(d.x0 + nd * d.dx);
            C.wx1 = 1e9f;
            C.wz0 = float(d.z0 + nd * d.dz);
            C.wz1 = float(d.z1 - nd * d.dz);
            // loader bounding box: shell support ∩ interior (dipole) or box
            float bx0, bx1, bz0, bz1;
            if (dipole) {
                bx0 = C.wx0;
                bx1 = std::min(float(d.x1 - nd * d.dx),
                               float(q.shell_L0 + 0.5 * q.shell_dL + 3 * q.edge_dL + 1));
                bz0 = C.wz0;
                bz1 = C.wz1;
            } else {
                bx0 = float(d.x0); bx1 = float(d.x1);
                bz0 = float(d.z0); bz1 = float(d.z1);
            }
            const double nint = dipole
                ? k2d::shell_density_integral(C, d.bg, bx0, bx1, bz0, bz1, 0.5)
                : double(bx1 - bx0) * (bz1 - bz0);
            const uint64_t N = q.nmax > 0
                ? q.nmax
                : uint64_t(double(q.ppc) * (bx1 - bx0) * (bz1 - bz0) / (d.dx * d.dz));
            const float wmark = float(q.n0 * nint / double(N));
            s.mk = std::make_unique<MarkerStore>();
            s.mk->allocate(N);
            s.mk->n = N;
            MarkerViews mv = s.mk->views();
            k2d::k_load<<<int((N + 255) / 256), 256>>>(mv, C, d.bg, bx0, bx1,
                                                       bz0, bz1, wmark,
                                                       20260817u + uint32_t(sp.size()),
                                                       N, runaway.data() + 2);
            CUDA_CHECK(cudaDeviceSynchronize());
            sp.push_back(std::move(s));
        }
        {   // loader-exhaust report (fallback markers sit at the shell centre)
            unsigned long long rr[3];
            CUDA_CHECK(cudaMemcpy(rr, runaway.data(), 24, cudaMemcpyDeviceToHost));
            if (rr[2])
                std::printf("  load: %llu exhausted rejections (%.1e of total) "
                            "placed at shell centre\n",
                            rr[2], double(rr[2]) / std::max(1.0, [&] {
                                double s = 0;
                                for (auto& q : sp) s += double(q.mk->n);
                                return s;
                            }()));
        }
        if (dipole) {
            diag.build_line(d.bg, d.bg.L0, d.lam_w * 180 / M_PI, 1.0,
                            int(sp.size()));
            diag_on = true;
            if (d.probe_L2 > 0) {
                diag2.build_line(d.bg, d.probe_L2, d.lam_w * 180 / M_PI, 1.0, 0);
                diag2_on = true;
            }
        }
    }

    // diagnostics passes (cadences owned by the runner)
    void diag_line() {
        if (diag_on) diag.sample_line(F.views(), float(F.x0), float(F.z0));
        if (diag2_on) diag2.sample_line(F.views(), float(F.x0), float(F.z0));
    }
    void diag_fv() {
        if (!diag_on) return;
        for (size_t i = 0; i < sp.size(); ++i)
            diag.fv_accumulate(int(i), sp[i].mk->views(), F.bg, sp[i].mk->n);
    }
    void diag_ledger(float dt_eff) {
        if (!diag_on) return;
        for (size_t i = 0; i < sp.size(); ++i)
            diag.ledger_accumulate(int(i), sp[i].mk->views(), sp[i].C,
                                   F.views(), F.bg, float(F.x0), float(F.z0),
                                   dt_eff, sp[i].mk->n);
    }

    void step() {
        if (sort_every > 0 && (nstep % sort_every == 0 || !sorted_once)) {
            for (auto& s : sp)
                sorter.sort(*s.mk, float(F.x0), float(F.z0), float(F.dx),
                            float(F.dz), F.nx, F.nz);
            sorted_once = true;
        }
        FieldViews2D v = F.views();
        const Range r = F.full();
        const dim3 nb = f2d::blocks_for(r), tb(f2d::TX, f2d::TZ);
        f2d::k_faraday<<<nb, tb>>>(v, r, float(F.dt / 2));
        const bool tiled = deposit_tiled && sort_every > 0 && sorted_once;
        if (tiled) {
            F.jx.zero(); F.jy.zero(); F.jz.zero();
            const dim3 tg((F.nx + k2d::TS_TILE - 1) / k2d::TS_TILE,
                          (F.nz + k2d::TS_TILE - 1) / k2d::TS_TILE);
            for (auto& s : sp) {
                MarkerViews mv = s.mk->views();
                k2d::k_push_deposit_tiled<<<tg, 256>>>(
                    mv, s.C, v, F.bg, float(F.x0), float(F.z0),
                    s.mk->cell_start.data(), runaway.data());
            }
        } else {
            F.zero_j();
            for (auto& s : sp) {
                MarkerViews mv = s.mk->views();
                k2d::k_push_deposit<<<int((s.mk->n + 255) / 256), 256>>>(
                    mv, s.C, v, F.bg, float(F.x0), float(F.z0), s.mk->n,
                    runaway.data());
            }
            F.reduce_j();
        }
        F.filter_j();
        if (F.nc > 0.0) {
            f2d::k_cold_step<<<nb, tb>>>(v, r, F.bg, float(F.x0), float(F.z0));
            f2d::k_cold_current<<<nb, tb>>>(v, r);
        }
        f2d::k_faraday<<<nb, tb>>>(v, r, float(F.dt / 2));
        f2d::k_ampere<<<nb, tb>>>(v, r);
        if (F.masks_on) {
            f2d::k_mask_e<<<nb, tb>>>(v, r);
            f2d::k_mask_b<<<nb, tb>>>(v, r);
        }
        time += F.dt;
        ++nstep;
    }

    // periodic validity sweep (cheap insurance for overnight runs): counts
    // non-finite markers + checks W_EM finiteness. The RUNNER decides what
    // to do on failure (write an emergency checkpoint, then abort) — the
    // V4R2 lesson: pathologies can surface hours in; a poisoned run must
    // die loudly WITH a resumable state, not corrupt silently.
    bool healthy() {
        acc.zero();
        for (auto& s : sp) {
            MarkerViews mv = s.mk->views();
            k2d::k_finite_scan<<<int((s.mk->n + 255) / 256), 256>>>(
                mv, acc.data(), s.mk->n);
        }
        double bad;
        CUDA_CHECK(cudaMemcpy(&bad, acc.data(), 8, cudaMemcpyDeviceToHost));
        unsigned long long rr[2];
        CUDA_CHECK(cudaMemcpy(rr, runaway.data(), 16, cudaMemcpyDeviceToHost));
        double W[2];
        F.energies(W);
        // delta-f representation gate, redesigned on V4R9 ckpt data
        // (2026-08-19): wd = 1 - f0/f < 1 ALWAYS, so a max-based gate
        // trips trivially; the collapse signal is the hole-tail fraction
        // (deep depletion, wd << 0). V4R9 at deep saturation measured
        // frac(|wd|>3) = 0.22% with healthy dynamics — gate at 2%.
        for (size_t i = 0; i < sp.size(); ++i)
            if (sp[i].C.deltaf) {
                wd_rms(int(i));
                double a[2];
                CUDA_CHECK(cudaMemcpy(a, acc.data(), 16,
                                      cudaMemcpyDeviceToHost));
                if (a[1] / double(sp[i].mk->n) > 0.02) return false;
            }
        return bad == 0.0 && rr[1] == 0 && std::isfinite(W[0]) &&
               std::isfinite(W[1]);
    }

    unsigned long long runaway_count() {
        unsigned long long h;
        CUDA_CHECK(cudaMemcpy(&h, runaway.data(), 8, cudaMemcpyDeviceToHost));
        return h;
    }

    double wd_rms(int is, float* wd_max = nullptr) {
        acc.zero();
        wdmax_dev.zero();
        MarkerViews mv = sp[is].mk->views();
        k2d::k_wd_stats<<<int((sp[is].mk->n + 255) / 256), 256>>>(
            mv, acc.data(), wdmax_dev.data(), sp[is].mk->n);
        double h;
        CUDA_CHECK(cudaMemcpy(&h, acc.data(), sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (wd_max) {
            unsigned int u;
            CUDA_CHECK(cudaMemcpy(&u, wdmax_dev.data(), 4,
                                  cudaMemcpyDeviceToHost));
            float f;
            std::memcpy(&f, &u, 4);   // monotone encoding: uint max = float max
            *wd_max = f;
        }
        return std::sqrt(h / double(sp[is].mk->n));
    }
#endif
};

}  // namespace arc2d

#endif  // ARC_PIC2D_SIM2D_HPP
