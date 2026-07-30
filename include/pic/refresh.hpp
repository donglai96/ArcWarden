// M-refresh — boundary-refresh thermal bath ("infinite train" experiment,
// docs/REFRESH_DESIGN.md, design record 2026-07-22).
//
// Rule (stateless, no crossing detection, no history): any hot marker inside
// the THIN SHELL s_R < |s| < s_R + w (s_R = s(lambda_R), w = [refresh] shell)
// that is moving OUTWARD (u_par carrying it away from the equator) has its
// velocity redrawn from f0(local) every step; position untouched; inward
// movers untouched. A marker is therefore refreshed ~once per outward
// crossing of lambda_R (it exits the shell either inward as a fresh f0
// sample or outward to free-fly, mirror, and return untouched) — the supply
// clock IS the marker's own bounce traffic through lambda_R, no imposed
// timescale (the knob-free contrast to taud/tau_D injection, Lu 2021 /
// Chen 2022 / Kong 2023).
//
// WHY A SHELL AND NOT THE WHOLE |s| > s_R REGION (the design note's first
// reading) — both wide-bath variants FAILED the Gate-1 null test on the
// Chen-2026 hybrid-boundary deck, each through a different absorber
// interaction (2026-07-26 forensics, kept as methods material):
//   1. wide bath, unconditioned f0 redraw: the Umeda hybrid layer carves a
//      cold-parallel pseudo-loss-cone population; the bath resurrected its
//      u_perp every outward pass and the layer re-drained it — an energy
//      churn measured at +16x hot KE per 450/wpe.
//   2. wide bath, trapped-conditioned redraw (mirror point before the
//      absorber): near the absorber the trapped cone closes (u_par^2 <
//      0.03 u_perp^2), so the deep bath was rebuilt as a u_par-starved,
//      perp-piled population (Tpar -45%, Tperp +12% in ONE application) —
//      the bath itself became a whistler-instability pump (+4.5x KE,
//      interior Tperp +40% in 450/wpe).
// The thin shell keeps the refresh at the lambda_R crossing only. Even so,
// two more hybrid-deck variants failed:
//   3. thin shell, unconditioned redraw: still fed the absorber at +40% hot
//      KE per 450/wpe — the absorber-bound part of each redraw is burned
//      once per bounce, a huge ARTIFICIAL loss-cone refill power (the
//      compressed box makes the "loss cone" alpha_eq < 43 deg!) that the
//      Chen-2026 closed system does not have.
//   4. thin shell, trapped-conditioned redraw (mirror before the absorber):
//      converts the ~40% parallel-leaning share of every crossing into
//      perp-leaning samples — near-shell Tperp +40%, same magnitude as 2.
// VERDICT: refresh + bnd_x = 2 (hybrid particle-damping layers) is
// UNSUPPORTED — the absorber makes the closed steady state deviate from f0,
// and a f0-restoring bath fights it whichever way it is conditioned (all
// four variants measured, init() throws). Use [boundary] x = damping
// (fields damped, particles reflect specularly = energy-conserving closed
// particle system). That is also CLOSER to Chen 2026's own "bare
// reflection" boundary than our hybrid deviation.
//
// f0(local): the (E,mu)-mapped mirror equilibrium is CLOSED FORM at every x
// (Tpar = const, 1/Tperp(x) = (1-1/b)/Tpa + (1/b)/Tpe_eq, loss-cone
// subtraction with the local T2) — the sampling below is math-verbatim from
// mirror_init_kernel (particles.hpp), so a redraw is distribution-identical
// to the loader (verified by the zero-dynamics sampler probe in
// test_refresh_null: one application moves no bin moment beyond +-0.4%).
// The design note's equatorial-draw + (E,mu) transport + <=16-retry ladder
// is unnecessary: the closed form needs no reachability rejection, and the
// only rejection loop left is the loader's own loss-cone envelope loop,
// kept verbatim (64 iters, break on accept).
//
// Feature isolation (user requirement): everything lives here; callers add
// ONE guarded call after sim.step() (runner level — simulation_maxwell.hpp
// is untouched) and ONE additive [refresh] deck block. refresh = 0: no
// allocations, no kernel launches, byte-for-byte legacy behavior.
//
// v1 scope: full-f only (a delta-f redraw would also need wd := 0 — deferred
// until an experiment needs it), single hot species, dipole profile only
// (lambda_R is a dipole-latitude concept). Diagnostics: drain() returns the
// interval's redraw count and net injected kinetic energy (refresh.csv).
// STEADY-STATE exchange: inflow = outflow, n_h fixed (1:1 exit swap); net
// injection (substorm n_h(t) ramp) is a reserved future extension.
//
// RNG: hashed rng_uniform with a per-step-derived seed (golden-ratio step
// mix), so every step is a fresh independent stream and a --resume replays
// the identical sequence (stateless in the checkpoint sense: nothing to
// save). Dims 1..4 and 10+2k mirror the loader's for math-identity clarity;
// collisions with the load streams are irrelevant (different seed).

#pragma once

#include "pic/background_b0.hpp"
#include "pic/particles.hpp"
#include "pic/species.hpp"

#include <cmath>
#include <stdexcept>

namespace arc {
namespace detail {

struct RefreshParams {
    double s_r         = 0.0;   // shell inner |s| = s(lambda_R) (physical units)
    double s_out       = 0.0;   // shell outer |s| = s_r + shell width. The bath
                                // is a THIN SHELL, not the whole high-latitude
                                // region — the wide bath failed the null test
                                // two independent ways (see header). The shell
                                // refreshes each marker ~once per outward
                                // crossing; beyond it, free flight to the
                                // mirror point and back, untouched.
    double uth_par     = 0.0;   // f0: parallel thermal width (equatorial = local)
    double uth_perp_eq = 0.0;   // f0: equatorial perp width
    int    dist        = 0;     // 0 = bi-Max, 1 = loss-cone subtracted
    double lc_rho      = 1.0;
    double lc_kappa    = 0.3;

    // REAL-PRECIPITATION mode ([refresh] precip, requires bnd_x = 1): any
    // marker inside the one-cell wall strips (x < xw_lo or x > xw_hi) has
    // its u_perp ZEROED (u_par kept — the reflected ghost streams home at
    // its incident parallel speed, mu = 0 so no mirror trapping, transit
    // ~ T_b/5, minimal density pile-up) and the removed energy counted as
    // precipitated. Only true loss-cone markers (alpha_eq < asin(b_wall
    // ^-1/2)) ever reach the strips — unlike the hybrid layers, the
    // 39.5-43.5 deg near-cone buffer band is never touched. The ghost is
    // recycled into a fresh trapped-f0 sample at its first OUTWARD shell
    // crossing on the far side (the bath's ordinary rule) — precipitation
    // removes the energy, the bath resupplies it at the bounce-flux rate:
    // the complete loss+refill loop with a measurable net budget
    // (dE_injected - dE_precip), at fixed N and with velocity-only ops
    // (charge continuity untouched).
    int    precip      = 0;
    double xw_lo       = 0.0;   // wall strips (physical x)
    double xw_hi       = 0.0;
    int    precip_soft = 0;     // 0 = HARD zero u_perp (default). 1 = SOFT
                                // depth-graded exp fold like the hybrid bnd_x=2
                                // layer (uy *= exp(-numax*dt*d^2), d = strip
                                // depth 0->1): keeps the mid-lat amplifier warm
                                // so the wave retains its convective gain,
                                // instead of the hard zero's cold beam that
                                // flattens it. Test of the soft-vs-hard gap.
    double precip_numax = 0.1;  // fold rate for precip_soft (match bnd_numax)
};

// One thread per marker; bath outward-movers redraw, everyone else falls
// through to the block reduction with (0,0). acc[0] += redraw count,
// acc[1] += net injected kinetic energy (weight-carried, rel-consistent).
__global__ void refresh_kernel(ParticleViews p, Grid g, RunParams rp,
                               RefreshParams rf, unsigned long seed,
                               double* __restrict__ acc, long n) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    double cnt = 0.0, dE = 0.0, pcnt = 0.0, pdE = 0.0;
    if (t < n) {
        const float xph = p.x[t] * (float)g.dx;
        const float d   = xph - (float)rp.b0_xc;      // signed dist from equator
        const float ux0 = p.ux[t];
        const float ad = fabsf(d);
        const int ti = static_cast<int>(t);
        if (ad > (float)rf.s_r && ad < (float)rf.s_out      // thin shell
            && ux0 * d > 0.f                                // && outward
            // Probability gate q = |u_par| dt / w: crossing the shell takes
            // w/|u_par| of time, so the expected redraws per outward crossing
            // is EXACTLY 1, velocity-independent, with zero crossing-detection
            // state. Without it (redraw every step while in the shell, ~250x
            // per crossing) the shell current is re-randomized every step and
            // becomes a white-noise whistler antenna: +38% hot KE per 450/wpe
            // radiated into the box even with reflecting walls (variant 5 of
            // the Gate-1 forensics; only ~1/5 of the ledger stayed in the
            // particles — the rest left as waves).
            && rng_uniform(ti, 0, seed)
                   < fabs((double)ux0) * rp.dt / (rf.s_out - rf.s_r)) {
            // ---- f0(local) sample: math verbatim from mirror_init_kernel ----
            const double b   = (double)bg::b0x(rp, xph) / (double)rp.B0[0];
            const double Tpa = rf.uth_par * rf.uth_par;
            const double Tpe = rf.uth_perp_eq * rf.uth_perp_eq;
            const double Tp  = 1.0 / ((1.0 - 1.0 / b) / Tpa + (1.0 / b) / Tpe);
            const double r1 = fmax(rng_uniform(ti, 1, seed), 1e-12);
            const double r2 = rng_uniform(ti, 2, seed);
            const double r4 = rng_uniform(ti, 4, seed);
            // u_par is FLUX-weighted (|u| Rayleigh, random sign), NOT the
            // loader's density-weighted Gaussian: the swap happens once per
            // CROSSING (a flux event), so the replacement stream must carry
            // the flux-weighted f0 u_par or the returned flux runs cold
            // (classic boundary-injection result; measured as a -6.8%
            // shell-adjacent interior Tpar dip with the Gaussian draw).
            const double upar = rf.uth_par * sqrt(-2.0 * log(r1))
                              * (r2 < 0.5 ? -1.0 : 1.0);
            double uperp;
            if (rf.dist == 1) {
                const double T2 = 1.0 / ((1.0 - 1.0 / b) / Tpa
                                         + (1.0 / b) / (rf.lc_kappa * Tpe));
                const double dinv = 0.5 * (1.0 / T2 - 1.0 / Tp);
                double u2 = 0.0;
                for (int k = 0; k < 64; ++k) {
                    const double rm = fmax(rng_uniform(ti, 10 + 2 * k, seed), 1e-12);
                    const double ra = rng_uniform(ti, 11 + 2 * k, seed);
                    u2 = -2.0 * Tp * log(rm);
                    if (ra < 1.0 - rf.lc_rho * exp(-u2 * dinv)) break;
                }
                uperp = sqrt(u2);
            } else {
                const double r3 = fmax(rng_uniform(ti, 3, seed), 1e-12);
                uperp = sqrt(Tp) * sqrt(-2.0 * log(r3));
            }
            {
                const float nux = static_cast<float>(upar);
                const float nuy = static_cast<float>(uperp * cos(2.0 * M_PI * r4));
                const float nuz = static_cast<float>(uperp * sin(2.0 * M_PI * r4));
                // ---- energy ledger (diagnostic only) ----
                const float uy0 = p.uy[t], uz0 = p.uz[t];
                const double u0sq = (double)ux0 * ux0 + (double)uy0 * uy0
                                  + (double)uz0 * uz0;
                const double u1sq = (double)nux * nux + (double)nuy * nuy
                                  + (double)nuz * nuz;
                const double e0 = rp.rel ? sqrt(1.0 + u0sq) - 1.0 : 0.5 * u0sq;
                const double e1 = rp.rel ? sqrt(1.0 + u1sq) - 1.0 : 0.5 * u1sq;
                dE  = (double)p.w[t] * (e1 - e0);
                cnt = 1.0;
                p.ux[t] = nux; p.uy[t] = nuy; p.uz[t] = nuz;
            }
        } else if (rf.precip && (xph < (float)rf.xw_lo || xph > (float)rf.xw_hi)) {
            // real precipitation: remove u_perp inside the wall strip, count
            // the removed energy (see RefreshParams::precip). u_par kept.
            const float uy0 = p.uy[t], uz0 = p.uz[t];
            const double up2 = (double)uy0 * uy0 + (double)uz0 * uz0;
            if (up2 > 0.0) {
                // HARD (default): m=0 -> full zero. SOFT: depth-graded exp fold
                // like the hybrid layer (d = 0 at inner strip edge -> 1 at wall).
                float m = 0.f;
                if (rf.precip_soft) {
                    const float d = (xph < (float)rf.xw_lo)
                        ? ((float)rf.xw_lo - xph) / (float)rf.xw_lo
                        : (xph - (float)rf.xw_hi)
                              / ((float)(g.nx * g.dx) - (float)rf.xw_hi);
                    m = __expf((float)(-rf.precip_numax * rp.dt) * d * d);
                }
                const float nuy = uy0 * m, nuz = uz0 * m;
                const double up2n = (double)nuy * nuy + (double)nuz * nuz;
                const double upar2 = (double)ux0 * ux0;
                const double e0 = rp.rel ? sqrt(1.0 + upar2 + up2) - 1.0
                                         : 0.5 * (upar2 + up2);
                const double e1 = rp.rel ? sqrt(1.0 + upar2 + up2n) - 1.0
                                         : 0.5 * (upar2 + up2n);
                pdE = (double)p.w[t] * (e0 - e1);   // energy leaving the box
                // count only genuine arrivals (u_perp^2 above the wave-noise
                // jitter a strip-dwelling ghost re-accretes between zeroings,
                // ~(v dB/B)^2 ~ 1e-8) — the ENERGY ledger takes every event.
                pcnt = (up2 - up2n) > 1e-6 ? 1.0 : 0.0;
                p.uy[t] = nuy; p.uz[t] = nuz;
            }
        }
    }
    // block-wide reduction -> 5 atomics per block (a marker-wise atomicAdd
    // would serialize ~1e6 adds/step on one address)
    __shared__ double sc[256], se[256], sf[256], sp[256], sq[256];
    sc[threadIdx.x] = cnt > 0.0 ? cnt : 0.0;
    se[threadIdx.x] = dE;
    sf[threadIdx.x] = cnt < 0.0 ? -cnt : 0.0;
    sp[threadIdx.x] = pcnt;
    sq[threadIdx.x] = pdE;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            sc[threadIdx.x] += sc[threadIdx.x + off];
            se[threadIdx.x] += se[threadIdx.x + off];
            sf[threadIdx.x] += sf[threadIdx.x + off];
            sp[threadIdx.x] += sp[threadIdx.x + off];
            sq[threadIdx.x] += sq[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && (sc[0] != 0.0 || sf[0] != 0.0 || sp[0] != 0.0)) {
        atomicAdd(acc + 0, sc[0]);
        atomicAdd(acc + 1, se[0]);
        atomicAdd(acc + 2, sf[0]);
        atomicAdd(acc + 3, sp[0]);
        atomicAdd(acc + 4, sq[0]);
    }
}

} // namespace detail

struct RefreshState {
    detail::RefreshParams prm{};
    DeviceArray<double>   acc;    // [0] redraws, [1] injected energy (interval)
    bool                  on = false;

    // Validate + freeze the bath parameters. Call once after the mirror load;
    // throws on any unsupported combination (fail loud, not silently off).
    void init(const Species& q, const Grid& g, const RunParams& rp,
              cudaStream_t s) {
        if (!rp.refresh) return;
        if (rp.b0_prof != 2)
            throw std::runtime_error("refresh: needs [background] profile = "
                                     "dipole (lambda_R is a dipole latitude)");
        if (rp.deltaf || q.deltaf)
            throw std::runtime_error("refresh v1: full-f only (delta-f redraw "
                                     "needs wd := 0 — not wired)");
        if (q.uth[1] != q.uth[2])
            throw std::runtime_error("refresh: gyrotropy uth[1] == uth[2] required");
        if (rp.refresh_lambda <= 0.0 || rp.refresh_lambda >= 90.0)
            throw std::runtime_error("refresh: lambda_deg must be in (0, 90)");
        if (rp.refresh_shell <= 0.0)
            throw std::runtime_error("refresh: shell width must be > 0");
        if (rp.bnd_x == 2)
            throw std::runtime_error("refresh: hybrid particle-damping layers "
                                     "are unsupported — the absorber and any "
                                     "f0-restoring bath churn energy (4 variants "
                                     "measured, see refresh.hpp header); use "
                                     "[boundary] x = damping");
        const double lam = rp.refresh_lambda * M_PI / 180.0;
        prm.s_r   = rp.b0_lre * bg::dipole_s_of_lambda(lam);
        prm.s_out = prm.s_r + rp.refresh_shell;
        if (prm.s_out >= (g.nx * g.dx - rp.b0_xc) - 2.0 * g.dx)
            throw std::runtime_error("refresh: shell reaches the x wall — "
                                     "lower lambda_deg/shell or grow the box");
        prm.uth_par     = q.uth[0];
        prm.uth_perp_eq = q.uth[1];
        prm.dist        = q.dist;
        prm.lc_rho      = q.lc_rho;
        prm.lc_kappa    = q.lc_kappa;
        if (rp.refresh_precip) {
            if (rp.bnd_x != 1)
                throw std::runtime_error("refresh: precip needs [boundary] "
                                         "x = damping (specular walls are the "
                                         "atmosphere; hybrid is rejected above)");
            prm.precip = 1;
            const double pw = rp.refresh_precip_cells; // strip width in cells
            prm.xw_lo  = pw * g.dx;                 // strips: default 1 cell
            prm.xw_hi  = g.nx * g.dx - pw * g.dx;   // (wall-toucher); wide -> layer
            prm.precip_soft  = rp.refresh_precip_soft;   // hard zero vs soft fold
            prm.precip_numax = rp.refresh_precip_numax;
        }
        acc = DeviceArray<double>(5);
        acc.zero(s);
        on = true;
    }

    // One call per step, AFTER sim.step() (velocity-only, deposit-neutral —
    // this step's J came from the pre-redraw velocities, same ordering as the
    // hybrid-boundary velocity fold). step feeds the per-step RNG seed, so a
    // --resume at step n replays the identical redraw sequence.
    void apply(Particles& parts, const Grid& g, const RunParams& rp,
               long step, cudaStream_t s) {
        if (!on || parts.n == 0) return;
        // splitmix64 step scramble. A LINEAR step mix (seed + G*step) is a
        // trap: rng_uniform hashes t*G + ..., so seed steps of G make marker
        // t at step n+1 replay marker t+1 at step n — the whole bath redraws
        // from ONE index-shifted stream, adjacent same-cell markers get
        // correlated velocities, and the coherent J turns the bath into an
        // antenna (measured: interior Tperp +40% in 3000 steps at f = f0).
        unsigned long long z = (unsigned long long)(step + 1)
                             + 0x9E3779B97F4A7C15ULL;
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 27; z *= 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const unsigned long seed = (unsigned long)rp.rng_seed ^ (unsigned long)z;
        const int blocks = (int)((parts.n + 255) / 256);
        detail::refresh_kernel<<<blocks, 256, 0, s>>>(
            parts.views(), g, rp, prm, seed, acc.data(), (long)parts.n);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Read + reset the interval counters (refresh.csv cadence).
    struct Stats { double redraws, de, fails, precips, de_precip; };
    Stats drain(cudaStream_t s) {
        double h[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        if (on) {
            CUDA_CHECK(cudaMemcpy(h, acc.data(), sizeof(h), cudaMemcpyDeviceToHost));
            acc.zero(s);
        }
        return {h[0], h[1], h[2], h[3], h[4]};
    }
};

} // namespace arc
