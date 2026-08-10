#!/usr/bin/env python3
"""PLAN_2D stage-2 gate (written before the t400 data lands).

Gates (PLAN_2D.md section 4, stage 2):
  G2a  band growth: LB (0.20-0.45), barrier (0.45-0.55), UB (0.55-0.70)
       band |B| envelopes at the eq probe — LB must grow; report gamma per
       band (log-linear fit over the growing window).
  G2b  UB linear fuel existence: UB-band envelope growing (gamma_UB > 0)
       at ANY probe station (runway argument needs fuel, not eq-local).
  G2c  finite-WNA power: eq-station ky spectrum (ycol FFT) — fraction of
       By,Bz power at |ky| > 0 modes; must exceed the t=0 noise share.
  G2d  eq f_shell: ANALYTIC quadrature of the pool's prodkappa x cone x
       chi_r at b = 1 (same formulas as the loader) -> cold_nc refinement
       number printed (1 - nh - n_r*f_shell_eq).
  G2e  stability: no NaN, WB(t) bounded, wall layers quiet.

Usage: python3 flag_stage2.py <rundir>
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_meta(d):
    m = {}
    for ln in open(os.path.join(d, "meta.txt")):
        parts = ln.split()
        if len(parts) >= 2 and parts[0] not in ("species", "probe_ix"):
            try:
                m[parts[0]] = float(parts[1])
            except ValueError:
                pass
    m["probe_ix"] = [int(l.split()[1]) for l in open(os.path.join(d, "meta.txt"))
                     if l.startswith("probe_ix")]
    m["probe_off"] = [float(l.split()[3]) for l in open(os.path.join(d, "meta.txt"))
                      if l.startswith("probe_ix") and "probe_off" in l]
    return m


# ---- G2d: analytic equatorial f_shell for the flagship pool ----------------
def eq_fshell(theta_par=0.12, uperp_th=0.15, kappa=4.0, cone_b=2.125,
              v1=0.04, v2=0.30, dv=0.02):
    NG = 512
    lo, hi = np.log(1e-4), np.log(50.0)
    G = np.exp(lo + (hi - lo) * (np.arange(NG) + 0.5) / NG)
    wG = G ** (kappa - 0.5) * np.exp(-G)          # Gamma(kappa-1/2) on log grid
    Tpa_m = kappa * theta_par ** 2 / G            # mixture parallel temps
    Tpe = uperp_th ** 2
    # equator b = 1: T1 = Tpe (perp unchanged), cone cut sin2(a) >= 1/cone_b
    #   <=> uperp^2 * cone_b >= u^2  (local = equatorial at b=1)
    # velocity grids: the loader caps |u| at 0.6c (kappa-tail guard), and the
    # chi window needs dP << dv — a Tpa_max-scaled grid puts dP ~ 0.4c and
    # zeroes the window (the bug the first run of this script exposed).
    up = np.linspace(0, 0.6, 600)                 # uperp grid
    upar = np.linspace(-0.6, 0.6, 1201)
    dU = up[1] - up[0]
    dP = upar[1] - upar[0]
    UP, PA = np.meshgrid(up, upar, indexing="ij")
    cone = (UP ** 2 * cone_b >= (UP ** 2 + PA ** 2)).astype(float)
    h = 0.5 * dv
    V = np.abs(PA)                                # b=1: v_par,eq = u_par
    chi = np.ones_like(V)
    chi[V <= v1 - h] = 0.0
    ramp = (V > v1 - h) & (V < v1 + h)
    chi[ramp] = 0.5 * (1 - np.cos(np.pi * (V[ramp] - v1 + h) / dv))
    hi_r = (V > v2 - h) & (V < v2 + h)
    chi[hi_r] *= 0.5 * (1 + np.cos(np.pi * (V[hi_r] - v2 + h) / dv))
    chi[V >= v2 + h] = 0.0
    num = den = 0.0
    for i in range(NG):
        f = (UP * np.exp(-UP ** 2 / (2 * Tpe))
             * np.exp(-PA ** 2 / (2 * Tpa_m[i])) / np.sqrt(Tpa_m[i]))
        den += wG[i] * (f * cone).sum() * dU * dP
        num += wG[i] * (f * cone * chi).sum() * dU * dP
    return num / den


def band_env(z, ts, w1, w2, wce):
    n = len(z)
    F = np.fft.fft(z)
    f = 2 * np.pi * np.fft.fftfreq(n, d=ts) / wce
    F[(np.abs(f) < w1) | (np.abs(f) > w2)] = 0   # both polarities
    a = np.abs(np.fft.ifft(F))
    k = max(1, n // 200)
    return np.convolve(a, np.ones(k) / k, mode="same")


def main(d):
    m = read_meta(d)
    nx, ny = int(m["nx"]), int(m["ny"])
    wce = m["wce"]
    dt = m["dt"]
    npb = int(m["nprobe"])
    ts = m["probe_every"] * dt
    raw = np.fromfile(os.path.join(d, "probe.bin"), np.float32)
    nt = len(raw) // (2 * npb)
    pb = raw[:nt * 2 * npb].reshape(nt, npb, 2)
    t = (np.arange(nt) + 1) * ts * wce            # in 1/Omega_e
    ieq = npb // 2
    offs = m.get("probe_off") or [0.0] * npb

    bands = {"LB 0.20-0.45": (0.20, 0.45), "BAR 0.45-0.55": (0.45, 0.55),
             "UB 0.55-0.70": (0.55, 0.70)}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axs[0, 0]
    print(f"== {d}: nt={nt} t_end={t[-1]:.0f}/We ==")
    gammas = {}
    for name, (w1, w2) in bands.items():
        z = pb[:, ieq, 0] + 1j * pb[:, ieq, 1]
        env = band_env(z, ts, w1, w2, wce)
        ax.semilogy(t, env / wce, label=f"eq {name}")
        # growth fit over the central window (skip first/last 10%)
        s = slice(nt // 10, -nt // 10)
        A = np.polyfit(t[s], np.log(env[s] + 1e-30), 1)
        gammas[name] = A[0]
        print(f"G2a eq {name}: gamma = {A[0]:+.2e} /We   env end {env[-1]/wce:.2e} B0")
    ax.legend(fontsize=8); ax.set_xlabel(r"$t\Omega_e$"); ax.set_ylabel(r"$|B_w|/B_0$")
    ax.set_title("eq band envelopes")

    # G2b: UB growth across stations
    ax = axs[0, 1]
    best = (-1e9, None)
    for p in range(npb):
        z = pb[:, p, 0] + 1j * pb[:, p, 1]
        env = band_env(z, ts, 0.55, 0.70, wce)
        s = slice(nt // 10, -nt // 10)
        gam = np.polyfit(t[s], np.log(env[s] + 1e-30), 1)[0]
        if gam > best[0]:
            best = (gam, p)
        ax.semilogy(t, env / wce, lw=0.8)
    print(f"G2b UB fuel: max gamma_UB = {best[0]:+.2e} /We at probe {best[1]} "
          f"(off {offs[best[1]] if best[1] is not None and offs else float('nan'):+.0f})")
    ax.set_title("UB 0.55-0.70 envelopes, all probes"); ax.set_xlabel(r"$t\Omega_e$")

    # G2c: ky spectrum at eq station, first vs last ycol dump
    fl = sorted(glob.glob(os.path.join(d, "ycol_*.bin")))
    ax = axs[1, 0]
    if fl:
        def kyspec(f):
            a = np.fromfile(f, np.float32).reshape(npb, 2, ny)
            z = a[ieq, 0] + 1j * a[ieq, 1]
            return np.abs(np.fft.fft(z)) ** 2
        P0, P1 = kyspec(fl[0]), kyspec(fl[-1])
        ky = np.fft.fftfreq(ny, d=m["dy"]) * 2 * np.pi
        o = np.argsort(ky)
        ax.semilogy(ky[o], P0[o], label="first dump")
        ax.semilogy(ky[o], P1[o], label="last dump")
        frac0 = 1 - P0[0] / P0.sum()
        frac1 = 1 - P1[0] / P1.sum()
        print(f"G2c finite-ky power fraction (eq): first {frac0:.3f} -> last {frac1:.3f}")
        ax.legend(fontsize=8); ax.set_xlabel(r"$k_y\,c/\omega_{pe}$")
        ax.set_title("eq-station $|B(k_y)|^2$")

    # G2d: analytic eq f_shell -> cold_nc refinement
    fsh = eq_fshell()
    ncref = 1.0 - 0.010 - 0.012 * fsh
    print(f"G2d eq f_shell (analytic, loader formulas) = {fsh:.4f} "
          f"-> cold_nc refinement = {ncref:.4f} (deck 0.9810)")

    # G2e: energy trace + wall quiet
    import csv
    rows = list(csv.reader(open(os.path.join(d, "energy.csv"))))[1:]
    te = [float(r[1]) * wce for r in rows]
    wb = [float(r[3]) for r in rows]
    axs[1, 1].semilogy(te, wb, label="WB")
    axs[1, 1].semilogy(te, [float(r[2]) for r in rows], label="WE")
    axs[1, 1].legend(); axs[1, 1].set_title("field energy"); axs[1, 1].set_xlabel(r"$t\Omega_e$")
    bl = sorted(glob.glob(os.path.join(d, "bline_*.bin")))
    b = np.fromfile(bl[-1], np.float32)
    am = np.abs(b[:nx] + 1j * b[nx:2*nx])
    nd = 240
    print(f"G2e wall/interior |B| (y-mid, last): wall {am[:nd].mean():.2e} "
          f"interior {am[nx//2-500:nx//2+500].mean():.2e}  NaN: {np.isnan(b).any()}")
    fig.savefig(f"{os.path.basename(d.rstrip('/'))}_stage2.png", dpi=120)
    print(f"fig -> {os.path.basename(d.rstrip('/'))}_stage2.png")


if __name__ == "__main__":
    main(sys.argv[1])
