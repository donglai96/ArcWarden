#!/usr/bin/env python3
"""Phase S gate (PLAN_2D v3 §S, 2026-08-11). FROZEN before the three arms'
data is judged. Hard gates + exit code; everything else is reported.

Per arm (s_chorus / s_pool / s_both), at the eq probe unless noted:
  measurements:
    - decimated-probe spectrogram (x20, nwin 512) saved per arm
    - band envelopes + log-linear gamma fits over t in [200, 900]:
      LB 0.20-0.45, BAR 0.45-0.55, UB 0.55-0.70 (local=eq units, wce meta)
    - W10 occupancy on the decimated eq probe, [0, 1000] (1D-calibrated
      metric, 2D recalibration caveat printed with the number)
    - w-kx map from bline y-avg (windows [200,600] / [600,1000]) with the
      parallel cold-whistler w(kx) overlay (dispersion membership, visual)
    - w-ky map at the eq station from ycol (late window)
  hard gates (pre-registered):
    G-S1 s_chorus: gamma_LB > +1e-3 /We (the source must actually drive LB)
    G-S2 s_pool:   UB QUIET — gamma_UB < +1e-3 /We AND
                   P_UB[700,1000] < 3 x same measure in qa_pool1 (numerics
                   baseline). If s_pool drives 0.5-0.7, combined-run UB/gap
                   cannot be attributed to resonant processing.
    G-S3 s_both:   not born-flooded — W10 < 0.20 over [0,1000]
    (cross-arm causal readout printed; no gate on it)

Usage: python3 s_gate.py <chorus_dir> <pool_dir> <both_dir> <quiet_baseline>
Exit 0 = all hard gates PASS, 1 otherwise.
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec, w10_occupancy


def read_meta(d):
    m = {}
    for ln in open(os.path.join(d, "meta.txt")):
        p = ln.split()
        if len(p) >= 2 and p[0] not in ("species", "probe_ix"):
            try:
                m[p[0]] = float(p[1])
            except ValueError:
                pass
    return m


def eq_probe(d, m):
    npb = int(m["nprobe"])
    raw = np.fromfile(os.path.join(d, "probe.bin"), np.float32)
    nt = len(raw) // (2 * npb)
    pb = raw[:nt * 2 * npb].reshape(nt, npb, 2)
    ieq = npb // 2
    z = (pb[:, ieq, 0] + 1j * pb[:, ieq, 1])[::20]
    ts = m["probe_every"] * m["dt"] * 20
    t = (np.arange(len(z)) + 1) * ts
    return z, t, ts


def band_env(z, ts, w1, w2, wce):
    F = np.fft.fft(z)
    f = 2 * np.pi * np.fft.fftfreq(len(z), d=ts) / wce
    F[(np.abs(f) < w1) | (np.abs(f) > w2)] = 0
    a = np.abs(np.fft.ifft(F))
    k = max(1, len(z) // 100)
    return np.convolve(a, np.ones(k) / k, mode="same")


def ub_power(d, m, t1=700.0, t2=1000.0):
    z, t, ts = eq_probe(d, m)
    wce = m["wce"]
    tt = t * wce
    e = band_env(z, ts, 0.55, 0.70, wce)
    return float((e[(tt > t1) & (tt <= t2)] ** 2).mean())


def gamma_fit(env, tOe, t1=200.0, t2=900.0):
    m = (tOe > t1) & (tOe < t2)
    return np.polyfit(tOe[m], np.log(env[m] + 1e-30), 1)[0]


def wkx_map(d, m, ax, t1, t2):
    nx = int(m["nx"])
    wce, dt = m["wce"], m["dt"]
    dt_dump = int(m["bline_every"]) * dt
    fl = sorted(glob.glob(os.path.join(d, "bline_*.bin")))
    tt = (np.arange(len(fl)) + 1) * dt_dump * wce
    sel = [f for f, tv in zip(fl, tt) if t1 < tv <= t2]
    A = np.empty((len(sel), nx), np.complex64)
    for i, f in enumerate(sel):
        b = np.fromfile(f, np.float32)
        A[i] = b[3 * nx:4 * nx] + 1j * b[4 * nx:5 * nx]      # y-avg By,Bz
    A *= np.hanning(len(sel))[:, None]
    F = np.fft.fft2(A)
    w = 2 * np.pi * np.fft.fftfreq(len(sel), d=dt_dump) / wce
    kx = np.fft.fftfreq(nx, d=m["dx"]) * 2 * np.pi
    ow, ok = np.argsort(w), np.argsort(kx)
    ax.pcolormesh(kx[ok], w[ow], np.log10(np.abs(F[ow][:, ok]) ** 2 + 1e-30),
                  cmap="turbo", shading="auto", rasterized=True)
    # parallel cold whistler overlay (eq: wpe^2 = 1, We = wce)
    kk = np.linspace(0.01, kx.max(), 200)
    wpar = []
    for k in kk:
        lo, hi = 1e-6, wce * (1 - 1e-6)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mid ** 2 + mid / (wce - mid) * 1.0 - k ** 2 > 0 * mid:
                hi = mid
            else:
                lo = mid
        wpar.append(0.5 * (lo + hi) / wce)
    ax.plot(kk, wpar, "w--", lw=1)
    ax.plot(-kk, wpar, "w--", lw=1)
    ax.set_ylim(0, 1.0); ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel(r"$k_x$"); ax.set_ylabel(r"$\omega/\Omega_e$")


def arm(d, name, axs_row):
    m = read_meta(d)
    wce = m["wce"]
    z, t, ts = eq_probe(d, m)
    tOe = t * wce
    P, fs, tt = spec(z, t, wce, 512)
    axs_row[0].pcolormesh(tt, fs, np.log10(P.T + 1e-30), cmap="turbo",
                          vmin=np.log10(P.max()) - 5, vmax=np.log10(P.max()),
                          shading="auto", rasterized=True)
    axs_row[0].axhline(0.5, color="w", ls="--", lw=1)
    axs_row[0].set_ylim(0.05, 0.9)
    axs_row[0].set_title(f"{name} eq spectrogram", fontsize=9)
    w10, _ = w10_occupancy(P, fs, tt, 0, 1000)
    gam = {}
    for bn, (w1, w2) in (("LB", (0.20, 0.45)), ("BAR", (0.45, 0.55)),
                         ("UB", (0.55, 0.70))):
        e = band_env(z, ts, w1, w2, wce)
        gam[bn] = gamma_fit(e, tOe)
        axs_row[1].semilogy(tOe, e / wce, label=f"{bn} g={gam[bn]:+.1e}")
    axs_row[1].legend(fontsize=7); axs_row[1].set_title(f"{name} band env", fontsize=9)
    wkx_map(d, m, axs_row[2], 600.0, 1000.0)
    axs_row[2].set_title(f"{name} w-kx [600,1000] (y-avg)", fontsize=9)
    print(f"[{name}] W10[0,1000] = {w10:.3f} (1D-calibrated metric)  "
          f"gamma LB {gam['LB']:+.2e}  BAR {gam['BAR']:+.2e}  UB {gam['UB']:+.2e}")
    return m, gam, w10


def main(dc, dp, db, dq):
    fails = []
    fig, axs = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    mc, gc, w10c = arm(dc, "chorus-only", axs[0])
    mp, gp, w10p = arm(dp, "pool-only", axs[1])
    mb, gb, w10b = arm(db, "chorus+pool", axs[2])

    if gc["LB"] <= 1e-3:
        fails.append(f"G-S1 chorus gamma_LB {gc['LB']:+.2e} <= 1e-3")
    print(f"G-S1 chorus-only drives LB: gamma_LB = {gc['LB']:+.2e} -> "
          f"{'PASS' if gc['LB'] > 1e-3 else 'FAIL'}")

    mq = read_meta(dq)
    pub_pool = ub_power(dp, mp)
    pub_base = ub_power(dq, mq)
    quiet = gp["UB"] < 1e-3 and pub_pool < 3 * pub_base
    print(f"G-S2 pool-only UB quiet: gamma_UB = {gp['UB']:+.2e}, "
          f"P_UB[700,1000] = {pub_pool:.3e} vs baseline {pub_base:.3e} "
          f"(x{pub_pool / pub_base:.2f}) -> {'PASS' if quiet else 'FAIL'}")
    if not quiet:
        fails.append("G-S2 pool-only drives UB")

    if w10b >= 0.20:
        fails.append(f"G-S3 combined W10 {w10b:.3f} >= 0.20")
    print(f"G-S3 combined not born-flooded: W10 = {w10b:.3f} -> "
          f"{'PASS' if w10b < 0.20 else 'FAIL'}")

    print("cross-arm readout: UB(combined)/UB(chorus-only) gamma "
          f"{gb['UB']:+.2e} vs {gc['UB']:+.2e}; BAR(combined) {gb['BAR']:+.2e}")
    fig.savefig("s_gate_arms.png", dpi=120)
    print("fig -> s_gate_arms.png")
    if fails:
        print("PHASE S GATE: FAIL —", "; ".join(fails))
        return 1
    print("PHASE S GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
