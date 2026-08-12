#!/usr/bin/env python3
"""UB Step 2 (PLAN_UB_1D v2): mine giant_x4_atmo40 for upper-band signatures.

The strongest stall-train dataset we own (elements 0.2 -> 0.45-0.5,
amp ~7e-3 B0, t20000, full-f). Questions, frozen before looking:
  S2a  band powers vs time at eq / +-5deg: P_LB(0.20-0.45),
       P_BAR(0.45-0.55), P_UB(0.55-0.90); UB/LB ratio + run maximum
       (the "P(>0.5) upper limit").
  S2b  lower-band-cascade search: b^2 bicoherence on the f1=f2 diagonal
       (coupling (w1,w1)->2w1) over the element train window vs the
       pre-burst noise window. Chen 2017 JGR saw the cascade in parallel
       1D PIC at comparable amplitudes — if it operates here it MUST
       show phase locking; absence bounds the cascade efficiency.
       Also P_2w vs P_w scaling over element bursts (cascade: ~quadratic).

Usage: python3 ub_mine_x4.py [rundir=giant_x4_atmo40]
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lurepro_source_verdict import probe_pair


def band_power(z, ts, wce, w1, w2, nseg=4096):
    """time series of band power via STFT segments (hop = nseg/2)."""
    hop = nseg // 2
    n = (len(z) - nseg) // hop + 1
    t = np.empty(n)
    P = np.empty(n)
    f = 2 * np.pi * np.fft.fftfreq(nseg, d=ts) / wce
    m = (f >= w1) & (f < w2)
    win = np.hanning(nseg)
    for i in range(n):
        seg = z[i * hop:i * hop + nseg] * win
        X = np.fft.fft(seg)
        P[i] = (np.abs(X[m]) ** 2).sum() / nseg
        t[i] = (i * hop + nseg / 2) * ts * wce
    return t, P


def bicoherence_diag(z, ts, wce, nseg=2048):
    """b^2(f, f) for the (f,f)->2f coupling, Hinich normalization."""
    hop = nseg // 2
    nwin = (len(z) - nseg) // hop + 1
    win = np.hanning(nseg)
    f = 2 * np.pi * np.fft.fftfreq(nseg, d=ts) / wce
    nf = nseg // 4                      # need 2f in range
    num = np.zeros(nf, complex)
    d1 = np.zeros(nf)
    d2 = np.zeros(nf)
    for i in range(nwin):
        X = np.fft.fft(z[i * hop:i * hop + nseg] * win)
        for j in range(1, nf):
            t1 = X[j] * X[j]
            t2 = X[2 * j]
            num[j] += t1 * np.conj(t2)
            d1[j] += np.abs(t1) ** 2
            d2[j] += np.abs(t2) ** 2
    b2 = np.abs(num) ** 2 / (d1 * d2 + 1e-300)
    return f[:nf], b2, nwin


def main(d="giant_x4_atmo40"):
    m0, t, by, bz, *_ = probe_pair(d, 0.0)
    wce = m0["wce"]
    ts = t[1] - t[0]
    print(f"== {d}: eq probe, {len(t)} samples, ts = {ts * wce:.3f}/We ==")

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    # S2a ----------------------------------------------------------------
    stations = [(0.0, "eq")]
    for off, lab in (( 291.0, "+5deg"), (-291.0, "-5deg")):
        stations.append((off, lab))
    ax = axs[0, 0]
    ub_max = {}
    for off, lab in stations:
        mm, tp, byp, bzp, *_ = probe_pair(d, off)
        z = byp + 1j * bzp
        tt, Plb = band_power(z, ts, wce, 0.20, 0.45)
        _, Pbar = band_power(z, ts, wce, 0.45, 0.55)
        _, Pub = band_power(z, ts, wce, 0.55, 0.90)
        if lab == "eq":
            ax.semilogy(tt, Plb, label="LB 0.20-0.45")
            ax.semilogy(tt, Pbar, label="BAR 0.45-0.55")
            ax.semilogy(tt, Pub, label="UB 0.55-0.90")
        burst = tt > 2500
        ub_max[lab] = (Pub[burst].max(), (Pub[burst] / Plb[burst]).max(),
                       np.median(Pub[burst] / Plb[burst]))
        print(f"S2a [{lab}] burst-window: max P_UB = {ub_max[lab][0]:.3e}  "
              f"max UB/LB = {ub_max[lab][1]:.3e}  median UB/LB = {ub_max[lab][2]:.3e}")
    ax.legend(fontsize=8); ax.set_xlabel(r"$t\Omega_e$")
    ax.set_title("eq band powers")

    # S2b ----------------------------------------------------------------
    z = by + 1j * bz
    iburst = t * wce > 2500
    inoise = t * wce < 1800
    f1, b2_burst, nw1 = bicoherence_diag(z[iburst], ts, wce)
    f2, b2_noise, nw2 = bicoherence_diag(z[inoise], ts, wce)
    sig = 1.0 / nw1                     # bicoherence noise floor ~ 1/nwin
    ax = axs[0, 1]
    ax.plot(f1, b2_burst, label=f"element train (n={nw1})")
    ax.plot(f2, b2_noise, alpha=0.6, label=f"pre-burst noise (n={nw2})")
    ax.axhline(3 * sig, color="r", ls=":", lw=1, label="3x floor")
    ax.set_xlim(0.1, 0.5); ax.set_xlabel(r"$f/\Omega_e$ (diagonal $f_1=f_2$)")
    ax.set_ylabel(r"$b^2(f,f)$"); ax.legend(fontsize=8)
    ax.set_title("cascade phase-locking (f,f)->2f")
    band = (f1 > 0.20) & (f1 < 0.45)
    jmax = np.argmax(b2_burst * band)
    print(f"S2b bicoherence: max b2 in LB diagonal = {b2_burst[jmax]:.4f} at "
          f"f = {f1[jmax]:.3f} (floor {sig:.4f}, 3x floor {3*sig:.4f}) -> "
          f"{'LOCKED (cascade active)' if b2_burst[jmax] > 3*sig else 'no significant locking'}")

    # scaling P_2w vs P_w over time (eq) --------------------------------
    tt, Pw = band_power(z, ts, wce, 0.25, 0.40)
    _, P2w = band_power(z, ts, wce, 0.50, 0.80)
    w = tt > 2500
    ax = axs[1, 0]
    ax.loglog(Pw[w], P2w[w], ".", ms=3)
    lg = np.polyfit(np.log(Pw[w]), np.log(P2w[w]), 1)
    xx = np.linspace(np.log(Pw[w].min()), np.log(Pw[w].max()), 10)
    ax.loglog(np.exp(xx), np.exp(np.polyval(lg, xx)), "r-",
              label=f"slope {lg[0]:.2f} (cascade ~ 2)")
    ax.set_xlabel("P(0.25-0.40)"); ax.set_ylabel("P(0.50-0.80)")
    ax.legend(fontsize=9); ax.set_title("2w vs w power scaling (eq, burst)")
    print(f"S2b scaling: log-log slope P_2w vs P_w = {lg[0]:.2f} "
          f"(cascade expectation ~2, independent bands ~uncorrelated)")

    # eq spectrum, burst average ----------------------------------------
    ax = axs[1, 1]
    nseg = 4096
    win = np.hanning(nseg)
    zz = z[iburst]
    hop = nseg // 2
    nwin = (len(zz) - nseg) // hop + 1
    S = np.zeros(nseg)
    for i in range(nwin):
        S += np.abs(np.fft.fft(zz[i * hop:i * hop + nseg] * win)) ** 2
    f = 2 * np.pi * np.fft.fftfreq(nseg, d=ts) / wce
    o = np.argsort(f)
    ax.semilogy(f[o], S[o] / nwin)
    ax.set_xlim(-0.2, 1.0); ax.axvline(0.5, color="gray", ls="--")
    ax.set_xlabel(r"$f/\Omega_e$"); ax.set_title("eq mean spectrum (burst)")
    fig.savefig(f"{os.path.basename(d)}_ubmine.png", dpi=120)
    print(f"fig -> {os.path.basename(d)}_ubmine.png")


if __name__ == "__main__":
    main(*sys.argv[1:2])
