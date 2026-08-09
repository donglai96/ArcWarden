#!/usr/bin/env python3
"""LRE scan — frequency-resolved energy transfer maps P_w(s,t)
(PLAN_LRE_SCAN §6): ridge/band complex-analytic

    P_w(s,t) = 1/2 Re[ J~_w . E~_w* ]   (hot J from jline)

per omega-slice, plus band Poynting flux F_w = Re[E~ x B~*]_x/2 and EM
energy W_w.  Sign convention (pre-registered): P_w > 0 electrons ABSORB
wave energy (damping); P_w < 0 electrons drive the wave (growth);
P_w -> 0 coherent resonant current lost.
Balance note: W/F are EM-only (cold-fluid sloshing energy not included);
the science discriminator is the SIGN/LOCATION structure of P_w and the
transmitted-flux drop across the 0.5 region, not closure to zero.

Spectrum-bite expectation : incoming F_w at 0.46-0.56, then P_w>0 layer,
                            transmitted action visibly reduced.
Generation-barrier        : P_w<0 source below 0.5 fading to zero near
                            0.5, NO P_w>0 layer, high-f never forms.
Mixed                     : drive fades -> chirp slows -> residence grows
                            -> P_w>0 eats the remainder.

Usage: python3 lrescan_pw.py <dir> [w1,w2 slices default 0.36-0.72/0.06]
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import read_meta


def stack(d, pref, ncomp, nx):
    fl = sorted(glob.glob(os.path.join(d, f"{pref}_*.bin")))
    out = [np.empty((len(fl), nx), np.float32) for _ in range(ncomp)]
    for i, f in enumerate(fl):
        a = np.fromfile(f, dtype=np.float32)
        for c in range(ncomp):
            out[c][i] = a[c * nx:(c + 1) * nx]
    return out


def band_analytic(a, dt_dump, w1, w2):
    """analytic signal per x-column restricted to [w1,w2] (rad/time)."""
    n = a.shape[0]
    F = np.fft.fft(a, axis=0)
    f = 2 * np.pi * np.fft.fftfreq(n, d=dt_dump)
    F[(f < w1) | (f > w2)] = 0
    return 2 * np.fft.ifft(F, axis=0)     # analytic (positive-freq) x2


def main(d, slices=None):
    m = read_meta(d)
    nx, wce = int(m["nx"]), m["wce"]
    dxu = m["dx"]
    xc = m["b0_xc"]
    lre = m.get("b0_lre", 1330.504)
    dt_dump = m["bline_every"] * m["dt"]
    by, bz = stack(d, "bline", 2, nx)
    jx, jy, jz = stack(d, "jline", 3, nx)
    ex, ey, ez = stack(d, "eline", 3, nx)
    nt = by.shape[0]
    tOe = (np.arange(nt) + 1) * dt_dump * wce
    s = (np.arange(nx) * dxu - xc)
    if slices is None:
        slices = [(w0, w0 + 0.06) for w0 in np.arange(0.36, 0.70, 0.06)]
    nsl = len(slices)
    fig, axs = plt.subplots(nsl, 2, figsize=(13, 2.1 * nsl), sharex=True,
                            sharey=True, constrained_layout=True)
    print(f"{d}: lre={lre:.0f}  slices={['%.2f-%.2f' % s_ for s_ in slices]}")
    for i, (a1, a2) in enumerate(slices):
        w1, w2 = a1 * wce, a2 * wce
        Ey = band_analytic(ey, dt_dump, w1, w2)
        Ez = band_analytic(ez, dt_dump, w1, w2)
        Jy = band_analytic(jy, dt_dump, w1, w2)
        Jz = band_analytic(jz, dt_dump, w1, w2)
        By = band_analytic(by, dt_dump, w1, w2)
        Bz = band_analytic(bz, dt_dump, w1, w2)
        P = 0.5 * np.real(Jy * np.conj(Ey) + Jz * np.conj(Ez))
        F = 0.5 * np.real(Ey * np.conj(Bz) - Ez * np.conj(By))
        del Ey, Ez, Jy, Jz, By, Bz
        for k, (A, name, cmap) in enumerate(
                ((P, "P_w (>0 absorb)", "RdBu_r"),
                 (F, "F_w Poynting_x", "PuOr_r"))):
            ax = axs[i, k]
            v = np.percentile(np.abs(A), 99.5) + 1e-30
            ax.pcolormesh(tOe, s, A.T, cmap=cmap, vmin=-v, vmax=v,
                          shading="auto", rasterized=True)
            ax.set_ylabel(f"[{a1:.2f},{a2:.2f}]\ns (c/wpe)", fontsize=8)
            if i == 0:
                ax.set_title(name)
        # integrated numbers for the verdict table
        mid = np.abs(s) < 0.25 * lre    # source region
        print(f"  [{a1:.2f},{a2:.2f}]: sum P dt dx (src) = "
              f"{P[:, mid].sum() * dt_dump * dxu:+.3e}   "
              f"peak|P| {np.abs(P).max():.2e}")
        del P, F
    for ax in axs[-1]:
        ax.set_xlabel(r"$t\,\Omega_e$")
    tag = os.path.basename(d.rstrip("/"))
    fig.savefig(f"{tag}_pw.png", dpi=110)
    print(f"fig -> {tag}_pw.png")


if __name__ == "__main__":
    main(sys.argv[1])
