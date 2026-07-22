#!/usr/bin/env python3
"""Giant-run summary figure: waveform / envelope / spectrogram [/ h-t map].

All panels share the exact same axis width: colorbars live in appended axes
(make_axes_locatable), and panels without one get an invisible placeholder.
The optional (d) panel is the deltaB(h,t) map from the bline dumps, drawn with
TIME HORIZONTAL so it shares the x-axis with (a)-(c); probe latitudes are the
white dotted lines, the equator the yellow one.

Usage: plot_chen2026_giant.py <rundir> <out.png>
           [--probe=116.4] [--ht] [--nwin=1024] [--title=...]
"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import convolve as ndconv

sys.path.insert(0, sys.path[0])
from plot_chen2026_fig1 import load_probe, read_meta, stft


def load_ht(d, xstride=4):
    m = read_meta(d)
    nx, wce = int(m["nx"]), m["wce"]
    dt_line = m["bline_every"] * m["dt"] * wce
    import os
    # drop a trailing partial dump (run killed mid-write)
    files = [f for f in sorted(glob.glob(f"{d}/bline_*.bin"))
             if os.path.getsize(f) >= 8 * nx]
    h = (np.arange(nx) * m["dx"] - m["b0_xc"])[::xstride]
    bt = np.empty((len(files), len(h)), dtype=np.float32)
    for i, f in enumerate(files):
        a = np.fromfile(f, dtype=np.float32, count=2 * nx)
        bt[i] = np.sqrt(a[:nx] ** 2 + a[nx:] ** 2)[::xstride] / wce
    t = (np.arange(len(files)) + 1) * dt_line
    return h, t, bt


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1")
                for a in sys.argv[1:] if a.startswith("--"))
    d = args[0]
    out = args[1] if len(args) > 1 else "chen2026_giant.png"
    poff = float(opts.get("probe", 116.4))
    nwin = int(opts.get("nwin", 1024))
    want_ht = "ht" in opts

    m, t_wpe, by, bz = load_probe(d, poff)
    wce = m["wce"]
    t_oe = t_wpe * wce
    byn, bzn = by / wce, bz / wce
    env = np.abs(byn + 1j * bzn)
    tmax = t_oe[-1]
    lat_deg = round(poff / 23.28)          # 116.4 c/wpe per 5 deg (their pin)

    npan = 4 if want_ht else 3
    heights = [1, 1, 1.8] + ([1.8] if want_ht else [])
    fig, axes = plt.subplots(npan, 1, figsize=(11, 10 if npan == 3 else 13),
                             gridspec_kw=dict(height_ratios=heights))
    caxes = [make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.08)
             for ax in axes]

    title = opts.get("title",
                     f"{d.rstrip('/').split('/')[-1]}: giant full-f, "
                     f"{int(m['nmarkers']/1e6)}M markers (ppc = {int(m['ppc'])}) "
                     f"— probe $\\lambda=+{lat_deg}^\\circ$")

    ax = axes[0]
    ax.plot(t_oe, byn, lw=0.2, color="tab:blue")
    a0 = 1.15 * np.max(np.abs(byn))
    ax.set_ylim(-a0, a0)
    ax.set_xlim(0, tmax)
    ax.set_ylabel(r"(a)  $\delta B_\varphi/B_{e0}$")
    ax.set_title(title, fontsize=11)
    caxes[0].axis("off")

    ax = axes[1]
    k = 256
    envs = np.convolve(env, np.ones(k) / k, mode="same")
    ax.semilogy(t_oe, envs, lw=0.6, color="k")
    ipk = int(np.argmax(envs))
    ax.axvline(t_oe[ipk], color="g", ls="--", lw=1)
    ax.text(t_oe[ipk], 2e-2, r" pk=%.4f @ %d" % (envs[ipk], t_oe[ipk]),
            color="g", fontsize=9)
    ax.set_ylim(1e-5, 1e-1)
    ax.set_xlim(0, tmax)
    ax.set_ylabel(r"(b)  $\delta B/B_{e0}$")
    caxes[1].axis("off")

    ax = axes[2]
    hop = nwin // 32
    spec = stft(byn + 1j * bzn, nwin, hop)
    dt_s = t_wpe[1] - t_wpe[0]
    freqs = np.fft.fftfreq(nwin, d=dt_s) * 2 * np.pi / wce
    tt = (t_wpe[0] + (np.arange(spec.shape[0]) * hop + nwin / 2) * dt_s) * wce
    sel = (freqs >= 0) & (freqs <= 0.85)
    pw = ndconv(np.abs(spec[:, sel]).T, np.ones((3, 3)) / 9.0, mode="nearest")
    pc = ax.pcolormesh(tt, freqs[sel], np.clip(pw, 1e-6, None),
                       norm=LogNorm(vmin=1e-4, vmax=2e-2),
                       cmap="jet", shading="auto", rasterized=True)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, tmax)
    ax.set_ylabel(r"(c)  $\omega/\Omega_{e0}$")
    if not want_ht:
        ax.set_xlabel(r"$\Omega_{e0} t$")
    fig.colorbar(pc, cax=caxes[2], label=r"$\delta B/B_{e0}$")

    if want_ht:
        h, t, bt = load_ht(d)
        ax = axes[3]
        pc = ax.pcolormesh(t, h, np.clip(bt, 1e-6, None).T,
                           norm=LogNorm(vmin=1e-4, vmax=3e-2),
                           cmap="jet", shading="auto", rasterized=True)
        for hp in (-116.4, 116.4):
            ax.axhline(hp, color="w", ls=":", lw=0.8, alpha=0.7)
        ax.axhline(0, color="yellow", ls=":", lw=0.8, alpha=0.9)
        ax.text(tmax * 0.99, 116.4, r" probes $\lambda=\pm5^\circ$ ", color="w",
                fontsize=8, ha="right", va="bottom")
        ax.text(tmax * 0.99, 0, " equator ", color="yellow", fontsize=8,
                ha="right", va="bottom")
        ax.set_xlim(0, tmax)
        ax.set_ylabel(r"(d)  $h\ (V_{Ae0}/\Omega_{e0})$")
        ax.set_xlabel(r"$\Omega_{e0} t$")
        fig.colorbar(pc, cax=caxes[3], label=r"$\delta B/B_{e0}$")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
