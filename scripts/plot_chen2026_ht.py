#!/usr/bin/env python3
"""δB(h,t) map (Chen et al. PoP 2026 Fig. 3a analog) from chirp2d bline dumps.

Usage: plot_chen2026_ht.py <rundir> <out.png> [tmax_Oe]
h = arc distance from the equator in c/ωpe (= their V_Ae0/Ω_e0), t in 1/Ω_e0.
A group-velocity guide line (their vg = 0.65 V_Ae0 at ω = 0.22 Ω_e0) is drawn
from the equator.
"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_meta(d):
    meta = {}
    for line in open(f"{d}/meta.txt"):
        p = line.split()
        if len(p) == 2:
            meta.setdefault(p[0], []).append(float(p[1]))
    return {k: (v[0] if len(v) == 1 else v) for k, v in meta.items()}


def main():
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "chen2026_ht.png"
    m = read_meta(d)
    nx, dx, wce = int(m["nx"]), m["dx"], m["wce"]
    dt_line = m["bline_every"] * m["dt"] * wce          # Ω_e0 units
    files = sorted(glob.glob(f"{d}/bline_*.bin"))
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 1e30
    nkeep = min(len(files), int(tmax / dt_line) + 1)
    files = files[:nkeep]

    xstride = 4
    h = (np.arange(nx) * dx - m["b0_xc"])[::xstride]
    bt = np.empty((len(files), len(h)), dtype=np.float32)
    for i, f in enumerate(files):
        a = np.fromfile(f, dtype=np.float32, count=2 * nx)
        bt[i] = np.sqrt(a[:nx] ** 2 + a[nx:] ** 2)[::xstride] / wce
    t = (np.arange(len(files)) + 1) * dt_line

    fig, ax = plt.subplots(figsize=(7.5, 8))
    pc = ax.pcolormesh(h, t, np.clip(bt, 1e-6, None),
                       norm=LogNorm(vmin=1e-4, vmax=3e-2),
                       cmap="jet", shading="auto", rasterized=True)
    # group-velocity guide: their vg = 0.65 V_Ae0 (ω = 0.22 Ω_e0);
    # in these axes dh/dt = 0.65 c/ωpe per 1/Ω_e0... V_Ae0/Ω_e0 per Ω_e0^-1
    for t0 in (500.0,):
        tt = np.linspace(t0, t0 + 900, 50)
        ax.plot(0.65 * (tt - t0), tt, "w--", lw=1.2)
        ax.text(0.65 * 800, t0 + 860, r"$v_g=0.65\,V_{Ae0}$", color="w", fontsize=9)
    for hp in (-116.4, 116.4):
        ax.axvline(hp, color="w", ls=":", lw=0.8, alpha=0.7)
    ax.axvline(0, color="yellow", ls=":", lw=0.8, alpha=0.9)
    ax.set_xlabel(r"$h\ (V_{Ae0}/\Omega_{e0})$")
    ax.set_ylabel(r"$\Omega_{e0}t$")
    ax.set_title(f"{d}: " + r"$\delta B/B_{e0}$ (h,t)   —   Chen PoP 2026 Fig. 3a analog")
    fig.colorbar(pc, ax=ax, label=r"$\delta B/B_{e0}$")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
