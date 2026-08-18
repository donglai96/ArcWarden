#!/usr/bin/env python3
"""Initial particle density check: dens_<species>.bin (dense [nz][nx]
float32, written by warden2d at build time) rendered over the analytic
background geometry — designed shell L0 ± dL/2 (solid) and the Gaussian
edge extent (dotted). The visual gate: the loaded density must sit exactly
between its designed L contours.

Usage: plot_density2d.py <outdir> [out.png]
Geometry is read from <outdir>/meta.txt (box line + shell lines).
"""
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1] if len(sys.argv) > 1 else "build/v4r5"
png = sys.argv[2] if len(sys.argv) > 2 else f"{outdir}/density_init.png"

meta = open(f"{outdir}/meta.txt").read()


def grab(pat):
    m = re.search(pat, meta)
    if not m:
        raise SystemExit(f"meta.txt missing: {pat}")
    return [float(g) for g in m.groups()]


x0, x1, z0, z1, nx, nz, dx, dz = grab(
    r"box ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) "
    r"(\d+) (\d+) ([-\d.e+]+) ([-\d.e+]+)")
nx, nz = int(nx), int(nz)
prof = re.search(r"profile (\w+)", meta).group(1)
shells = re.findall(r"shell (\S+) L0=([\d.]+) dL=([\d.]+) edge=([\d.]+)", meta)
bands = re.findall(r"active_band ([\d.]+) ([\d.]+)", meta)

X, Z = np.meshgrid(x0 + (np.arange(nx) + 0.5) * dx,
                   z0 + (np.arange(nz) + 0.5) * dz)
R2 = X**2 + Z**2
L = R2**1.5 / X**2 if prof == "dipole2d" else R2 / X   # r³/x² vs r²/x

fig, ax = plt.subplots(figsize=(11, 7))
first = True
for name, L0, dL, edge in shells:
    L0, dL, edge = float(L0), float(dL), float(edge)
    dens = np.fromfile(f"{outdir}/dens_{name}.bin", dtype=np.float32)
    dens = dens.reshape(nz, nx)
    if first:
        v = dens / max(dens.max(), 1e-30)
        pc = ax.pcolormesh(X, Z, v, cmap="inferno", vmin=0, vmax=1,
                           shading="auto", rasterized=True)
        plt.colorbar(pc, ax=ax, label=f"n/n_max ({name})")
        first = False
    ax.contour(X, Z, L, levels=[L0 - dL / 2, L0 + dL / 2],
               colors="cyan", linewidths=1.2)
    ax.contour(X, Z, L, levels=[L0 - dL / 2 - 3 * edge, L0 + dL / 2 + 3 * edge],
               colors="cyan", linewidths=0.7, linestyles=":")
    # loaded barycentre-in-L vs design (the quantitative line of the gate)
    wsum = dens.sum()
    Lbar = (dens * L).sum() / wsum
    inside = dens[(L > L0 - dL / 2 - 3 * edge) & (L < L0 + dL / 2 + 3 * edge)].sum()
    print(f"{name}: <L> = {Lbar:.1f} (design {L0}), "
          f"weight inside 3σ envelope = {100 * inside / wsum:.2f}%")
for lo, hi in bands:
    ax.contour(X, Z, L, levels=[float(lo), float(hi)], colors="w",
               linewidths=0.8, linestyles="--")
ax.set_xlabel(r"$x$ [$c/\omega_{pe}$]")
ax.set_ylabel(r"$z$ [$c/\omega_{pe}$]")
ax.set_title(f"initial density vs designed shell — {prof}"
             " (cyan: flat top; dotted: +3σ edge; white dashed: active band)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(png, dpi=140)
print(f"wrote {png}")
