#!/usr/bin/env python3
"""Wave propagation maps: By (out-of-plane wave B, the whistler's cleanest
single component) from up to 6 f2d_*.bin snapshots, over the shell design
contours and the active band — the visual gate that waves are excited at
the shell and propagate along the field lines inside the domain.

Usage: plot_wavemap2d.py <outdir> [out.png]
"""
import glob
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1] if len(sys.argv) > 1 else "build/v4r5"
png = sys.argv[2] if len(sys.argv) > 2 else f"{outdir}/wavemap.png"

meta = open(f"{outdir}/meta.txt").read()
m = re.search(r"box ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) "
              r"(\d+) (\d+) ([-\d.e+]+) ([-\d.e+]+)", meta)
x0, x1, z0, z1 = (float(m.group(i)) for i in range(1, 5))
nx, nz = int(m.group(5)), int(m.group(6))
dx, dz = float(m.group(7)), float(m.group(8))
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
prof = re.search(r"profile (\w+)", meta).group(1)
shells = [(n, float(a), float(b), float(e)) for n, a, b, e in
          re.findall(r"shell (\S+) L0=([\d.]+) dL=([\d.]+) edge=([\d.]+)", meta)]
bands = re.findall(r"active_band ([\d.]+) ([\d.]+)", meta)

snaps = sorted(glob.glob(f"{outdir}/f2d_*.bin"))
if not snaps:
    raise SystemExit("no f2d snapshots yet")
pick = snaps if len(snaps) <= 6 else \
    [snaps[i] for i in np.linspace(0, len(snaps) - 1, 6).astype(int)]

ds = 4                                     # downsample for plotting speed
X, Z = np.meshgrid(x0 + (np.arange(nx)[::ds] + 0.5) * dx,
                   z0 + (np.arange(nz)[::ds] + 0.5) * dz)
R2 = X**2 + Z**2
L = R2**1.5 / X**2 if prof == "dipole2d" else R2 / X

ncol = min(3, len(pick))
nrow = (len(pick) + ncol - 1) // ncol
fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow),
                        squeeze=False)
for ax, path in zip(axs.flat, pick):
    step = int(re.search(r"f2d_(\d+)\.bin", path).group(1))
    raw = np.memmap(path, dtype=np.float32, mode="r", shape=(6, nz, nx))
    by = np.array(raw[4, ::ds, ::ds])
    amp = max(np.percentile(np.abs(by), 99.9), 1e-12)
    pc = ax.pcolormesh(X, Z, by, cmap="RdBu_r", vmin=-amp, vmax=amp,
                       shading="auto", rasterized=True)
    for _, L0, dL, _e in shells:
        ax.contour(X, Z, L, levels=[L0 - dL / 2, L0 + dL / 2],
                   colors="k", linewidths=0.6)
    for lo, hi in bands:
        ax.contour(X, Z, L, levels=[float(lo), float(hi)], colors="g",
                   linewidths=0.8, linestyles="--")
    ax.set_title(f"$B_y$  step {step}  t = {step * dt:.0f}"
                 f"  (max {np.abs(by).max():.1e})")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    plt.colorbar(pc, ax=ax)
for ax in axs.flat[len(pick):]:
    ax.axis("off")
fig.suptitle(f"{os.path.basename(outdir)} — wave $B_y$ maps "
             "(black: shell; green dashed: active band)")
fig.tight_layout()
fig.savefig(png, dpi=120)
print(f"wrote {png} ({len(pick)} of {len(snaps)} snapshots)")
