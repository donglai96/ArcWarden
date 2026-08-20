#!/usr/bin/env python3
"""High-pass wave map with zooms (user ritual 2026-08-19): every wave-field
figure MUST outline the hot-particle initial shell — flat-top (solid) and
±3σ envelope (dashed) — on every panel, plus raw-By reference panels.

Usage: wavemap_hipass.py <outdir> <deck.ini> [frame_index=-2]
"""
import glob
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

d, deck = sys.argv[1], sys.argv[2]
fidx = int(sys.argv[3]) if len(sys.argv) > 3 else -2

meta = open(f"{d}/meta.txt").read()
m = re.search(r"box ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) (\d+) (\d+)", meta)
x0, x1, z0, z1 = (float(m.group(i)) for i in range(1, 5))
nx, nz = int(m.group(5)), int(m.group(6))
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
B0eq = float(re.search(r"B0eq ([\d.]+)", meta).group(1))
sh = re.search(r"shell \S+ L0=([\d.]+) dL=([\d.]+) edge=([\d.]+)", meta)
sL0, sdL, sedge = (float(sh.group(i)) for i in (1, 2, 3))

s = sorted(glob.glob(f"{d}/f2d_*.bin"))[fidx]
n = int(re.search(r"f2d_(\d+)", s).group(1))
a = np.memmap(s, dtype=np.float32, mode="r", shape=(6, nz, nx))
by = np.asarray(a[4], dtype=np.float32)
hi = by - gaussian_filter(by, 40)
xs = np.linspace(x0, x1, nx)
zs = np.linspace(z0, z1, nz)

def shell_outline(ax, xa=None, xb=None, za=None, zb=None):
    lam = np.linspace(-1.0, 1.0, 400)
    c = np.cos(lam)
    for Lc, st in [(sL0 - sdL/2, "-"), (sL0 + sdL/2, "-"),
                   (sL0 - sdL/2 - 3*sedge, "--"), (sL0 + sdL/2 + 3*sedge, "--")]:
        px, pz = Lc * c**3, Lc * c**2 * np.sin(lam)
        keep = np.ones_like(px, bool)
        if xa is not None:
            keep = (px > xa) & (px < xb) & (pz > za) & (pz < zb)
        ax.plot(px[keep], pz[keep], st, color="k", lw=0.7, alpha=0.7)

fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.2, 1.2])
axA = fig.add_subplot(gs[:, 0])
v = np.percentile(np.abs(hi), 99.8)
axA.imshow(hi, origin="lower", extent=[x0, x1, z0, z1], cmap="RdBu_r",
           vmin=-v, vmax=v, aspect="equal")
shell_outline(axA)
axA.set_title(f"high-pass By  step {n} t={n*dt*B0eq:.0f}/Ωe  (±{v:.1e})\n"
              "shell flat-top (—) / ±3σ (--)")
axA.set_xlabel("x"); axA.set_ylabel("z")

zooms = [("equator source", sL0*0.94, sL0*1.09, -120, 120),
         ("north mid-lat", sL0*0.87, sL0*1.05, 150, 400)]
for j, (t2, xa, xb, za, zb) in enumerate(zooms):
    ii = (xs > xa) & (xs < xb); kk = (zs > za) & (zs < zb)
    for col, (fld, tag) in enumerate([(hi, "high-pass"), (by, "RAW")]):
        sub = fld[np.ix_(kk, ii)]
        vv = np.percentile(np.abs(sub), 99.8)
        ax = fig.add_subplot(gs[j, 1 + col])
        ax.imshow(sub, origin="lower", extent=[xa, xb, za, zb], cmap="RdBu_r",
                  vmin=-vv, vmax=vv, aspect="equal")
        shell_outline(ax, xa, xb, za, zb)
        ax.set_title(f"{t2} {tag} (±{vv:.1e})")
fig.tight_layout()
out = f"{d}/wavemap_v2_{n//1000}k.png"
fig.savefig(out, dpi=110)
print("wrote", out)
