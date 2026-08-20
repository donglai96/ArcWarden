#!/usr/bin/env python3
"""Wave-field GIF from f2d_*.bin snapshots (By = out-of-plane wave B).

Conventions follow plot_wavemap2d.py / the wave-figure ritual: shell
contours (black) + active band (green dashed) on EVERY frame; ONE fixed
symmetric color scale across all frames (per-frame scaling would fake
growth); time stamp + per-frame max annotation.

Usage: wavemap_gif.py <outdir> [out.gif] [--ds=3] [--fps=12] [--vpct=99.5]
"""
import glob
import io
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

outdir = sys.argv[1]
gif = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
    else f"{outdir}/wavemap.gif"
ds, fps, vpct = 3, 12, 99.5
for a in sys.argv[2:]:
    if a.startswith("--ds="): ds = int(a[5:])
    if a.startswith("--fps="): fps = int(a[6:])
    if a.startswith("--vpct="): vpct = float(a[7:])

meta = open(f"{outdir}/meta.txt").read()
m = re.search(r"box ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) "
              r"(\d+) (\d+) ([-\d.e+]+) ([-\d.e+]+)", meta)
x0, x1, z0, z1 = (float(m.group(i)) for i in range(1, 5))
nx, nz = int(m.group(5)), int(m.group(6))
dx, dz = float(m.group(7)), float(m.group(8))
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
prof = re.search(r"profile (\w+)", meta).group(1)
B0eq = float(re.search(r"B0eq ([\d.]+)", meta).group(1))
shells = [(n, float(a), float(b), float(e)) for n, a, b, e in
          re.findall(r"shell (\S+) L0=([\d.]+) dL=([\d.]+) edge=([\d.]+)", meta)]
bands = re.findall(r"active_band ([\d.]+) ([\d.]+)", meta)

snaps = sorted(glob.glob(f"{outdir}/f2d_*.bin"))
if not snaps:
    raise SystemExit("no f2d snapshots")

X, Z = np.meshgrid(x0 + (np.arange(nx)[::ds] + 0.5) * dx,
                   z0 + (np.arange(nz)[::ds] + 0.5) * dz)
R2 = X**2 + Z**2
L = R2**1.5 / X**2 if prof == "dipole2d" else R2 / X

# fixed scale: percentile over a late-third sample of frames
probe_frames = snaps[2 * len(snaps) // 3::max(1, len(snaps) // 12)]
amp = max(max(np.percentile(np.abs(np.memmap(p, dtype=np.float32, mode="r",
                                             shape=(6, nz, nx))[4, ::4 * ds, ::4 * ds]),
                            vpct) for p in probe_frames), 1e-12)

frames = []
for path in snaps:
    step = int(re.search(r"f2d_(\d+)\.bin", path).group(1))
    by = np.array(np.memmap(path, dtype=np.float32, mode="r",
                            shape=(6, nz, nx))[4, ::ds, ::ds])
    fig, ax = plt.subplots(figsize=(4.6, 8.2))
    ax.pcolormesh(X, Z, by, cmap="RdBu_r", vmin=-amp, vmax=amp,
                  shading="auto", rasterized=True)
    for _, L0s, dLs, _e in shells:
        ax.contour(X, Z, L, levels=[L0s - dLs / 2, L0s + dLs / 2],
                   colors="k", linewidths=0.5)
    for lo, hi in bands:
        ax.contour(X, Z, L, levels=[float(lo), float(hi)], colors="g",
                   linewidths=0.6, linestyles="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x [c/ωpe]"); ax.set_ylabel("z [c/ωpe]")
    ax.set_title(f"$B_y$   t = {step * dt:6.0f}/ωpe\n"
                 f"|By|max = {np.abs(by).max():.2e}"
                 f" ({np.abs(by).max() / B0eq:.1e} B0eq)", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

frames[0].save(gif, save_all=True, append_images=frames[1:],
               duration=int(1000 / fps), loop=0, optimize=True)
print(f"wrote {gif}: {len(frames)} frames @ {fps} fps, "
      f"scale ±{amp:.2e} ({amp / B0eq:.1e} B0eq)")
