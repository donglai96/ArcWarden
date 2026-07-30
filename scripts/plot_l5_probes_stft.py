#!/usr/bin/env python3
"""17-probe R-mode spectrogram stack for the l5_diag run (works mid-run).

Reads <dir>/probe.bin (float32, records of (By,Bz) x nprobe every probe_every
steps) + meta.txt. R-mode = By+iBz at POSITIVE STFT frequencies (calibrated
convention). Column-normalized log-power, nwin from --nwin (default 2048).
Dashed line = local wce(lat)/We0; dotted = half of it.
"""
import argparse
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default="build/l5_diag")
ap.add_argument("--nwin", type=int, default=2048)
ap.add_argument("--fmax", type=float, default=1.2, help="max freq in We0 units")
ap.add_argument("--out", default=None)
args = ap.parse_args()

meta = {}
for line in open(args.dir + "/meta.txt"):
    k, v = line.split(None, 1)
    meta.setdefault(k, []).append(v.strip())
dt = float(meta["dt"][0])
pe = int(meta["probe_every"][0])
npr = int(meta["nprobe"][0])
wce = float(meta["wce"][0])
lre = float(meta["b0_lre"][0])
xc = float(meta["b0_xc"][0])
dx = float(meta["dx"][0])
probe_ix = [int(v) for v in meta["probe_ix"]]

# probe latitude from arc length s = |x - xc|
def lat_of_s(s_target):
    if s_target < 1e-6:
        return 0.0
    lo, hi = 0.0, 40.0
    f = lambda lam: lre * 0.5 * (math.sin(math.radians(lam)) *
        math.sqrt(1 + 3 * math.sin(math.radians(lam)) ** 2) +
        math.asinh(math.sqrt(3) * math.sin(math.radians(lam))) / math.sqrt(3))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < s_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def bratio(lam):
    s, c = math.sin(math.radians(lam)), math.cos(math.radians(lam))
    return math.sqrt(1 + 3 * s * s) / c ** 6

offs = [(ix + 0.5) * dx - xc for ix in probe_ix]
lats = [math.copysign(lat_of_s(abs(o)), o) for o in offs]

raw = np.fromfile(args.dir + "/probe.bin", dtype=np.float32)
nrec = raw.size // (2 * npr)
raw = raw[: nrec * 2 * npr].reshape(nrec, npr, 2)
sig = raw[:, :, 0] + 1j * raw[:, :, 1]
dts = dt * pe                       # sample spacing in 1/wpe
t_end_We0 = nrec * dts * wce
print(f"{nrec} records, t = 0..{t_end_We0:.0f}/We0, {npr} probes")

order = np.argsort(lats)[::-1]      # +25 deg top -> -25 deg bottom
fig, axes = plt.subplots(npr, 1, figsize=(11, 1.35 * npr + 1.5),
                         sharex=True, constrained_layout=True)
for row, p in enumerate(order):
    ax = axes[row]
    f, tt, Z = stft(sig[:, p], fs=2 * np.pi / dts, nperseg=args.nwin,
                    noverlap=args.nwin - args.nwin // 8,
                    return_onesided=False, boundary=None)
    pos = f >= 0
    f, Z = f[pos] / wce, np.abs(Z[pos]) ** 2       # omega/We0, R-mode power
    sel = f <= args.fmax
    f, Z = f[sel], Z[sel]
    Z = Z / np.maximum(Z.max(axis=0, keepdims=True), 1e-30)   # column-norm
    # stft with fs=2*pi/dts returns f in rad/time (wanted) but t in
    # time/(2*pi) — undo the 2*pi on the time axis.
    ax.pcolormesh(tt * 2 * np.pi * wce, f, np.log10(np.maximum(Z, 1e-4)),
                  cmap="turbo", vmin=-3, vmax=0, rasterized=True)
    r = bratio(abs(lats[p]))
    ax.axhline(r, ls="--", c="w", lw=0.7, alpha=0.8)
    ax.axhline(0.5 * r, ls=":", c="w", lw=0.7, alpha=0.8)
    ax.set_ylim(0, args.fmax)
    ax.set_ylabel(f"{lats[p]:+.1f}$^\\circ$", fontsize=8)
    ax.tick_params(labelsize=7)
axes[-1].set_xlabel("t  [1/We0]")
fig.suptitle(f"l5_diag 17-probe R-mode STFT (col-norm log10 P, nwin={args.nwin})"
             f"  —  dashed: local wce, dotted: 0.5 wce", fontsize=10)
out = args.out or args.dir + "_probes_stft.png"
fig.savefig(out, dpi=110)
print("wrote", out)
