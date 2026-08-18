#!/usr/bin/env python3
"""Deck-agnostic live quicklook: W_EM history + B1 spectrograms at the +5
and +10 degree probes, everything (cadence, dt, local Omega_e per probe)
read from <outdir>/meta.txt. Safe to run while the simulation writes.

Usage: quicklook2d.py <outdir> [out.png]
"""
import csv
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1] if len(sys.argv) > 1 else "build/v4r5"
png = sys.argv[2] if len(sys.argv) > 2 else f"{outdir}/quicklook.png"

meta = open(f"{outdir}/meta.txt").read()
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
pev = int(re.search(r"probe_every (\d+)", meta).group(1))
probes = [(float(a), float(b)) for a, b in
          re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
names = [f"{lam:+.0f}°" if lam else "eq" for lam, _ in probes]
NP, NC = len(probes), 6
dt_s = dt * pev

raw = np.fromfile(f"{outdir}/probes.bin", dtype=np.float32)
ns = raw.size // (NP * NC)
pr = raw[:ns * NP * NC].reshape(ns, NC, NP)

t, wem = [], []
with open(f"{outdir}/energy.csv") as f:
    for row in csv.DictReader(f):
        t.append(float(row["t"]))
        wem.append(float(row["W_EM"]))

want = [i for i, (lam, _) in enumerate(probes) if lam in (5.0, 10.0)][:2]
fig, axs = plt.subplots(1, 1 + len(want), figsize=(16, 4.6))
axs = np.atleast_1d(axs)
axs[0].semilogy(t, wem, "k-")
axs[0].set_xlabel(r"$t$ [$1/\omega_{pe}$]")
axs[0].set_ylabel(r"$W_{EM}$")
axs[0].grid(alpha=0.3)
axs[0].set_title(f"wave energy — {ns} probe samples, t = {ns * dt_s:.0f}")

NW = max(256, min(1024, (ns // 8) & ~1))
HOP = max(32, NW // 8)
w = np.hanning(NW)
for ax, pidx in zip(axs[1:], want):
    sig = pr[:, 4, pidx]                       # B1 (perp in-plane wave B)
    nwin = max(1, (ns - NW) // HOP)
    S = np.zeros((NW // 2 + 1, nwin))
    for i in range(nwin):
        S[:, i] = np.abs(np.fft.rfft(sig[i * HOP:i * HOP + NW] * w))**2
    fr = np.fft.rfftfreq(NW, dt_s) * 2 * np.pi / probes[pidx][1]
    tw = (np.arange(nwin) * HOP + NW / 2) * dt_s
    m = (fr > 0.05) & (fr < 0.9)
    v = np.log10(S[m] + 1e-24)
    pc = ax.pcolormesh(tw, fr[m], v, cmap="turbo",
                       vmin=np.percentile(v, 55), vmax=np.percentile(v, 99.7),
                       shading="auto", rasterized=True)
    ax.axhline(0.5, color="w", ls="--", lw=0.8)
    ax.set_title(f"B1 spectrogram, probe {names[pidx]} (local $\\Omega_e$)")
    ax.set_xlabel(r"$t$ [$1/\omega_{pe}$]")
    ax.set_ylabel(r"$\omega/\Omega_{e,\rm local}$")
    plt.colorbar(pc, ax=ax)
fig.tight_layout()
fig.savefig(png, dpi=130)
print(f"rendered {png}: {ns} samples, t = {ns * dt_s:.0f}")
