#!/usr/bin/env python3
"""Live quicklook for the V4R3 run: wave-energy history + B1 spectrograms
at the +5 and +10 degree probes (local Omega_e normalization). Reads
build/v4r3/{probes.bin,energy.csv}; writes docs/figs/v4r3_live.png.
Safe to run while the simulation is writing (reads whole samples only).
"""
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "build/v4r3"
raw = np.fromfile(f"{BASE}/probes.bin", dtype=np.float32)
NP, NC = 7, 6
ns = raw.size // (NP * NC)
pr = raw[:ns * NP * NC].reshape(ns, NC, NP)
dt_s = 0.15 * 4
names = ["eq", "+5", "-5", "+10", "-10", "+20", "-20"]
wce = [0.2 * math.sqrt(1 + 4 * math.tan(math.radians(l))**2) /
       math.cos(math.radians(l))**2 for l in [0, 5, -5, 10, -10, 20, -20]]

t, wem = [], []
with open(f"{BASE}/energy.csv") as f:
    for row in csv.DictReader(f):
        t.append(float(row["t"]))
        wem.append(float(row["W_EM"]))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axs[0]
ax.semilogy(t, wem, "k-")
ax.set_xlabel(r"$t$ [$1/\omega_{pe}$]")
ax.set_ylabel(r"$W_{EM}$")
ax.grid(alpha=0.3)
ax.set_title(f"V4R3 (rel, A=3) wave energy — t = {ns*dt_s:.0f}/30000")

NW = max(256, min(1024, (ns // 8) & ~1))
HOP = max(32, NW // 8)
w = np.hanning(NW)
for ax, pidx in zip(axs[1:], [1, 3]):
    sig = pr[:, 4, pidx]
    nwin = max(1, (ns - NW) // HOP)
    S = np.zeros((NW // 2 + 1, nwin))
    for i in range(nwin):
        S[:, i] = np.abs(np.fft.rfft(sig[i * HOP:i * HOP + NW] * w))**2
    fr = np.fft.rfftfreq(NW, dt_s) * 2 * np.pi / wce[pidx]
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
fig.savefig("docs/figs/v4r3_live.png", dpi=130)
print(f"rendered: {ns} samples, t = {ns*dt_s:.0f}")
