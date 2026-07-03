#!/usr/bin/env python3
"""omega-k dispersion diagram of delta E_L for whistler-pump Sims 1/2/3
(An et al. 2019 Fig 2b / 3b-c / 5b). 2D FFT of delta E_L(x,t) -> |E(omega,k)|^2,
showing the whistler line (low omega), its harmonics, and the beam-mode / Langmuir
branch, so phase velocities (omega/k) are directly readable.

Reads whistler_s{N}_kt.bin (nsamp x nx float32) + .meta.
Usage: python3 scripts/plot_whistler_dispersion.py --sim 2 [--out FILE]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("--sim", type=int, default=2)
ap.add_argument("--out", default=None)
args = ap.parse_args()
pref = f"whistler_s{args.sim}_"
out = args.out or f"whistler_s{args.sim}_dispersion.png"

m = dict(l.split() for l in open(pref + "kt.meta"))
nx, nsamp = int(m["nx"]), int(m["nsamp"])
dts = float(m["dt_sample"]); Lx = float(m["Lx"])
M = int(m["M"]); w0 = float(m["w0"]); vphx = float(m["vphx"])   # w0/k0

E = np.fromfile(pref + "kt.bin", dtype=np.float32).reshape(nsamp, nx)
win = np.hanning(nsamp)[:, None]
F = np.fft.fftshift(np.fft.fft2(E * win))
P = np.abs(F)**2
omega = np.fft.fftshift(np.fft.fftfreq(nsamp, d=dts)) * 2 * np.pi
kmode = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0/nx))

oi = (omega >= -1.3) & (omega <= 1.3)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# full mode range: beam-mode / Langmuir branch
kfull = 40 * M
mi = (kmode >= 1) & (kmode <= kfull)
Psub = P[np.ix_(oi, mi)]
im0 = ax[0].pcolormesh(omega[oi], kmode[mi], Psub.T,
                       norm=LogNorm(vmin=Psub.max()*1e-6, vmax=Psub.max()),
                       cmap="turbo", shading="auto")
ax[0].axvline(1.0, color="w", ls=":", lw=0.7); ax[0].axvline(-1.0, color="w", ls=":", lw=0.7)
ax[0].set_xlabel(r"$\omega/\omega_{pe}$"); ax[0].set_ylabel("mode number (k)")
ax[0].set_title(r"$|\delta E_L(\omega,k)|^2$ — nonlinear branch near/below $\omega_{pe}$")
fig.colorbar(im0, ax=ax[0])

# zoom: whistler + harmonics (low modes), with phase-velocity line
mzoom = 5 * M
mi2 = (kmode >= 1) & (kmode <= mzoom)
P2 = P[np.ix_(oi, mi2)]
im1 = ax[1].pcolormesh(omega[oi], kmode[mi2], P2.T,
                       norm=LogNorm(vmin=P2.max()*1e-5, vmax=P2.max()),
                       cmap="turbo", shading="auto")
# whistler phase-velocity line: k(mode) = |omega|/vphx * Lx/(2pi)
ax[1].plot(omega[oi], np.abs(omega[oi])/vphx*Lx/(2*np.pi), "w--", lw=0.9,
           label=f"whistler $v_{{ph,x}}$={vphx:.1f}")
for h in (M, 2*M, 3*M): ax[1].axhline(h, color="w", lw=0.3, ls=":")
ax[1].set_xlim(-6*w0, 6*w0); ax[1].set_ylim(1, mzoom)
ax[1].set_xlabel(r"$\omega/\omega_{pe}$"); ax[1].set_ylabel("mode number")
ax[1].set_title(f"zoom: whistler M={M} + harmonics on one phase-velocity line")
ax[1].legend(fontsize=8, loc="upper left"); fig.colorbar(im1, ax=ax[1])

fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out, " whistler v_ph,x = %.3f v_th" % vphx)
