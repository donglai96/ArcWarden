#!/usr/bin/env python3
"""omega-k dispersion diagram of delta E_L for the whistler-pump run
(An et al. 2019 Fig 2b). 2D FFT of delta E_L(x,t) -> |E(omega,k)|^2, showing the
whistler line (low omega), its harmonics, and the Langmuir branch near omega_pe,
so phase velocities (omega/k) are directly readable.

Reads whistler_kt.bin (nsamp x nx float32) + whistler_kt.meta.
Usage: python3 scripts/plot_whistler_dispersion.py [outfile]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out = sys.argv[1] if len(sys.argv) > 1 else "whistler_dispersion.png"
m = dict(l.split() for l in open("whistler_kt.meta"))
nx, nsamp = int(m["nx"]), int(m["nsamp"])
dts = float(m["dt_sample"]); Lx = float(m["Lx"])

E = np.fromfile("whistler_kt.bin", dtype=np.float32).reshape(nsamp, nx)
win = np.hanning(nsamp)[:, None]                       # time window
F = np.fft.fftshift(np.fft.fft2(E * win))
P = np.abs(F)**2
omega = np.fft.fftshift(np.fft.fftfreq(nsamp, d=dts)) * 2 * np.pi   # /omega_pe
kmode = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0/nx))               # integer mode

oi = (omega >= -1.3) & (omega <= 1.3)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# full mode range
mi = (kmode >= 1) & (kmode <= 450)
Psub = P[np.ix_(oi, mi)]
im0 = ax[0].pcolormesh(omega[oi], kmode[mi], Psub.T,
                       norm=LogNorm(vmin=Psub.max()*1e-6, vmax=Psub.max()),
                       cmap="turbo", shading="auto")
ax[0].axvline(1.0, color="w", ls=":", lw=0.7); ax[0].axvline(-1.0, color="w", ls=":", lw=0.7)
ax[0].set_xlabel(r"$\omega/\omega_{pe}$"); ax[0].set_ylabel("mode number (k)")
ax[0].set_title(r"$|\delta E_L(\omega,k)|^2$ — Langmuir near $\omega_{pe}$")
fig.colorbar(im0, ax=ax[0])

# zoom: whistler + harmonics (low modes), with phase-velocity line
mi2 = (kmode >= 1) & (kmode <= 40)
P2 = P[np.ix_(oi, mi2)]
k0 = 2*np.pi*10/Lx; w0 = 0.0215; vphx = w0/k0
im1 = ax[1].pcolormesh(omega[oi], kmode[mi2], P2.T,
                       norm=LogNorm(vmin=P2.max()*1e-5, vmax=P2.max()),
                       cmap="turbo", shading="auto")
ax[1].plot(omega[oi], np.abs(omega[oi])/vphx*Lx/(2*np.pi), "w--", lw=0.9,
           label=f"whistler $v_{{ph,x}}$={vphx:.2f}")
for h in (10, 20, 30): ax[1].axhline(h, color="w", lw=0.3, ls=":")
ax[1].set_xlim(-0.12, 0.12); ax[1].set_ylim(1, 40)
ax[1].set_xlabel(r"$\omega/\omega_{pe}$"); ax[1].set_ylabel("mode number")
ax[1].set_title("zoom: whistler M=10 + harmonics on one phase-velocity line")
ax[1].legend(fontsize=8, loc="upper left"); fig.colorbar(im1, ax=ax[1])

fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out, " whistler v_ph,x = %.3f v_th" % vphx)
