#!/usr/bin/env python3
"""k-t (mode-number vs time) spectrum of the longitudinal field delta E_L for
whistler-pump Sims 1/2/3 (An et al. 2019, Fig 2a / 3a / 5a). Shows whether the
small-scale nonlinear structures (Langmuir band / electron-acoustic / bipolar
modes) are coherently excited, vs the whistler (mode M) and its harmonics.

Reads whistler_s{N}_kt.bin (nsamp x nx float32 = delta E_L(x,t)) + .meta.
Usage: python3 scripts/plot_whistler_kt.py --sim 2 [--out FILE]
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
out = args.out or f"whistler_s{args.sim}_kt.png"

m = dict(l.split() for l in open(pref + "kt.meta"))
nx, nsamp = int(m["nx"]), int(m["nsamp"])
dts = float(m["dt_sample"]); toff = float(m["toff"])
M = int(m["M"]); blo, bhi = int(m["band_lo"]), int(m["band_hi"])
E = np.fromfile(pref + "kt.bin", dtype=np.float32).reshape(nsamp, nx)

Fk = np.abs(np.fft.rfft(E, axis=1))**2 / nx**2     # (nsamp, nx/2+1)
t = np.arange(nsamp) * dts
kmode = np.arange(Fk.shape[1])

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1]})

# full mode range, log mode axis
kmax = min(bhi + 100, Fk.shape[1] - 1)
P = Fk[:, 1:kmax+1].T
im0 = ax[0].pcolormesh(t, kmode[1:kmax+1], P,
                       norm=LogNorm(vmin=P.max()*1e-7, vmax=P.max()),
                       cmap="turbo", shading="auto")
ax[0].set_yscale("log"); ax[0].set_ylim(1, kmax)
ax[0].axvline(toff, color="w", ls=":", lw=1.0)
ax[0].text(toff, 1.3, " t_off", color="w", fontsize=9)
for mm in (M, 2*M, 3*M): ax[0].axhline(mm, color="w", lw=0.3, ls=":")
ax[0].set_xlabel(r"$t\,\omega_{pe}$"); ax[0].set_ylabel("mode number")
ax[0].set_title(rf"$|\delta E_L(k,t)|^2$  (whistler M={M}, harmonics {2*M}/{3*M}, band {blo}-{bhi})")
fig.colorbar(im0, ax=ax[0])

# zoom on the nonlinear-structure band, linear mode axis
P2 = Fk[:, blo:bhi+1].T
im1 = ax[1].pcolormesh(t, np.arange(blo, bhi+1), P2,
                       norm=LogNorm(vmin=max(P2.max()*1e-3, 1e-12), vmax=P2.max()),
                       cmap="turbo", shading="auto")
ax[1].axvline(toff, color="w", ls=":", lw=1.0)
ax[1].set_xlabel(r"$t\,\omega_{pe}$"); ax[1].set_ylabel("mode number")
ax[1].set_title(f"zoom: nonlinear-structure band (modes {blo}-{bhi})")
fig.colorbar(im1, ax=ax[1])

fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)
band = Fk[:, blo:bhi+1].sum(1)
floor = band[:nsamp//5].mean()
above = np.where(band > 10*floor)[0]
if len(above):
    print(f"band ({blo}-{bhi}) exceeds 10x its early floor at t = {t[above[0]]:.0f} wpe^-1 "
          f"(t_off={toff:.0f}); peak/floor = {band.max()/floor:.1f}")
else:
    print(f"band never exceeds 10x floor (peak/floor={band.max()/floor:.1f})")
