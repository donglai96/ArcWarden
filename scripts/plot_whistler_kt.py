#!/usr/bin/env python3
"""k-t (mode-number vs time) spectrum of the longitudinal field delta E_L,
reproducing An et al. (2019) Fig 2(a). Shows whether the small-scale Langmuir
waves (modes 300-400) are coherently excited, vs the whistler (mode 10) and its
harmonics (20, 30).

Reads whistler_kt.bin (nsamp x nx float32, = delta E_L(x,t)) + .meta.
Usage: python3 scripts/plot_whistler_kt.py [outfile]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out = sys.argv[1] if len(sys.argv) > 1 else "whistler_kt.png"
m = dict(l.split() for l in open("whistler_kt.meta"))
nx, nsamp = int(m["nx"]), int(m["nsamp"])
dts = float(m["dt_sample"]); toff = float(m["toff"])
E = np.fromfile("whistler_kt.bin", dtype=np.float32).reshape(nsamp, nx)

# |E_L(k, t)|^2 : FFT in x for each time snapshot
Fk = np.abs(np.fft.rfft(E, axis=1))**2 / nx**2     # (nsamp, nx/2+1)
t = np.arange(nsamp) * dts
kmode = np.arange(Fk.shape[1])                       # integer mode number

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios":[1,1]})

# full mode range, log mode axis (like Fig 2a)
kmax = 500
P = Fk[:, 1:kmax+1].T                                # (modes, time)
im0 = ax[0].pcolormesh(t, kmode[1:kmax+1], P,
                       norm=LogNorm(vmin=P.max()*1e-7, vmax=P.max()),
                       cmap="turbo", shading="auto")
ax[0].set_yscale("log"); ax[0].set_ylim(1, kmax)
ax[0].axvline(toff, color="w", ls=":", lw=1.0)
ax[0].text(toff, 1.3, " t_off", color="w", fontsize=9)
for mm in (10, 20, 30): ax[0].axhline(mm, color="w", lw=0.3, ls=":")
ax[0].set_xlabel(r"$t\,\omega_{pe}$"); ax[0].set_ylabel("mode number")
ax[0].set_title(r"$|\delta E_L(k,t)|^2$  (whistler M=10, harmonics 20/30, Langmuir 300-400)")
fig.colorbar(im0, ax=ax[0])

# zoom on the Langmuir band, linear mode axis
P2 = Fk[:, 250:451].T
im1 = ax[1].pcolormesh(t, np.arange(250,451), P2,
                       norm=LogNorm(vmin=max(P2.max()*1e-3,1e-12), vmax=P2.max()),
                       cmap="turbo", shading="auto")
ax[1].axvline(toff, color="w", ls=":", lw=1.0)
ax[1].set_xlabel(r"$t\,\omega_{pe}$"); ax[1].set_ylabel("mode number")
ax[1].set_title("zoom: Langmuir band (modes 250-450) — excited after beam forms?")
fig.colorbar(im1, ax=ax[1])

fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)
# quantitative: when does the Langmuir band rise above its early-time floor?
band = Fk[:, 300:401].sum(1)
floor = band[:nsamp//5].mean()
above = np.where(band > 10*floor)[0]
if len(above):
    print(f"Langmuir band (300-400) exceeds 10x its early floor at t = {t[above[0]]:.0f} wpe^-1 "
          f"(t_off={toff:.0f}); peak/floor = {band.max()/floor:.1f}")
else:
    print(f"Langmuir band never exceeds 10x floor (peak/floor={band.max()/floor:.1f}) -> likely noise")
