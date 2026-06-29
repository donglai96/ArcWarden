#!/usr/bin/env python3
"""Paper-style figure for the whistler-pump Simulation 1 (An et al. 2019).

Two panels sharing x (parallel/perpendicular are w.r.t. the background B0):
  (1) delta E_parallel (red) and delta E_perp (blue) waveforms
  (2) f(x, v_parallel) phase-space density (full particle histogram)

Reads whistler_fields.csv + whistler_fhist.bin/.meta written by tools/whistler_pump.cu.
Usage: python3 scripts/plot_whistler_sim1.py [outfile]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out = sys.argv[1] if len(sys.argv) > 1 else "whistler_sim1.png"

m = dict(l.split() for l in open("whistler_fhist.meta"))
nxb, nvb = int(m["nxb"]), int(m["nvb"])
Lx = float(m["Lx"]); vlo, vhi = float(m["vlo"]), float(m["vhi"]); vr = float(m["vr"])
H = np.fromfile("whistler_fhist.bin", dtype=np.float32).reshape(nvb, nxb)
d = np.loadtxt("whistler_fields.csv", delimiter=",", skiprows=1)
# cols: x, Epar(=E·b̂), Eperp(in-plane proj, mixes longitudinal Ex), Ey(pure transverse), dBy, dBmag
x, Epar, Ey = d[:, 0], d[:, 1], d[:, 3]
Eperp = Ey   # δE⊥ = pure transverse EM field (clean): longitudinal Ex (Langmuir/noise) is ⊥ to y

fig = plt.figure(figsize=(11, 6.5))
# fixed colorbar column so both plot panels have identical width
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.022], height_ratios=[1, 1.7],
                      hspace=0.10, wspace=0.02)
a0 = fig.add_subplot(gs[0, 0])
a1 = fig.add_subplot(gs[1, 0], sharex=a0)
c1 = fig.add_subplot(gs[1, 1])
fig.add_subplot(gs[0, 1]).axis("off")

# Panel 1: dE_par (red) + dE_perp (blue), both relative to B0
a0.plot(x, Epar,  color="#d1495b", lw=0.7, label=r"$\delta E_\parallel$")
a0.plot(x, Eperp, color="#3a6ea5", lw=0.7, label=r"$\delta E_\perp$")
a0.set_ylabel(r"$\delta E$"); a0.grid(alpha=.2)
a0.legend(loc="upper right", fontsize=10, ncol=2)
a0.set_title("ArcWarden — whistler-pump Sim 1 (An et al. 2019),  "
             "$v_r=%.2f\\,v_{th}$,  fields $\\parallel/\\perp$ to $B_0$" % vr)
plt.setp(a0.get_xticklabels(), visible=False)

# Panel 2: f(x, v_par)
im = a1.imshow(H, origin="lower", extent=[0, Lx, vlo, vhi], aspect="auto",
               cmap="turbo", norm=LogNorm(vmin=max(H.max() * 2e-4, 1), vmax=H.max()))
a1.axhline(vr, color="w", ls="--", lw=0.8)
a1.set_ylabel(r"$v_\parallel/v_{th}$"); a1.set_xlabel("x")
fig.colorbar(im, cax=c1, label="f")

fig.savefig(out, dpi=130)
print("wrote", out)
