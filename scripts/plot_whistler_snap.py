#!/usr/bin/env python3
"""Paper-style snapshot figure for whistler-pump Sims 1/2/3 (An et al. 2019),
reproducing Fig 2(c-d) / Fig 4(a-b or c-d) / Fig 5(c-d).

Two panels sharing x (parallel/perpendicular are w.r.t. background B0):
  (1) delta E_parallel (red) and delta E_perp (blue) waveforms
  (2) f(x, v_parallel/v_th) phase-space density (full particle histogram)

Reads whistler_s{N}_fields.csv + whistler_s{N}_fhist.bin/.meta.
Usage: python3 scripts/plot_whistler_snap.py --sim 2 [--out FILE]
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
ap.add_argument("--df", action="store_true",
                help="add a signed-log perturbed density panel δf=f-<f>_x (reveals holes/beams)")
args = ap.parse_args()
pref = f"whistler_s{args.sim}_"
out = args.out or f"whistler_s{args.sim}_snap.png"

TAG = {1: r"$v_r/v_{th}=3.2$ — Langmuir waves",
       2: r"$v_r/v_{th}=2.1$ — electron-acoustic + unipolar",
       3: r"$v_r/v_{th}=1.0$ — phase-space holes + bipolar"}.get(args.sim, "")

m = dict(l.split() for l in open(pref + "fhist.meta"))
nxb, nvb = int(m["nxb"]), int(m["nvb"])
Lx = float(m["Lx"]); vlo, vhi = float(m["vlo"]), float(m["vhi"])
vr = float(m["vr"]); tsnap = float(m.get("tsnap", 0))
H = np.fromfile(pref + "fhist.bin", dtype=np.float32).reshape(nvb, nxb)
d = np.loadtxt(pref + "fields.csv", delimiter=",", skiprows=1)
# cols: x, Epar(=E·b̂), Eperp(=Ey pure transverse), Ey, dBy, dBmag
x, Epar, Eperp = d[:, 0], d[:, 1], d[:, 2]

nrow = 3 if args.df else 2
fig = plt.figure(figsize=(11, 8.5 if args.df else 6.5))
hr = [1, 1.7, 1.7] if args.df else [1, 1.7]
gs = fig.add_gridspec(nrow, 2, width_ratios=[1, 0.022], height_ratios=hr,
                      hspace=0.10, wspace=0.02)
a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0)
c1 = fig.add_subplot(gs[1, 1]); fig.add_subplot(gs[0, 1]).axis("off")

a0.plot(x, Epar,  color="#d1495b", lw=0.7, label=r"$\delta E_\parallel$")
a0.plot(x, Eperp, color="#3a6ea5", lw=0.7, label=r"$\delta E_\perp$")
a0.set_ylabel(r"$\delta E$"); a0.grid(alpha=.2)
a0.legend(loc="upper right", fontsize=10, ncol=2)
a0.set_title(f"ArcWarden — whistler-pump Sim {args.sim} (An et al. 2019),  {TAG},  "
             f"$t={tsnap:.0f}\\,\\omega_{{pe}}^{{-1}}$")
plt.setp(a0.get_xticklabels(), visible=False)

im = a1.imshow(H, origin="lower", extent=[0, Lx, vlo, vhi], aspect="auto",
               cmap="turbo", norm=LogNorm(vmin=max(H.max() * 2e-4, 1), vmax=H.max()))
a1.axhline(vr, color="w", ls="--", lw=0.8)
a1.set_ylabel(r"$v_\parallel/v_{th}$")
fig.colorbar(im, cax=c1, label="f")

if args.df:
    plt.setp(a1.get_xticklabels(), visible=False)
    a2 = fig.add_subplot(gs[2, 0], sharex=a0); c2 = fig.add_subplot(gs[2, 1])
    # perturbed density: subtract the x-averaged distribution, signed-log scale.
    dF = H - H.mean(axis=1, keepdims=True)
    s = np.sign(dF) * np.log10(1.0 + np.abs(dF))
    vmax = np.percentile(np.abs(s), 99.5)
    im2 = a2.imshow(s, origin="lower", extent=[0, Lx, vlo, vhi], aspect="auto",
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    a2.axhline(vr, color="k", ls="--", lw=0.8)
    a2.set_ylabel(r"$v_\parallel/v_{th}$"); a2.set_xlabel("x")
    fig.colorbar(im2, cax=c2, label=r"sng$\log_{10}\delta f$")
    a2.set_ylim(vr - 3, vr + 3)          # zoom on the resonant island (holes/beams)
else:
    a1.set_xlabel("x")

fig.savefig(out, dpi=130)
print("wrote", out, f"(v_r/v_th={vr:.2f}, t_snap={tsnap:.0f})")
