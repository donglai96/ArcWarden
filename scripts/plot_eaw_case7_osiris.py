#!/usr/bin/env python3
"""Ma et al. (PoP 2024) Figure-1 analog from an OSIRIS run of case 7 — the
same 8 panels as scripts/plot_eaw_case7.py, so the ArcWarden and OSIRIS
results can be compared frame by frame.

Reads the OSIRIS MS/ tree in ZDF format (this build has no HDF5):
2D dumps MS/FLD/<q>/<q>-NNNNNN.zdf, x1 lineouts
MS/FLD/<q>-line/<q>-line-x1-01-NNNNNN.zdf. Uses the zdf reader shipped in
the OSIRIS source tree.

Usage: python3 scripts/plot_eaw_case7_osiris.py <osiris_rundir> [--ta 800] [--tb 1100]
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/donggua/Donglai_Ma_reborn/osiris_develop/osiris/source/zdf")
import zdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("--ta", type=float, default=800.0)
ap.add_argument("--tb", type=float, default=1100.0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or args.rundir

def read_zdf(path):
    d, info = zdf.read(path)
    return float(info.iteration.t), np.asarray(d)

def series(pattern):
    return [read_zdf(p) for p in sorted(glob.glob(pattern))]

ms = os.path.join(args.rundir, "MS")
Lx = Ly = 13.5
wce = 0.25

# ---- lineouts of e1 along x1 (kx-t and w-kx panels) ----
lines = series(os.path.join(ms, "FLD", "e1-line", "e1-line-x1-01-*.zdf"))
if not lines:
    sys.exit("no e1 lineout files found under MS/FLD/e1-line/")

t = np.array([x[0] for x in lines])
Ext = np.array([np.asarray(x[1]).ravel() for x in lines])
nx = Ext.shape[1]
tline = t[1] - t[0]
print(f"lineouts: {len(t)} samples, dt_line={tline:.4f}, nx={nx}")

dEx = Ext - Ext.mean(axis=0)
kx = 2 * np.pi * np.arange(nx // 2 + 1) / Lx
Pkt = np.abs(np.fft.rfft(dEx, axis=1) / nx) ** 2

nline = len(t)
w = 2 * np.pi * np.arange(nline // 2 + 1) / (nline * tline)
F = np.fft.fft2(dEx) / (nline * nx)
nw, nk = nline // 2 + 1, nx // 2 + 1
Pwk = np.abs(F[:nw, :nk]) ** 2
Pwk[:, 1:] += np.abs(F[:nw, :-(nk):-1]) ** 2

# ---- 2D snapshots of e1 (Ex) and b2 (By) at ta, tb ----
def snap_at(quant, tt):
    s = series(os.path.join(ms, "FLD", quant, f"{quant}-*.zdf"))
    if not s:
        sys.exit(f"no 2D dumps for {quant}")
    times = np.array([x[0] for x in s])
    i = int(np.argmin(np.abs(times - tt)))
    print(f"{quant}: snapshot at t={times[i]:.1f} (requested {tt})")
    return s[i][1]

ExA, ExB = snap_at("e1", args.ta), snap_at("e1", args.tb)
ByA, ByB = snap_at("b2", args.ta), snap_at("b2", args.tb)
ny = ExA.shape[0]

kx2 = np.fft.fftshift(np.fft.fftfreq(ExA.shape[1], d=Lx / ExA.shape[1])) * 2 * np.pi
ky2 = np.fft.fftshift(np.fft.fftfreq(ny, d=Ly / ny)) * 2 * np.pi

def p2d(a):
    return np.fft.fftshift(np.abs(np.fft.fft2(a - a.mean()) / a.size) ** 2)

def wna(a, kmax=8.0):
    P = p2d(a)
    KX, KY = np.meshgrid(kx2, ky2)
    k = np.hypot(KX, KY)
    sel = (k > 0.5) & (k < kmax)
    th = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))
    return (P[sel] * th[sel]).sum() / P[sel].sum()

# ---- figure (same layout as plot_eaw_case7.py) ----
fig, ax = plt.subplots(2, 4, figsize=(19, 8.5))
a = ax[0, 0]
im = a.pcolormesh(t, kx[kx <= 30], Pkt[:, kx <= 30].T, norm=LogNorm(1e-11, 1e-6),
                  cmap="rainbow", shading="auto")
a.axvline(args.ta, color="r", ls="--", lw=0.8); a.axvline(args.tb, color="k", ls="--", lw=0.8)
a.set_xlabel(r"$t\,[\omega_{pe}^{-1}]$"); a.set_ylabel(r"$k_x\,[\omega_{pe}/c]$")
a.set_title(r"(a) $|\delta E_x(t,k_x)|^2$"); plt.colorbar(im, ax=a)

a = ax[1, 0]
im = a.pcolormesh(kx[kx <= 30], w[w <= 1.6], Pwk[w <= 1.6][:, kx <= 30],
                  norm=LogNorm(1e-12, 1e-7), cmap="rainbow", shading="auto")
a.axhline(wce, color="k", ls="--", lw=0.8)
a.plot(kx[kx <= 30], 0.0546 * kx[kx <= 30], "b-.", lw=1.0)
a.set_ylim(0, 1.6); a.set_xlabel(r"$k_x\,[\omega_{pe}/c]$"); a.set_ylabel(r"$\omega\,[\omega_{pe}]$")
a.set_title(r"(b) $|\delta E_x(\omega,k_x)|^2$,  line: $0.0546c\,k_x$"); plt.colorbar(im, ax=a)

for col, (Ex2, tt) in enumerate([(ExA, args.ta), (ExB, args.tb)]):
    dE = Ex2 - Ex2.mean()
    a = ax[0, 1 + col]
    vm = np.abs(dE).max()
    a.imshow(dE / vm, origin="lower", extent=[0, Ly, 0, Lx], cmap="bwr", vmin=-1, vmax=1,
             aspect="equal")
    a.set_title(rf"({'cd'[col]}) $\delta E_x$, t={tt:.0f}")
    a.set_xlabel(r"$y\,[c/\omega_{pe}]$"); a.set_ylabel(r"$x\,[c/\omega_{pe}]$")
    a = ax[1, 1 + col]
    Pe = p2d(dE)
    im = a.pcolormesh(ky2, kx2, Pe, norm=LogNorm(Pe.max() * 1e-5 + 1e-30, Pe.max() + 1e-25),
                      cmap="rainbow", shading="auto")
    a.set_xlim(-15, 15); a.set_ylim(-15, 15)
    a.set_title(rf"({'ef'[col]}) $|\delta E_x(k_x,k_y)|^2$, t={tt:.0f}")
    a.set_xlabel(r"$k_y\,[\omega_{pe}/c]$"); a.set_ylabel(r"$k_x\,[\omega_{pe}/c]$")
    plt.colorbar(im, ax=a)

for col, (By2, tt) in enumerate([(ByA, args.ta), (ByB, args.tb)]):
    a = ax[col, 3]
    Pb2 = p2d(By2)
    im = a.pcolormesh(ky2, kx2, Pb2, norm=LogNorm(Pb2.max() * 1e-5 + 1e-30, Pb2.max() + 1e-25),
                      cmap="rainbow", shading="auto")
    a.set_xlim(-15, 15); a.set_ylim(-15, 15)
    a.set_title(rf"({'gh'[col]}) $|\delta B_y(k_x,k_y)|^2$, t={tt:.0f}, "
                rf"WNA={wna(By2):.1f}$^\circ$")
    a.set_xlabel(r"$k_y\,[\omega_{pe}/c]$"); a.set_ylabel(r"$k_x\,[\omega_{pe}/c]$")
    plt.colorbar(im, ax=a)

fig.suptitle(r"OSIRIS 4.4.4 CUDA — Ma et al. PoP 2024 case 7 "
             r"($\beta_\parallel$=0.0091, $T_\perp/T_\parallel$=5)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out, "eaw_case7_osiris_fig1.png"), dpi=130)
print("wrote", os.path.join(out, "eaw_case7_osiris_fig1.png"))
print(f"WNA(By) tA: {wna(ByA):.1f} deg   tB: {wna(ByB):.1f} deg")
