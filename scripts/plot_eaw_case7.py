#!/usr/bin/env python3
"""Ma et al. (PoP 2024) Figure-1 analog from an eaw2d_yee run (case 7).

Panels (matching the paper):
  (a) |dEx(t,kx)|^2 wave-number spectrogram      (from <p>_linex.bin)
  (b) |dEx(w,kx)|^2 dispersion + v_ph line       (from <p>_linex.bin)
  (c,d) dEx(x,y) at tA=800, tB=1100              (from <p>_snap.bin)
  (e,f) |dEx(kx,ky)|^2 at tA, tB
  (g,h) |dBy(kx,ky)|^2 at tA, tB
Second figure: energy/anisotropy history + Ex(x) waveform (TDS) + WNA numbers.

Usage: python3 scripts/plot_eaw_case7.py <outdir> [--prefix eaw_case7] [--ta 800] [--tb 1100]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--prefix", default="eaw_case7")
ap.add_argument("--ta", type=float, default=800.0)
ap.add_argument("--tb", type=float, default=1100.0)
args = ap.parse_args()
p = os.path.join(args.outdir, args.prefix + "_")

m = dict(l.split() for l in open(p + "meta.txt"))
nx, ny = int(m["nx"]), int(m["ny"])
Lx, Ly = float(m["Lx"]), float(m["Ly"])
tline, tsnap = float(m["t_line"]), float(m["t_snap"])
nline, nsnap = int(m["nline"]), int(m["nsnap"])
wce, np_ = float(m["wce"]), float(m["np"])
uthx = float(m["uthx"])

lx = np.fromfile(p + "linex.bin", dtype=np.float32)
nline = lx.size // (6 * nx)
lx = lx[: nline * 6 * nx].reshape(nline, 6, nx)
Ext = lx[:, 0, :].astype(np.float64)          # Ex(t, x) at j=jline
t = np.arange(nline) * tline

# ---- (a) kx-t spectrogram ----------------------------------------------------
dEx = Ext - Ext.mean(axis=0)                  # remove static (Gauss-noise) pattern
kx = 2 * np.pi * np.arange(nx // 2 + 1) / Lx
Pkt = np.abs(np.fft.rfft(dEx, axis=1) / nx) ** 2

# ---- (b) w-kx dispersion -----------------------------------------------------
w = 2 * np.pi * np.arange(nline // 2 + 1) / (nline * tline)
F = np.fft.fft2(dEx) / (nline * nx)          # axes (t, x) -> (w, kx)
nw, nk = nline // 2 + 1, nx // 2 + 1
# fold to (w>0, kx>0): both propagation directions (w,k) and (w,-k)
Pwk = np.abs(F[:nw, :nk]) ** 2
Pwk[:, 1:] += np.abs(F[:nw, :-(nk):-1]) ** 2

# ---- snapshots ----------------------------------------------------------------
snap = np.fromfile(p + "snap.bin", dtype=np.float32)
nsnap = snap.size // (6 * ny * nx)
snap = snap[: nsnap * 6 * ny * nx].reshape(nsnap, 6, ny, nx)
ia, ib = int(round(args.ta / tsnap)), int(round(args.tb / tsnap))
ia, ib = min(ia, nsnap - 1), min(ib, nsnap - 1)
kx2 = np.fft.fftshift(np.fft.fftfreq(nx, d=Lx / nx)) * 2 * np.pi
ky2 = np.fft.fftshift(np.fft.fftfreq(ny, d=Ly / ny)) * 2 * np.pi

def p2d(a):
    return np.fft.fftshift(np.abs(np.fft.fft2(a - a.mean()) / (nx * ny)) ** 2)

def wna(a, kmax=8.0):
    P = p2d(a)
    KX, KY = np.meshgrid(kx2, ky2)
    k = np.hypot(KX, KY)
    sel = (k > 0.5) & (k < kmax)
    th = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))
    return (P[sel] * th[sel]).sum() / P[sel].sum()

ExA, ExB = snap[ia, 0], snap[ib, 0]
ByA, ByB = snap[ia, 4], snap[ib, 4]
x = np.arange(nx) * Lx / nx
y = np.arange(ny) * Ly / ny

# ---- figure 1 analog -----------------------------------------------------------
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

for col, (Ex2, By2, tt) in enumerate([(ExA, ByA, args.ta), (ExB, ByB, args.tb)]):
    dE = Ex2 - Ex2.mean()
    a = ax[0, 1 + col]
    vm = np.abs(dE).max()
    a.imshow(dE / vm, origin="lower", extent=[0, Ly, 0, Lx], cmap="bwr", vmin=-1, vmax=1,
             aspect="equal")
    a.set_title(rf"({'cd'[col]}) $\delta E_x$, t={tt:.0f}")
    a.set_xlabel(r"$y\,[c/\omega_{pe}]$"); a.set_ylabel(r"$x\,[c/\omega_{pe}]$")
    a = ax[1, 1 + col]
    Pe = p2d(dE)
    im = a.pcolormesh(ky2, kx2, Pe, norm=LogNorm(Pe.max() * 1e-5, Pe.max()),
                      cmap="rainbow", shading="auto")
    a.set_xlim(-15, 15); a.set_ylim(-15, 15)
    a.set_title(rf"({'ef'[col]}) $|\delta E_x(k_x,k_y)|^2$, t={tt:.0f}")
    a.set_xlabel(r"$k_y\,[\omega_{pe}/c]$"); a.set_ylabel(r"$k_x\,[\omega_{pe}/c]$")
    plt.colorbar(im, ax=a)

for col, (By2, tt) in enumerate([(ByA, args.ta), (ByB, args.tb)]):
    a = ax[col, 3]
    Pb2 = p2d(By2)
    im = a.pcolormesh(ky2, kx2, Pb2, norm=LogNorm(Pb2.max() * 1e-5, Pb2.max()),
                      cmap="rainbow", shading="auto")
    a.set_xlim(-15, 15); a.set_ylim(-15, 15)
    a.set_title(rf"({'gh'[col]}) $|\delta B_y(k_x,k_y)|^2$, t={tt:.0f}, "
                rf"WNA={wna(By2):.1f}$^\circ$")
    a.set_xlabel(r"$k_y\,[\omega_{pe}/c]$"); a.set_ylabel(r"$k_x\,[\omega_{pe}/c]$")
    plt.colorbar(im, ax=a)

fig.suptitle(rf"ArcWarden Yee-2D — Ma et al. PoP 2024 case 7 "
             rf"($\beta_\parallel$=0.0091, $T_\perp/T_\parallel$=5)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(args.outdir, "eaw_case7_fig1.png"), dpi=130)
print("wrote", os.path.join(args.outdir, "eaw_case7_fig1.png"))

# ---- figure 2: history + waveform (TDS) -----------------------------------------
h = np.loadtxt(p + "hist.txt")
dA = (Lx / nx) * (Ly / ny)
th = h[:, 0]
WB = 0.5 * (h[:, 4] + h[:, 5]) * dA            # transverse B energy (By,Bz)
WBx = 0.5 * h[:, 3] * dA
WEx = 0.5 * h[:, 1] * dA
Tpar = h[:, 7] / np_; Tperp = (h[:, 8] + h[:, 9]) / (2 * np_)
dB_B0 = np.sqrt((h[:, 3] + h[:, 4] + h[:, 5]) / (nx * ny)) / wce

fig2, bx = plt.subplots(1, 3, figsize=(16, 4.2))
a = bx[0]
a.semilogy(th, WB, label=r"$W_{B_y}+W_{B_z}$")
a.semilogy(th, WBx, label=r"$W_{B_x}$")
a.semilogy(th, WEx, label=r"$W_{E_x}$")
a.set_xlabel(r"$t\,[\omega_{pe}^{-1}]$"); a.set_ylabel("field energy")
a.legend(fontsize=8); a.grid(alpha=.3)
a.set_title(rf"growth & saturation ($\delta B/B_0$ max = {dB_B0.max():.3f})")
a2 = a.twinx(); a2.plot(th, Tperp / Tpar, "k--", lw=0.8)
a2.set_ylabel(r"$T_\perp/T_\parallel$")

a = bx[1]
kb = int(round(args.tb / tline))
a.plot(x, dEx[kb] / np.abs(dEx[kb]).max(), lw=0.7)
a.set_xlabel(r"$x\,[c/\omega_{pe}]$"); a.set_ylabel(r"$\delta E_x/|\delta E_x|_{max}$")
a.set_title(rf"$\delta E_x(x)$ at $t$={args.tb:.0f} (TDS check)"); a.grid(alpha=.3)

a = bx[2]
ka = int(round(args.ta / tline))
a.plot(x, dEx[ka] / max(np.abs(dEx[ka]).max(), 1e-30), lw=0.7)
a.set_xlabel(r"$x\,[c/\omega_{pe}]$")
a.set_title(rf"$\delta E_x(x)$ at $t$={args.ta:.0f} (pre-EAW)"); a.grid(alpha=.3)

fig2.tight_layout()
fig2.savefig(os.path.join(args.outdir, "eaw_case7_history.png"), dpi=130)
print("wrote", os.path.join(args.outdir, "eaw_case7_history.png"))
print(f"WNA(By) tA={args.ta:.0f}: {wna(ByA):.1f} deg   tB={args.tb:.0f}: {wna(ByB):.1f} deg")
print(f"dB/B0 at tB: {dB_B0[min(kb, len(dB_B0)-1)]:.4f}   max: {dB_B0.max():.4f}")
