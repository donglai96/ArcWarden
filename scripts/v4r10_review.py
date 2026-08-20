#!/usr/bin/env python3
"""V4R10 mid-run review (user 2026-08-19: "waves look weird, not chirping
elements — probe position? shell length? need zoom + animation of wave
generation").

Products (all into build/v4r10/):
  review_growth.png    W_EM(t) v4r9 vs v4r10 + gamma fits + saturation ETA
  review_powerL.png    equatorial By^2 vs L at several times, probe lines marked
  review_hovmoller.png B1(lambda, t) along the L0 line from sline_*.bin
  review_wavemap.gif   zoomed By animation (per-frame normalized), shell contours
Usage: v4r10_review.py [outdir9 outdir10]
"""
import csv
import glob
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

d9 = sys.argv[1] if len(sys.argv) > 2 else "build/v4r9"
d10 = sys.argv[2] if len(sys.argv) > 2 else "build/v4r10"

meta = open(f"{d10}/meta.txt").read()
m = re.search(r"box ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) "
              r"(\d+) (\d+) ([-\d.e+]+) ([-\d.e+]+)", meta)
x0, x1, z0, z1 = (float(m.group(i)) for i in range(1, 5))
nx, nz = int(m.group(5)), int(m.group(6))
dx, dz = float(m.group(7)), float(m.group(8))
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
L0s, dLs, edges = 4000.0, 90.0, 15.0
ms = re.search(r"shell \S+ L0=([\d.]+) dL=([\d.]+) edge=([\d.]+)", meta)
if ms:
    L0s, dLs, edges = float(ms.group(1)), float(ms.group(2)), float(ms.group(3))

def read_energy(d):
    t, w = [], []
    with open(f"{d}/energy.csv") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            w.append(float(row["W_EM"]))
    return np.array(t), np.array(w)

# ---- 1. growth comparison + extrapolation --------------------------------
t9, w9 = read_energy(d9)
t10, w10 = read_energy(d10)
# de-dup any resume overlap (keep first occurrence)
keep = np.concatenate([[True], np.diff(t10) > 0])
t10, w10 = t10[keep], w10[keep]

def gfit(t, w, ta, tb):
    m = (t >= ta) & (t <= tb) & (w > 0)
    p = np.polyfit(t[m], np.log(w[m]), 1)
    return p[0], p  # energy-rate gamma_W

g9, _ = gfit(t9, w9, 12000, 20000)
g10, p10 = gfit(t10, w10, 15000, 21000)
# v4r9 saturation-entry reference: W at t ~ 23250 (step 155k)
w9_sat = np.interp(23250.0, t9, w9)
t_now, w_now = t10[-1], w10[-1]
t_sat10 = t_now + np.log(w9_sat / w_now) / g10
step_sat10 = t_sat10 / dt

fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(t9, w9, "C0", label=f"v4r9 (dL=60)  $\\gamma_W$={g9:.2e}")
ax.semilogy(t10, w10, "C3", label=f"v4r10 (dL=90)  $\\gamma_W$={g10:.2e}")
tt = np.linspace(t_now, t_sat10 + 2000, 100)
ax.semilogy(tt, np.exp(np.polyval(p10, tt)), "C3--", alpha=0.6,
            label=f"v4r10 extrapolation → sat-entry W={w9_sat:.1e}\n"
                  f"at t≈{t_sat10:.0f} (step ≈ {step_sat10/1000:.0f}k)")
ax.axhline(w9_sat, color="k", ls=":", lw=0.8)
ax.axvline(29250, color="gray", ls=":", lw=0.8)
ax.text(29250, 1e-8, " planned stop 195k", rotation=90, fontsize=8)
ax.axvline(35250, color="gray", ls=":", lw=0.8)
ax.text(35250, 1e-8, " resume target 235k", rotation=90, fontsize=8)
ax.set_xlabel("t [1/ωpe]"); ax.set_ylabel("W_EM")
ax.set_title("v4r9 vs v4r10 growth — same deck family, shell dL 60→90")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{d10}/review_growth.png", dpi=110)
print(f"gamma_W  v4r9 {g9:.3e}  v4r10 {g10:.3e}  ratio {g9/g10:.2f}")
print(f"v4r9 sat-entry W(t=23250) = {w9_sat:.2e}")
print(f"v4r10 now W({t_now:.0f}) = {w_now:.2e} → sat-entry at t≈{t_sat10:.0f} "
      f"= step {step_sat10:.0f}")

# ---- 2. equatorial power vs L, several times -----------------------------
snaps = sorted(glob.glob(f"{d10}/f2d_*.bin"))
pick = snaps[-1::-4][::-1][-4:]
fig, ax = plt.subplots(figsize=(10, 6))
kc = int(round((0.0 - z0) / dz - 0.5))
rows = slice(kc - 20, kc + 21)          # |z| < 5 → |lam| < 0.08 deg
for s in pick:
    n = int(re.search(r"f2d_(\d+)", s).group(1))
    a = np.memmap(s, dtype=np.float32, mode="r", shape=(6, nz, nx))
    by = np.asarray(a[4, rows, :], dtype=np.float64)
    p = (by ** 2).mean(axis=0)
    xs = x0 + (np.arange(nx) + 0.5) * dx      # z=0: L = x exactly
    ax.semilogy(xs, np.convolve(p, np.ones(11) / 11, "same"),
                label=f"step {n//1000}k", alpha=0.85)
for L, c, lab in [(4000, "k", "probe L0"), (4030, "gray", "probe L2")]:
    ax.axvline(L, color=c, ls="--", lw=1.2)
    ax.text(L, ax.get_ylim()[0], f" {lab}", rotation=90, fontsize=8)
for e in (L0s - dLs / 2, L0s + dLs / 2):
    ax.axvline(e, color="C2", ls=":", lw=1)
for e in (L0s - dLs / 2 - 3 * edges, L0s + dLs / 2 + 3 * edges):
    ax.axvline(e, color="C2", ls=":", lw=0.6, alpha=0.6)
ax.set_xlim(3700, 4300)
ax.set_xlabel("x = L at equator [c/ωpe]")
ax.set_ylabel("⟨By²⟩ (|z|<5)")
ax.set_title("v4r10 equatorial wave power vs L — shell flat-top / ±3σ (green), "
             "probe lines (dashed)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{d10}/review_powerL.png", dpi=110)
# ridge quote
a = np.memmap(snaps[-1], dtype=np.float32, mode="r", shape=(6, nz, nx))
by = np.asarray(a[4, rows, :], dtype=np.float64)
p = np.convolve((by ** 2).mean(axis=0), np.ones(11) / 11, "same")
xs = x0 + (np.arange(nx) + 0.5) * dx
band = (xs > 3800) & (xs < 4200)
xr = xs[band][np.argmax(p[band])]
print(f"latest eq power ridge at L≈{xr:.0f} "
      f"(power there / power at 4000 = {p[band].max() / p[np.argmin(np.abs(xs-4000))]:.1f})")

# ---- 3. Hovmoller B1(lambda, t) from slines ------------------------------
sls = sorted(glob.glob(f"{d10}/sline_*.bin"))
ns_line = int(re.search(r"line ns (\d+)", meta).group(1))
lam_w = float(re.search(r"lam_w_deg ([\d.]+)", meta).group(1))
lams = np.linspace(-lam_w, lam_w, ns_line)
H, tsl = [], []
for s in sls:
    n = int(re.search(r"sline_(\d+)", s).group(1))
    a = np.fromfile(s, dtype=np.float32).reshape(6, ns_line)
    H.append(a[4])                          # B1
    tsl.append(n * dt)
H = np.array(H)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
im = a1.pcolormesh(lams, tsl, np.log10(np.abs(H) + 1e-12), cmap="turbo",
                   vmin=-8, vmax=np.log10(np.abs(H).max()))
a1.set_xlabel("latitude λ [deg]"); a1.set_ylabel("t [1/ωpe]")
a1.set_title("log10 |B1| along L0 line (sline)")
fig.colorbar(im, ax=a1)
Hn = H / (np.abs(H).max(axis=1, keepdims=True) + 1e-20)
im2 = a2.pcolormesh(lams, tsl, Hn, cmap="RdBu_r", vmin=-1, vmax=1)
a2.set_xlabel("latitude λ [deg]")
a2.set_title("B1 per-time normalized (phase fronts / generation region)")
fig.colorbar(im2, ax=a2)
fig.tight_layout(); fig.savefig(f"{d10}/review_hovmoller.png", dpi=110)

# ---- 4. zoom animation of By over the source region ----------------------
ds = 2
xa = x0 + (np.arange(nx)[::ds] + 0.5) * dx
za = z0 + (np.arange(nz)[::ds] + 0.5) * dz
xi = (xa > 3600) & (xa < 4300)
zi = (za > -900) & (za < 900)
Xg, Zg = np.meshgrid(xa[xi], za[zi])
Lg = (Xg ** 2 + Zg ** 2) ** 1.5 / Xg ** 2   # dipole2d L = r^3/x^2

fig, ax = plt.subplots(figsize=(8, 9))
frames = []
for s in snaps:
    n = int(re.search(r"f2d_(\d+)", s).group(1))
    a = np.memmap(s, dtype=np.float32, mode="r", shape=(6, nz, nx))
    by = np.asarray(a[4, ::ds, ::ds][np.ix_(zi, xi)], dtype=np.float32)
    frames.append((n, by))
qm = [None]
def draw(i):
    n, by = frames[i]
    ax.clear()
    v = np.percentile(np.abs(by), 99.5) + 1e-12
    ax.pcolormesh(xa[xi], za[zi], by, cmap="RdBu_r", vmin=-v, vmax=v)
    for Lc, st in [(L0s - dLs / 2, "-"), (L0s + dLs / 2, "-"),
                   (L0s - dLs / 2 - 3 * edges, ":"),
                   (L0s + dLs / 2 + 3 * edges, ":")]:
        ax.contour(xa[xi], za[zi], Lg, levels=[Lc], colors="k",
                   linewidths=0.7, linestyles=st)
    ax.set_title(f"By, step {n//1000}k  t={n*dt:.0f}  (scale ±{v:.1e})")
    ax.set_xlabel("x"); ax.set_ylabel("z")
anim = FuncAnimation(fig, draw, frames=len(frames))
anim.save(f"{d10}/review_wavemap.gif", writer=PillowWriter(fps=2), dpi=80)
print("wrote review_growth/powerL/hovmoller.png + review_wavemap.gif")
