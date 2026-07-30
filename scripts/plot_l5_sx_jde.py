#!/usr/bin/env python3
"""Generation-mechanism maps for l5_diag: Poynting S_x(x,t) + hot-J.E ledger.

Per frame (bline+eline+jline, every 100 steps = 3/We0):
  S_x     = Ey*Bz - Ez*By          wave energy flux (+x = northward)
  P_cyc   = Jy*Ey + Jz*Ez          cyclotron channel (hot J only)
  P_lan   = Jx*Ex                  Landau channel
Sign convention: P = J.E = work done BY fields ON particles.
  P < 0  => particles pump the wave  => GENERATION region.
Products oscillate at 2w; boxcar-smooth in t (~11 frames = 33/We0) and x.
Works mid-run (reads whatever frames exist).
"""
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "build/l5_diag"
meta = {}
for line in open(d + "/meta.txt"):
    k, v = line.split(None, 1)
    meta.setdefault(k, []).append(v.strip())
nx = int(meta["nx"][0]); dx = float(meta["dx"][0]); dt = float(meta["dt"][0])
wce = float(meta["wce"][0]); xc = float(meta["b0_xc"][0]); lre = float(meta["b0_lre"][0])
ble = int(meta["bline_every"][0])

frames = sorted(glob.glob(d + "/bline_*.bin"))
frames = [f for f in frames
          if os.path.exists(f.replace("bline", "eline"))
          and os.path.exists(f.replace("bline", "jline"))]
# drop possibly half-written last frame
frames = frames[:-1]
nf = len(frames)
DS = 10                              # x downsample
nxd = nx // DS
Sx = np.empty((nf, nxd), np.float32)
Pc = np.empty((nf, nxd), np.float32)
Pl = np.empty((nf, nxd), np.float32)
WB = np.empty((nf, nxd), np.float32)
for n, fb in enumerate(frames):
    b = np.fromfile(fb, dtype=np.float32, count=2 * nx)
    e = np.fromfile(fb.replace("bline", "eline"), dtype=np.float32, count=3 * nx)
    j = np.fromfile(fb.replace("bline", "jline"), dtype=np.float32, count=3 * nx)
    by, bz = b[:nx], b[nx:]
    ex, ey, ez = e[:nx], e[nx:2*nx], e[2*nx:]
    jx, jy, jz = j[:nx], j[nx:2*nx], j[2*nx:]
    def ds(a): return a[:nxd*DS].reshape(nxd, DS).mean(1)
    Sx[n] = ds(ey*bz - ez*by)
    Pc[n] = ds(jy*ey + jz*ez)
    Pl[n] = ds(jx*ex)
    WB[n] = ds(0.5*(by*by + bz*bz))
t = (np.arange(nf) + 1) * ble * dt * wce          # 1/We0
xl = ((np.arange(nxd) + 0.5) * DS + 0.5) * dx - xc

def lat_of_s(s):
    sgn = math.copysign(1.0, s); s = abs(s)
    lo, hi = 0.0, 40.0
    f = lambda lam: lre*0.5*(math.sin(math.radians(lam)) *
        math.sqrt(1+3*math.sin(math.radians(lam))**2) +
        math.asinh(math.sqrt(3)*math.sin(math.radians(lam)))/math.sqrt(3))
    for _ in range(50):
        mid = 0.5*(lo+hi)
        (lo, hi) = (mid, hi) if f(mid) < s else (lo, mid)
    return sgn*0.5*(lo+hi)
lats = np.array([lat_of_s(s) for s in xl])

def smooth_t(a, w=11):
    k = np.ones(w)/w
    return np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, a)
Sxs, Pcs, Pls = smooth_t(Sx), smooth_t(Pc), smooth_t(Pl)

def show(ax, A, ttl, vmax=None):
    v = vmax or np.percentile(np.abs(A), 99.5)
    im = ax.pcolormesh(t, lats, A.T, cmap="RdBu_r", vmin=-v, vmax=v, rasterized=True)
    for L in (25.07, -25.07): ax.axhline(L, ls="--", c="k", lw=0.6, alpha=0.6)
    ax.set_ylabel("latitude [deg]"); ax.set_title(ttl, fontsize=10)
    plt.colorbar(im, ax=ax, pad=0.01)

fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True, constrained_layout=True)
show(axes[0], np.log10(np.maximum(WB, 1e-12)) - 0, "log10 WB(x,t)  [reference]", None)
axes[0].collections[0].set_cmap("magma"); axes[0].collections[0].set_clim(-9, -4)
show(axes[1], Sxs, "S_x = EyBz-EzBy (smoothed): red = northward flux; source = red above / blue below")
show(axes[2], Pcs, "J.E cyclotron channel JyEy+JzEz (hot J): BLUE = generation")
show(axes[3], Pls, "J.E Landau channel JxEx: BLUE = generation")
axes[-1].set_xlabel("t [1/We0]")
fig.suptitle("l5_diag generation ledger (dashes: hybrid layer edge ±25°)", fontsize=12)
fig.savefig(d + "_sx_jde_map.png", dpi=115)

# time-averaged latitude profiles in three windows
wins = [(1000, 3000, "flood 1000-3000"), (4500, 6500, "lull 4500-6500"),
        (6800, t[-1], f"complex 6800-{t[-1]:.0f}")]
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False, constrained_layout=True)
for ax, (t0, t1, lb) in zip(axes2, wins):
    m = (t >= t0) & (t <= t1)
    ax.plot(lats, Pc[m].mean(0), "b-", lw=1.3, label="cyclotron JyEy+JzEz")
    ax.plot(lats, Pl[m].mean(0), "r-", lw=1.0, label="Landau JxEx")
    ax.axhline(0, c="k", lw=0.6)
    for L in (25.07, -25.07): ax.axvline(L, ls="--", c="gray", lw=0.7)
    ax.set_title(lb, fontsize=10); ax.set_xlabel("latitude [deg]"); ax.grid(alpha=0.3)
axes2[0].set_ylabel("<J.E>  (neg = generation)"); axes2[0].legend(fontsize=8)
fig2.suptitle("time-averaged J.E latitude profiles", fontsize=11)
fig2.savefig(d + "_jde_profiles.png", dpi=115)
print(f"{nf} frames, t up to {t[-1]:.0f}/We0 -> {d}_sx_jde_map.png + _jde_profiles.png")
