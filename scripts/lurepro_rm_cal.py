#!/usr/bin/env python3
"""x10 R_m calibration verdict (PLAN_LUMORPH A5.4, exec-order step 4).

R_m = 2 rms|B1| / rms|B0w| over the source x-window |s| <= 175 (eq..7.5°),
per bline/m1line dump; quoted in the element-active window (m0 window-rms
> 0.3 of its run max).  Same rms convention as the x4 ladder measurement
(k16/k24/k32 -> 0.97/0.44/0.19) so numbers are comparable — but the x4
mapping itself does NOT carry; that is why this run exists.
MAIN-case rule: pick k1 with R_m ~ 0.2-0.4; R_m ~ 1 = mode competition.

Usage: python3 lurepro_rm_cal.py <dir> [<dir> ...]
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import read_meta

SWIN = 175.1


def series(d):
    m = read_meta(d)
    nx, dx, xc = int(m["nx"]), m["dx"], m["b0_xc"]
    x = np.arange(nx) * dx - xc
    w = np.abs(x) <= SWIN
    fb = sorted(glob.glob(os.path.join(d, "bline_*.bin")))
    t, r0, r1 = [], [], []
    for f in fb:
        n = int(f[-10:-4])
        fm = os.path.join(d, f"m1line_{n:06d}.bin")
        if not os.path.exists(fm):
            continue
        b = np.fromfile(f, dtype=np.float32)
        by, bz = b[:nx], b[nx:2 * nx]
        c = np.fromfile(fm, dtype=np.complex64)
        b1y, b1z = c[:nx], c[nx:2 * nx]
        t.append(n * m["bline_every"] * m["dt"] * m["wce"])
        r0.append(np.sqrt(np.mean(by[w] ** 2 + bz[w] ** 2)))
        r1.append(np.sqrt(np.mean(np.abs(b1y[w]) ** 2 + np.abs(b1z[w]) ** 2)))
    return np.array(t), np.array(r0), np.array(r1)


def main(dirs):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                           constrained_layout=True)
    print(f"{'dir':32s} {'m0 peak':>9s} {'2|B1| pk':>9s} "
          f"{'R_m med':>8s} {'R_m max':>8s}  window")
    for d in dirs:
        t, r0, r1 = series(d)
        rm = 2 * r1 / (r0 + 1e-30)
        act = r0 > 0.3 * r0.max()
        med = float(np.median(rm[act]))
        mx = float(rm[act].max())
        t0, t1 = t[act][0], t[act][-1]
        lab = os.path.basename(d.rstrip("/"))
        print(f"{lab:32s} {r0.max():9.2e} {2*r1.max():9.2e} "
              f"{med:8.3f} {mx:8.3f}  t[{t0:.0f},{t1:.0f}]")
        ax[0].plot(t, r0, label=f"{lab} m0")
        ax[0].plot(t, 2 * r1, "--", label=f"{lab} 2|B1|")
        ax[1].plot(t, rm, label=f"{lab} (med {med:.2f})")
        ax[1].axvspan(t0, t1, alpha=0.06)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("window rms (B units)")
    ax[0].legend(fontsize=7)
    ax[1].axhspan(0.2, 0.4, color="g", alpha=0.15, label="target 0.2-0.4")
    ax[1].set_ylim(0, 1.5)
    ax[1].set_xlabel(r"$t\,\Omega_e$")
    ax[1].set_ylabel(r"$R_m = 2\,\mathrm{rms}|B_1| / \mathrm{rms}|B_{0w}|$")
    ax[1].legend(fontsize=8)
    ax[0].set_title(f"x10 R_m calibration, |s|<={SWIN:.0f} (eq..7.5deg), "
                    "element window = m0 rms > 0.3 max")
    fig.savefig("lurepro_rm_cal.png", dpi=130)
    print("fig -> lurepro_rm_cal.png")


if __name__ == "__main__":
    main(sys.argv[1:])
