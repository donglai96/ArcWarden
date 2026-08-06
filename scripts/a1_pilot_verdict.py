#!/usr/bin/env python3
"""A1 pilot verdict (PLAN_TWO_TRACK v2.1 execution step 5): triggered
(a05/a10) vs untriggered same-seed baseline (d120_ctrl, ant_amp=0).

Pilot questions:
  1. Does the calibrated 0.25 We0 trigger ADVANCE and/or ORGANIZE the
     element vs the self-igniting baseline? (onset times per probe, from
     band-limited envelope crossing an ABSOLUTE floor — 20%-of-own-max
     would move with each run's max and fake alignment)
  2. Does the triggered ridge reach the Track-A ctrl bar w_max >= 0.55?
     (ridge quantiles + max over strong frames, nwin=1024)
  3. Element-scale amplitude preserved? (peak band-limited envelope)

All windows in t*We0. The trigger packet itself (dB/B0 ~ 7e-4-1.5e-3,
t < ~60 We0 near the equator) sits BELOW the element floor 3e-3, so the
onset detector does not fire on the trigger passage itself.

Usage: python3 a1_pilot_verdict.py <ctrl_dir> <a05_dir> <a10_dir> [tmax_Oe]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import load_probe, stft

ELEMENT_FLOOR = 3e-3     # |B_w|/B0, band 0.15-0.55: element-scale, > trigger
PROBES = (0.0, 145.2, -145.2, 291.0, -291.0)


def band_env(t, by, bz, wce, f1=0.15, f2=0.55):
    z = by + 1j * bz
    n = len(z)
    F = np.fft.fft(z)
    f = np.fft.fftfreq(n, d=t[1] - t[0]) * 2 * np.pi / wce
    F[(f < f1) | (f > f2)] = 0.0
    zb = np.fft.ifft(F)
    env = np.abs(zb)
    nbox = max(1, int(round(60.0 / wce / (t[1] - t[0]))))   # 60/We0 boxcar
    return np.convolve(env, np.ones(nbox) / nbox, mode="same")


def run_stats(d, tmax=None):
    out = {"dir": d, "onset": {}, "peak": {}}
    for off in PROBES:
        m, t, by, bz = load_probe(d, off)
        wce = m["wce"]
        tOe = t * wce
        if tmax is not None:
            keep = tOe <= tmax
            t, by, bz, tOe = t[keep], by[keep], bz[keep], tOe[keep]
        env = band_env(t, by, bz, wce) / wce      # /B0
        above = np.nonzero(env > ELEMENT_FLOOR)[0]
        out["onset"][off] = tOe[above[0]] if len(above) else np.nan
        out["peak"][off] = float(env.max())
    # ridge stats at the equator
    m, t, by, bz = load_probe(d, 0.0)
    wce = m["wce"]
    if tmax is not None:
        keep = t * wce <= tmax
        t, by, bz = t[keep], by[keep], bz[keep]
    S = stft(by + 1j * bz, 1024, 64)
    f = np.fft.fftfreq(1024, d=t[1] - t[0]) * 2 * np.pi / wce
    sel = (f > 0.1) & (f < 0.95)
    fs = f[sel]
    P = np.abs(S[:, sel]) ** 2
    rf, rp = fs[P.argmax(axis=1)], P.max(axis=1)
    strong = rp > 0.05 * rp.max()
    out["wmax"] = float(rf[strong].max())
    out["wq"] = np.quantile(rf[strong], [0.5, 0.9, 0.99])
    out["spec"] = (S, f, t, wce)
    return out


def main(dirs, tmax=None):
    runs = [run_stats(d, tmax) for d in dirs]
    hdr = f"{'probe':>8s}" + "".join(f"{r['dir']:>18s}" for r in runs)
    print("# onset t*We0 (band 0.15-0.55 envelope crosses "
          f"{ELEMENT_FLOOR:g} B0):")
    print(hdr)
    for off in PROBES:
        print(f"{off:8.1f}" + "".join(f"{r['onset'][off]:18.0f}" for r in runs))
    print("# peak band-limited |Bw|/B0:")
    print(hdr)
    for off in PROBES:
        print(f"{off:8.1f}" + "".join(f"{r['peak'][off]:18.2e}" for r in runs))
    print("# equator ridge:")
    for r in runs:
        q = r["wq"]
        print(f"  {r['dir']:>12s}: wmax {r['wmax']:.3f}   "
              f"q50/90/99 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}   "
              f"vs bar 0.55 -> {'PASS' if r['wmax'] >= 0.55 else 'FAIL'}")

    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.6 * len(runs)),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        S, f, t, wce = r["spec"]
        tt = (t[512] + np.arange(S.shape[0]) * 64 * (t[1] - t[0])) * wce
        sel = (f > 0.05) & (f < 0.9)
        P = np.log10(np.abs(S[:, sel]).T ** 2 + 1e-30)
        im = ax.pcolormesh(tt, f[sel], P, cmap="turbo",
                           vmin=P.max() - 5, vmax=P.max(), shading="auto")
        ax.axhline(0.5, color="w", lw=2)
        ax.axhline(0.5, color="k", lw=1, ls="--")
        ax.axhline(0.55, color="w", lw=0.8, ls=":")
        on = r["onset"][0.0]
        if np.isfinite(on):
            ax.axvline(on, color="magenta", lw=1.2, ls="--")
        ax.set_ylabel(r"$\omega/\Omega_e$")
        ax.set_title(f"{r['dir']}   wmax={r['wmax']:.3f}   eq onset="
                     f"{on:.0f}/We0", fontsize=10)
        plt.colorbar(im, ax=ax, label="log10 P")
    np.atleast_1d(axes)[-1].set_xlabel(r"$t\,\Omega_e$")
    fig.savefig("a1_pilot_verdict.png", dpi=130)
    print("fig -> a1_pilot_verdict.png")


if __name__ == "__main__":
    tmax = float(sys.argv[4]) if len(sys.argv) > 4 else None
    main(sys.argv[1:4], tmax)
