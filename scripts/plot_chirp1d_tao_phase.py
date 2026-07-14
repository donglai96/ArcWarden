#!/usr/bin/env python3
"""Tao GRL17-style phase-space analysis of a chirp1d run (Figs 2 & 3 analogs):
  (1) delta-f(u_par, u_perp) near the equator with the cyclotron resonance curve;
  (2) marker-count and delta-f histograms in (zeta, u_par) at fixed u_perp,
      zeta = gyrophase of u_perp measured from the LOCAL wave B_perp direction
      (phase dumps carry By,Bz,Ey,Ez interpolated to each marker).
Phase records: 9 floats = x-xc, ux, uy, uz, wd, Byp, Bzp, Eyp, Ezp.

Usage: plot_chirp1d_tao_phase.py <prefix> --times t1,t2,t3 [--hband lo,hi]
       [--uperp lo,hi] [--wpe W] [--out out.png]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_dumps(path, rec=9):
    raw = np.fromfile(path, dtype=np.float32)
    recs, i = [], 0
    while i + 2 <= len(raw):
        t0, npk = raw[i], int(raw[i + 1])
        i += 2
        if npk <= 0 or i + rec * npk > len(raw):
            break
        recs.append((float(t0), raw[i:i + rec * npk].reshape(npk, rec)))
        i += rec * npk
    return recs


def cold_whistler_k(w, wpe, wce=1.0):
    """ck for the R-mode at frequency w (cold), returns k > 0."""
    n2 = 1.0 + wpe**2 / (w * (wce - w))
    return w * np.sqrt(n2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--times", default="1200,1400,1600")
    ap.add_argument("--hband", default="-15,15", help="h range for equator panels")
    ap.add_argument("--uperp", default="0.55,0.75", help="u_perp slice for zeta plots")
    ap.add_argument("--wpe", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    times = [float(v) for v in args.times.split(",")]
    h0, h1 = (float(v) for v in args.hband.split(","))
    up0, up1 = (float(v) for v in args.uperp.split(","))

    recs = read_dumps(args.prefix + "_phase.bin")
    print("dumps at t =", [f"{t:.0f}" for t, _ in recs])
    picks = []
    for tw in times:
        j = int(np.argmin([abs(t - tw) for t, _ in recs]))
        picks.append(recs[j])

    ncol = len(picks)
    fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol, 11.5), squeeze=False)

    for c, (t0, d) in enumerate(picks):
        h, ux, uy, uz, wd = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
        Byp, Bzp = d[:, 5], d[:, 6]
        up = np.hypot(uy, uz)
        eq = (h > h0) & (h < h1)

        # ---- row 0: delta-f(u_par, u_perp) at the equator ----
        ax = axes[0, c]
        H, xe, ye = np.histogram2d(ux[eq], up[eq], bins=[140, 120],
                                   range=[[-0.6, 0.6], [0, 1.2]], weights=wd[eq])
        lim = np.percentile(np.abs(H), 99.5) or 1
        im = ax.pcolormesh(xe, ye, H.T, cmap="RdBu_r", vmin=-lim, vmax=lim)
        # cyclotron resonance curves for a few frequencies in the element band
        for w, ls in [(0.3, ":"), (0.45, "--")]:
            k = cold_whistler_k(w, args.wpe)
            upg = np.linspace(0, 1.2, 100)
            # v_par = (w - wce/gamma)/k with gamma(u_par~vR, u_perp)
            vr = np.zeros_like(upg)
            for i, u_ in enumerate(upg):
                v = -0.2
                for _ in range(40):
                    g = np.sqrt(1 + (v * 1.0)**2 + u_**2)  # u_par ~ gamma v
                    v = (w - 1.0 / g) / k
                vr[i] = v
            g = 1.0 / np.sqrt(1 - np.clip(vr**2 + (upg / np.sqrt(1 + upg**2))**2, 0, 0.99))
            ax.plot(vr * g, upg, "k" + ls, lw=1.0,
                    label=fr"$v_R(\omega={w})$" if c == 0 else None)
        ax.set_title(fr"$t\,\Omega_{{e0}}={t0:.0f}$")
        ax.set_xlabel(r"$u_\parallel/c$"); ax.set_ylabel(r"$u_\perp/c$")
        if c == 0: ax.legend(fontsize=7, loc="upper left")
        plt.colorbar(im, ax=ax, label=r"$\delta f$ (arb)")

        # ---- rows 1-2: (zeta, u_par) at fixed u_perp slice ----
        sl = eq & (up > up0) & (up < up1)
        phase_u = np.arctan2(uz[sl], uy[sl])
        phase_B = np.arctan2(Bzp[sl], Byp[sl])
        zeta = np.mod(phase_u - phase_B, 2 * np.pi)

        ax = axes[1, c]
        H, xe, ye = np.histogram2d(zeta / np.pi, ux[sl], bins=[64, 90],
                                   range=[[0, 2], [-0.45, 0.15]])
        im = ax.pcolormesh(xe, ye, H.T, cmap="Blues")
        ax.set_xlabel(r"$\zeta/\pi$"); ax.set_ylabel(r"$u_\parallel/c$")
        ax.set_title(f"marker count, $u_\\perp\\in[{up0},{up1}]$", fontsize=9)
        plt.colorbar(im, ax=ax, label="count")

        ax = axes[2, c]
        Hw, xe, ye = np.histogram2d(zeta / np.pi, ux[sl], bins=[64, 90],
                                    range=[[0, 2], [-0.45, 0.15]], weights=wd[sl])
        lim = np.percentile(np.abs(Hw), 99.5) or 1
        im = ax.pcolormesh(xe, ye, Hw.T, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xlabel(r"$\zeta/\pi$"); ax.set_ylabel(r"$u_\parallel/c$")
        ax.set_title(r"$\delta f(\zeta, u_\parallel)$ — hole = blue", fontsize=9)
        plt.colorbar(im, ax=ax, label=r"$\delta f$ (arb)")

    fig.suptitle(f"chirp1d Tao-GRL17-style phase space  (h in [{h0},{h1}] $c/\\Omega_{{e0}}$)")
    fig.tight_layout()
    out = args.out or args.prefix + "_tao_phase.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
