#!/usr/bin/env python3
"""Paired ctrl/RSM comparison figure (diagnostic, not a gate).

Left column: 5° (pre-registered hemisphere) spectrograms, ctrl top / rsm
bottom, shared color scale, dominant-component ridge overlaid, 0.5 local
line bold.  Right: Welch band spectra in the CTRL element window (both
arms, local-fce barrier/LB bands shaded) + per-band ratio.

Usage: python3 lurepro_pair_fig.py <ctrl> <rsm> [tag]
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec
from lurepro_source_verdict import probe_pair, direction
from lurepro_ridge_audit import components, dominant_riser, S5
from plot_chen2026_fig1 import read_meta


def main(ctrl, rsm, tag=""):
    m0 = read_meta(ctrl)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    b5 = 1 + 4.5 * (S5 / lre) ** 2
    dom = direction(ctrl)

    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1])
    axL = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    axR = fig.add_subplot(gs[:, 1])

    data = {}
    vmax = -np.inf
    for d in (ctrl, rsm):
        m, t, by, bz, *_ = probe_pair(d, dom * S5)
        z = by + 1j * bz
        P, fs, tt = spec(z, t, wce, 1024)
        dm = dominant_riser(components(P, fs, tt))
        data[d] = (z, t, P, fs, tt, dm)
        vmax = max(vmax, np.log10(P.max() + 1e-30))

    dmc = data[ctrl][5]
    wname = f"ctrl element t[{dmc['t0']:.0f},{dmc['t1']:.0f}]"
    for ax, d, name in zip(axL, (ctrl, rsm), ("ctrl", "rsm")):
        z, t, P, fs, tt, dm = data[d]
        Pl = np.log10(P.T + 1e-30)
        im = ax.pcolormesh(tt, fs, Pl, cmap="turbo", vmin=vmax - 5,
                           vmax=vmax, shading="auto")
        if dm is not None:
            rt, rf = dm["ridge"]
            ax.plot(rt, rf, "m.", ms=1.5)
            ax.text(0.99, 0.95,
                    f"{name}: birth {dm['birth']:.3f} end {dm['end']:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    color="w", fontsize=10, fontweight="bold")
        ax.axhline(0.5 * b5, color="w", lw=2)
        ax.axhline(0.55, color="w", lw=0.8, ls=":")
        ax.axvspan(dmc["t0"], dmc["t1"], color="w", alpha=0.06)
        ax.set_ylim(0.1, 0.95)
        ax.set_ylabel(rf"{name}  $\omega/\Omega_{{e,eq}}$")
    axL[0].set_title(f"5° probe ({'SOUTH' if dom < 0 else 'NORTH'} "
                     f"pre-reg), bold = 0.5 local ({0.5*b5:.3f})")
    axL[1].set_xlabel(r"$t\,\Omega_e$")
    fig.colorbar(im, ax=axL, label="log10 P", shrink=0.8)

    # Welch spectra in ctrl element window
    spectra = {}
    for d, name, c in ((ctrl, "ctrl", "k"), (rsm, "rsm", "r")):
        z, t = data[d][0], data[d][1]
        w = (t * wce > dmc["t0"]) & (t * wce < dmc["t1"] + 100)
        fr, Pa = welch(z[w].real, fs=1 / (t[1] - t[0]), nperseg=4096)
        _, Pb = welch(z[w].imag, fs=1 / (t[1] - t[0]), nperseg=4096)
        fo = fr * 2 * np.pi / wce
        sel = (fo > 0.1) & (fo < 0.95)
        spectra[name] = (fo[sel], (Pa + Pb)[sel])
        axR.semilogx((Pa + Pb)[sel], fo[sel], c, lw=1.2, label=name)
    fo, Pc = spectra["ctrl"]
    _, Pr = spectra["rsm"]
    axR.axhspan(0.46 * b5, 0.56 * b5, color="orange", alpha=0.2,
                label="barrier (local)")
    axR.axhspan(0.25 * b5, 0.45 * b5, color="g", alpha=0.12,
                label="LB (local)")
    axR.axhline(0.5 * b5, color="k", lw=2)
    axR.set_ylim(0.1, 0.95)
    axR.set_ylabel(r"$\omega/\Omega_{e,eq}$")
    axR.set_xlabel("Welch P (ctrl element window)")
    bb = (fo >= 0.46 * b5) & (fo < 0.56 * b5)
    lb = (fo >= 0.25 * b5) & (fo < 0.45 * b5)
    Rb, Rl = Pr[bb].mean() / Pc[bb].mean(), Pr[lb].mean() / Pc[lb].mean()
    axR.set_title(f"R_barrier {Rb:.2f}  R_LB {Rl:.2f}\n"
                  f"S_A = {Rb/Rl:.3f}  ({wname})")
    axR.legend(fontsize=8, loc="lower right")

    out = f"lurepro_pair{('_' + tag) if tag else ''}_fig.png"
    fig.savefig(out, dpi=130)
    print(f"fig -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
