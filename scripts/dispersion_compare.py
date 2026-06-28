#!/usr/bin/env python3
"""Compare bump-on-tail omega-k spectra across beam velocities v_b.

Runs the same analysis as dispersion_bump.py on several run folders and lays the
spectra out in one figure, so you can see the power ridge (and the measured
phase velocity) track the beam line omega = v_b * k as v_b changes.

Usage:
    python3 scripts/dispersion_compare.py [folder:vb ...] [--vth 0.2] [--out PATH]

Defaults to the four runs: build/bump_field (v_b=1.0) + build/bump_vb0{8,6,4}.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dispersion_bump import load_spectrum, find_peak   # noqa: E402

DEFAULT_RUNS = [
    ("build/bump_field", 1.0),
    ("build/bump_vb08", 0.8),
    ("build/bump_vb06", 0.6),
    ("build/bump_vb04", 0.4),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="folder:vb pairs (override the defaults)")
    ap.add_argument("--vth", type=float, default=0.2, help="bulk thermal (Bohm-Gross)")
    ap.add_argument("--wpe", type=float, default=1.0)
    ap.add_argument("--out", default="build/dispersion_compare.png")
    args = ap.parse_args()

    runs = ([(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) for s in args.runs]
            if args.runs else DEFAULT_RUNS)

    n = len(runs)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 5.5 * nrow),
                             squeeze=False)
    kline = np.linspace(0.0, 4.0, 200)

    print(f"{'v_b':>5} {'k':>8} {'omega':>8} {'v_phase':>8} {'|dv|/v_b':>9}")
    print("-" * 42)
    rows = []
    for ax, (folder, vb) in zip(axes.flat, runs):
        path = folder if os.path.exists(os.path.join(folder, "field_xt.csv")) else folder
        if not os.path.exists(os.path.join(folder, "field_xt.csv")):
            ax.set_title(f"{folder}: field_xt.csv missing"); ax.axis("off"); continue
        kk, om, Z = load_spectrum(folder)
        k_pk, w_pk = find_peak(kk, om, Z)
        vph = w_pk / k_pk
        rel = abs(vph - vb) / vb
        rows.append((vb, k_pk, w_pk, vph, rel))
        print(f"{vb:5.2f} {k_pk:8.3f} {w_pk:8.3f} {vph:8.3f} {100*rel:8.1f}%")

        pm = ax.pcolormesh(kk, om, np.log10(Z + Z.max() * 1e-6), shading="auto",
                           cmap="magma")
        ax.plot(kline, vb * kline, "c--", lw=1.4, label=rf"$\omega=v_b k$ ($v_b$={vb})")
        ax.plot(kline, np.sqrt(args.wpe**2 + 3.0 * (kline * args.vth)**2), "w:",
                lw=1.1, label="Bohm-Gross")
        ax.plot([k_pk], [w_pk], "o", mfc="none", mec="lime", mew=1.8, ms=12)
        ax.set_xlim(0, 4.0); ax.set_ylim(0, 3.0)
        ax.set_xlabel("k"); ax.set_ylabel(r"$\omega$")
        ax.set_title(rf"$v_b$={vb}:  peak $v_\phi$={vph:.2f}  ({100*rel:.0f}% off)")
        ax.legend(loc="upper left", fontsize=8)

    for ax in axes.flat[len(runs):]:
        ax.axis("off")

    fig.suptitle("Bump-on-tail: electrostatic-wave phase velocity tracks the beam "
                 r"velocity ($v_\phi \approx v_b$)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")

    # trend check: v_phase should rise monotonically with v_b
    if len(rows) >= 2:
        vbs = [r[0] for r in rows]; vphs = [r[3] for r in rows]
        order = np.argsort(vbs)
        mono = all(np.diff(np.array(vphs)[order]) > 0)
        print(f"v_phase increases with v_b: {'YES' if mono else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
