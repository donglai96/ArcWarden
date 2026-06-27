#!/usr/bin/env python3
"""Visualize the two-stream instability phase space dumped by test_two_stream.

The test (tests/test_two_stream.cu) writes `two_stream_phase.csv` with columns
`x,vx` — the particle positions and x-velocities at saturation. At saturation the
two counter-streaming beams have rolled up into phase-space vortices (BGK holes),
the classic signature of the nonlinear two-stream instability.

Usage:
    python3 scripts/plot_two_stream.py [csv] [-o out.png] [--show]

Defaults: reads ./two_stream_phase.csv (or build/two_stream_phase.csv), writes a
PNG next to it. Requires numpy + matplotlib.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe default; overridden by --show
import matplotlib.pyplot as plt


def find_default_csv() -> str:
    """Pick the first existing default CSV location."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for candidate in (
        os.path.join(root, "two_stream_phase.csv"),
        os.path.join(root, "build", "two_stream_phase.csv"),
        "two_stream_phase.csv",
    ):
        if os.path.exists(candidate):
            return candidate
    return os.path.join(root, "two_stream_phase.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=None,
                    help="phase-space CSV (default: two_stream_phase.csv)")
    ap.add_argument("-o", "--out", default=None, help="output PNG path")
    ap.add_argument("--bins", type=int, default=200, help="histogram bins (default 200)")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    csv = args.csv or find_default_csv()
    if not os.path.exists(csv):
        sys.stderr.write(
            f"error: {csv} not found.\n"
            "Run the test first:  cmake --build build && ./build/test_two_stream\n"
        )
        return 1

    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] == 0:
        sys.stderr.write(f"error: {csv} has no usable x,vx rows\n")
        return 1
    x, vx = data[:, 0], data[:, 1]

    if args.show:
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            sys.stderr.write("warning: no interactive backend; saving PNG only\n")
            args.show = False

    fig, (ax_ph, ax_hist) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]}
    )

    # ---- phase-space density (x, vx) ----
    h = ax_ph.hist2d(x, vx, bins=args.bins, cmap="inferno",
                     norm=matplotlib.colors.LogNorm())
    fig.colorbar(h[3], ax=ax_ph, label="particle count")
    ax_ph.set_xlabel("x")
    ax_ph.set_ylabel(r"$v_x$")
    ax_ph.set_title("Two-stream instability — phase space (x, $v_x$) at saturation")

    # ---- velocity distribution f(vx): the trapped gap between the beams fills in ----
    ax_hist.hist(vx, bins=args.bins, orientation="horizontal",
                 color="#3b6fb5", histtype="stepfilled", alpha=0.85)
    ax_hist.set_xlabel("count")
    ax_hist.set_ylabel(r"$v_x$")
    ax_hist.set_title(r"$f(v_x)$")
    ax_hist.set_ylim(ax_ph.get_ylim())
    ax_hist.grid(True, alpha=0.3)

    n = x.size
    trapped = np.mean(np.abs(vx) < 0.5)  # v0 = 1 in the test
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.02,
             f"N = {n} sampled particles    trapped fraction (|$v_x$|<0.5) = {trapped:.3f}",
             ha="center", fontsize=10)

    out = args.out or os.path.splitext(csv)[0] + ".png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({n} points, trapped fraction {trapped:.3f})")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
