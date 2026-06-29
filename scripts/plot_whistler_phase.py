#!/usr/bin/env python3
"""Plot the final whistler-pump x-v_parallel phase-space histogram and fields.

Input is the folder written by the `whistler_pump` tool:

    whistler_fhist.bin    float32 histogram, shape (nvb, nxb)
    whistler_fhist.meta   nxb/nvb/Lx/vlo/vhi/vr metadata
    whistler_fields.csv   x,Epar,Eperp,Ey,dBy,dBmag field profiles

Usage:
    python3 scripts/plot_whistler_phase.py [data_dir] [-o out.png] [--show]

Example:
    cmake --build build --target whistler_pump
    (cd build && ./whistler_pump 2000 6000 4096 1 1 2)
    python3 scripts/plot_whistler_phase.py build
"""

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arcwarden-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/arcwarden-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm


def find_data_dir(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for c in (os.path.join(root, "build"), root, "."):
        if os.path.exists(os.path.join(c, "whistler_fhist.bin")):
            return c
    return os.path.join(root, "build")


def load_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            key, value = parts
            if key in ("nxb", "nvb"):
                meta[key] = int(value)
            else:
                meta[key] = float(value)
    required = ("nxb", "nvb", "Lx", "vlo", "vhi")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"{path} is missing metadata keys: {', '.join(missing)}")
    return meta


def load_hist(data_dir):
    meta_path = os.path.join(data_dir, "whistler_fhist.meta")
    bin_path = os.path.join(data_dir, "whistler_fhist.bin")
    if not os.path.exists(meta_path) or not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"expected whistler_fhist.bin and whistler_fhist.meta in {data_dir}")

    meta = load_meta(meta_path)
    nxb = meta["nxb"]
    nvb = meta["nvb"]
    hist = np.fromfile(bin_path, dtype=np.float32)
    expected = nxb * nvb
    if hist.size != expected:
        raise ValueError(
            f"{bin_path} has {hist.size} float32 values, expected {expected} "
            f"from nxb={nxb}, nvb={nvb}")
    return hist.reshape(nvb, nxb), meta


def load_fields(data_dir):
    path = os.path.join(data_dir, "whistler_fields.csv")
    if not os.path.exists(path):
        return None
    fields = np.genfromtxt(path, delimiter=",", names=True)
    fields = np.atleast_1d(fields)
    required = ("x", "Epar", "Eperp", "dBy", "dBmag")
    missing = [k for k in required if k not in fields.dtype.names]
    if missing:
        raise ValueError(f"{path} is missing field columns: {', '.join(missing)}")
    return fields


def default_output_path(data_dir):
    return os.path.join(data_dir, "whistler_phase_x_vpar.png")


def positive_limits(hist, high_percentile):
    positive = hist[hist > 0.0]
    if positive.size == 0:
        return None
    vmin = np.percentile(positive, 1.0)
    vmax = np.percentile(positive, high_percentile)
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = positive.min()
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = positive.max()
    return vmin, vmax


def draw_fields(ax_e, ax_b, fields):
    ax_e.plot(fields["x"], fields["Epar"], lw=1.0, color="#1f77b4",
              label=r"$E_\parallel$")
    ax_e.plot(fields["x"], fields["Eperp"], lw=1.0, color="#d62728",
              label=r"$E_\perp$")
    ax_e.set_ylabel("E")
    ax_e.legend(loc="upper right", ncol=2, fontsize=9, frameon=True)
    ax_e.grid(True, alpha=0.25)

    ax_b.plot(fields["x"], fields["dBy"], lw=1.0, color="#2ca02c",
              label=r"$\delta B_y$")
    ax_b.plot(fields["x"], fields["dBmag"], lw=1.0, color="#9467bd",
              label=r"$|\delta B|$")
    ax_b.set_ylabel(r"$\delta B$")
    ax_b.legend(loc="upper right", ncol=2, fontsize=9, frameon=True)
    ax_b.grid(True, alpha=0.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default=None,
                    help="directory containing whistler_fhist.bin/.meta; default: build/")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG path; default: data_dir/whistler_phase_x_vpar.png")
    ap.add_argument("--linear", action="store_true",
                    help="use a linear color scale instead of log density")
    ap.add_argument("--norm", choices=("log", "power", "linear"), default="log",
                    help="density color normalization; default: log")
    ap.add_argument("--vlim", nargs=2, type=float, default=(-6.0, 6.0),
                    metavar=("VMIN", "VMAX"),
                    help="v_parallel axis range; default: -6 6")
    ap.add_argument("--no-vr", action="store_true",
                    help="do not draw the resonant velocity from metadata")
    ap.add_argument("--no-fields", action="store_true",
                    help="plot phase space only, without field panels")
    ap.add_argument("--cmap", default="turbo",
                    help="matplotlib colormap for density; default: turbo")
    ap.add_argument("--clip-percentile", type=float, default=99.8,
                    help="upper color clipping percentile; default: 99.8")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--show", action="store_true",
                    help="display interactively instead of only writing the PNG")
    args = ap.parse_args()
    if args.linear:
        args.norm = "linear"
    if args.vlim[0] >= args.vlim[1]:
        sys.stderr.write("error: --vlim needs VMIN < VMAX\n")
        return 1

    if args.show:
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            sys.stderr.write("warning: no interactive backend; saving PNG only\n")
            args.show = False

    data_dir = find_data_dir(args.data_dir)
    try:
        hist, meta = load_hist(data_dir)
    except (OSError, ValueError) as e:
        sys.stderr.write(
            f"error: {e}\n"
            "Run the dumper first, for example:\n"
            "  cmake --build build --target whistler_pump\n"
            "  (cd build && ./whistler_pump 2000 6000 4096 1 1 2)\n")
        return 1
    try:
        fields = None if args.no_fields else load_fields(data_dir)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"warning: {e}; plotting phase space only\n")
        fields = None

    out = args.output or default_output_path(data_dir)
    extent = (0.0, meta["Lx"], meta["vlo"], meta["vhi"])
    if fields is None:
        fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    else:
        fig, (ax_e, ax_b, ax) = plt.subplots(
            3, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 3.5]})
        draw_fields(ax_e, ax_b, fields)
        ax_e.tick_params(labelbottom=False)
        ax_b.tick_params(labelbottom=False)

    if args.norm == "linear":
        vmax = np.percentile(hist, args.clip_percentile)
        image = ax.imshow(hist, origin="lower", aspect="auto", extent=extent,
                          cmap=args.cmap, vmin=0.0, vmax=vmax)
        cbar_label = "particle count / bin"
    elif args.norm == "power":
        vmax = np.percentile(hist, args.clip_percentile)
        if not np.isfinite(vmax) or vmax <= 0.0:
            sys.stderr.write("error: histogram is empty\n")
            return 1
        image = ax.imshow(hist, origin="lower", aspect="auto", extent=extent,
                          cmap=args.cmap, norm=PowerNorm(gamma=0.45,
                                                          vmin=0.0, vmax=vmax))
        cbar_label = "particle count / bin"
    else:
        lim = positive_limits(hist, args.clip_percentile)
        if lim is None:
            sys.stderr.write("error: histogram is empty\n")
            return 1
        image = ax.imshow(np.ma.masked_less_equal(hist, 0.0), origin="lower",
                          aspect="auto", extent=extent, cmap=args.cmap,
                          norm=LogNorm(vmin=lim[0], vmax=lim[1]))
        cbar_label = "particle count / bin (log)"

    if not args.no_vr and "vr" in meta:
        ax.axhline(meta["vr"], color="cyan", lw=1.0, ls="--", alpha=0.9,
                   label=rf"$v_r = {meta['vr']:.3g}$")
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax.set_xlim(0.0, meta["Lx"])
    ax.set_ylim(args.vlim[0], args.vlim[1])
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_\parallel$")
    ax.set_title(r"Whistler pump final phase space: $f(x, v_\parallel)$")
    ax.grid(True, color="white", alpha=0.12, lw=0.5)

    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(cbar_label)

    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
