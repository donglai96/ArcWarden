#!/usr/bin/env python3
"""Render bump-on-tail with the electric field UNDER the phase space, same x-axis.

Reads the folder written by `run_deck` (with a [time] dump_every): each row of
manifest.csv points to a phase frame (frame_XXXX.csv: x,vx) and a field profile
(field_XXXX.csv: x,Ex averaged over y). Each figure stacks:
    top    : (x, vx) phase space, colored by starting population (bulk vs beam)
    bottom : Ex(x), the longitudinal field growing as the instability develops

Usage:
    python3 scripts/plot_bump_field.py [frames_dir] [--fps 12] [--no-video]

Outputs (inside frames_dir):
    img_field/frame_XXXX.png   per-frame stacked images
    begin_vs_end_field.png     start vs end (phase + field)
    bump_field_movie.mp4       the time evolution (needs ffmpeg)
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

POP_COLORS = ("#d1495b", "#3a6ea5")    # beam (red), bulk (blue)
V_B = 1.0                              # beam drift in the bump-on-tail deck
BEAM_CUT = 0.5                         # initial v_x above this -> "beam"


def find_frames_dir(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for c in (os.path.join(root, "build", "bump_field"),
              os.path.join(root, "bump_field"), "bump_field"):
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "manifest.csv")):
            return c
    return os.path.join(root, "bump_field")


def load_manifest(frames_dir):
    rows = []
    with open(os.path.join(frames_dir, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), int(r["step"]), float(r["time"]),
                         r["file"], r.get("field", "")))
    return rows


def load_xy(frames_dir, fname):
    d = np.loadtxt(os.path.join(frames_dir, fname), delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def render(axp, axe, x, vx, beam, xf, ef, xlim, vlim, elim, title):
    axp.clear(); axe.clear()
    for b, color in ((False, POP_COLORS[1]), (True, POP_COLORS[0])):
        m = beam if b else ~beam
        axp.scatter(x[m], vx[m], s=1.5, c=color, alpha=0.35, linewidths=0, rasterized=True)
    axp.axhline(V_B, color="k", lw=0.6, ls="--", alpha=0.4)
    axp.set_xlim(xlim); axp.set_ylim(vlim)
    axp.set_ylabel(r"$v_x$"); axp.set_title(title)
    axp.grid(True, alpha=0.2); axp.tick_params(labelbottom=False)

    axe.plot(xf, ef, color="#2a9d8f", lw=1.2)
    axe.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    axe.set_xlim(xlim); axe.set_ylim(elim)
    axe.set_xlabel("x"); axe.set_ylabel(r"$E_x$")
    axe.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", nargs="?", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    frames_dir = find_frames_dir(args.frames_dir)
    if not os.path.exists(os.path.join(frames_dir, "manifest.csv")):
        sys.stderr.write(
            f"error: {frames_dir}/manifest.csv not found.\n"
            "Run the deck first (needs dump_every>0):\n"
            "  ./build/run_deck decks/bump_on_tail.ini build/bump_field\n")
        return 1
    rows = load_manifest(frames_dir)
    if not rows or not rows[0][4]:
        sys.stderr.write("error: manifest has no field column; re-run run_deck.\n")
        return 1

    # population label + global axis limits from a first pass
    x0, vx0 = load_xy(frames_dir, rows[0][3])
    beam = vx0 > BEAM_CUT
    vmin, vmax, xmax, emax = vx0.min(), vx0.max(), x0.max(), 0.0
    for _, _, _, pf, ff in rows:
        _, vx = load_xy(frames_dir, pf)
        _, ef = load_xy(frames_dir, ff)
        vmin = min(vmin, vx.min()); vmax = max(vmax, vx.max())
        emax = max(emax, float(np.max(np.abs(ef))))
    vpad = 0.08 * (vmax - vmin)
    vlim = (vmin - vpad, vmax + vpad)
    elim = (-1.15 * emax, 1.15 * emax)
    xlim = (0.0, xmax)

    img_dir = os.path.join(frames_dir, "img_field")
    os.makedirs(img_dir, exist_ok=True)

    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    axp, axe = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for frame, step, time, pf, ff in rows:
        x, vx = load_xy(frames_dir, pf)
        xf, ef = load_xy(frames_dir, ff)
        render(axp, axe, x, vx, beam, xf, ef, xlim, vlim, elim,
               f"Bump-on-tail   t = {time:6.2f}   (step {step})")
        fig.savefig(os.path.join(img_dir, f"frame_{frame:04d}.png"), dpi=120,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {len(rows)} stacked frames to {img_dir}/")

    # begin vs end
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.12)
    for col, (frame, step, time, pf, ff) in enumerate((rows[0], rows[-1])):
        axp = fig.add_subplot(gs[0, col]); axe = fig.add_subplot(gs[1, col])
        x, vx = load_xy(frames_dir, pf); xf, ef = load_xy(frames_dir, ff)
        render(axp, axe, x, vx, beam, xf, ef, xlim, vlim, elim,
               f"{'start' if col == 0 else 'end'}   t = {time:.2f}")
    cmp_path = os.path.join(frames_dir, "begin_vs_end_field.png")
    fig.savefig(cmp_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cmp_path}")

    if not args.no_video:
        mp4 = os.path.join(frames_dir, "bump_field_movie.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-framerate", str(args.fps),
                   "-i", os.path.join(img_dir, "frame_%04d.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-pix_fmt", "yuv420p", mp4]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"wrote {mp4}  ({len(rows)} frames @ {args.fps} fps)")
            else:
                sys.stderr.write("ffmpeg failed:\n" + r.stderr[-800:] + "\n")
        else:
            sys.stderr.write("ffmpeg not found; skipping video\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
