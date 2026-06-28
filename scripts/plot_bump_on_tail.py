#!/usr/bin/env python3
"""Render the bump-on-tail instability: per-frame phase-space images, a
begin-vs-end comparison with the velocity distribution f(v), and an mp4.

Input is the folder written by the `bump_on_tail_movie` tool: one
`frame_XXXX.csv` (columns x,vx) per time, plus a `manifest.csv`
(frame,step,time,file). Particles keep their index across frames, so each one is
colored by the population it *started* in (bulk vs. tail beam, split on initial
v_x) — letting you watch the beam get trapped into phase-space vortices while
the tail of f(v) flattens into a quasilinear plateau.

Usage:
    python3 scripts/plot_bump_on_tail.py [frames_dir] [--fps 12] [--no-video]

Outputs (inside frames_dir):
    img/frame_XXXX.png    per-frame phase-space images
    begin_vs_end.png      start vs end phase space + f(v) plateau
    bump_on_tail_movie.mp4 the time evolution (needs ffmpeg)
"""

import argparse
import csv
import os
import subprocess
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POP_COLORS = ("#d1495b", "#3a6ea5")    # beam (red), bulk (blue)
V_B = 1.0                              # beam drift in the bump-on-tail tool
BEAM_CUT = 0.5                         # initial v_x above this -> labeled "beam"


def find_frames_dir(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for c in (os.path.join(root, "build", "bump_frames"),
              os.path.join(root, "bump_frames"),
              "bump_frames"):
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "manifest.csv")):
            return c
    return os.path.join(root, "bump_frames")


def load_manifest(frames_dir):
    rows = []
    with open(os.path.join(frames_dir, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), int(r["step"]), float(r["time"]), r["file"]))
    return rows


def load_frame(frames_dir, fname):
    data = np.loadtxt(os.path.join(frames_dir, fname), delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def render_frame(ax, x, vx, beam, xlim, vlim, title):
    ax.clear()
    # draw bulk first, beam on top so the tail stays visible
    for b, color in ((False, POP_COLORS[1]), (True, POP_COLORS[0])):
        m = beam if b else ~beam
        ax.scatter(x[m], vx[m], s=1.5, c=color, alpha=0.35, linewidths=0,
                   rasterized=True)
    ax.axhline(V_B, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlim(xlim)
    ax.set_ylim(vlim)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_x$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", nargs="?", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-video", action="store_true", help="skip mp4 assembly")
    args = ap.parse_args()

    frames_dir = find_frames_dir(args.frames_dir)
    man_path = os.path.join(frames_dir, "manifest.csv")
    if not os.path.exists(man_path):
        sys.stderr.write(
            f"error: {man_path} not found.\n"
            "Run the dumper first:\n"
            "  cmake --build build --target bump_on_tail_movie && "
            "(cd build && ./bump_on_tail_movie)\n")
        return 1

    rows = load_manifest(frames_dir)
    if not rows:
        sys.stderr.write("error: manifest has no frames\n")
        return 1

    # population label + global axis limits from a first pass
    x0, vx0 = load_frame(frames_dir, rows[0][3])
    beam = vx0 > BEAM_CUT                     # initial population, per particle
    vmin, vmax, xmax = vx0.min(), vx0.max(), x0.max()
    for _, _, _, fname in rows:
        _, vx = load_frame(frames_dir, fname)
        vmin = min(vmin, vx.min()); vmax = max(vmax, vx.max())
    pad = 0.08 * (vmax - vmin)
    vlim = (vmin - pad, vmax + pad)
    xlim = (0.0, xmax)

    img_dir = os.path.join(frames_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # ---- per-frame images ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for frame, step, time, fname in rows:
        x, vx = load_frame(frames_dir, fname)
        render_frame(ax, x, vx, beam, xlim, vlim,
                     f"Bump-on-tail phase space   t = {time:6.2f}   (step {step})")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"frame_{frame:04d}.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {len(rows)} frame images to {img_dir}/")

    # ---- begin vs end: phase space + f(v) plateau ----
    fig = plt.figure(figsize=(16, 5))
    axb = fig.add_subplot(1, 3, 1)
    axe = fig.add_subplot(1, 3, 2, sharey=axb)
    axf = fig.add_subplot(1, 3, 3)

    xb, vb = load_frame(frames_dir, rows[0][3])
    xe, ve = load_frame(frames_dir, rows[-1][3])
    render_frame(axb, xb, vb, beam, xlim, vlim, f"start   t = {rows[0][2]:.2f}")
    render_frame(axe, xe, ve, beam, xlim, vlim, f"end   t = {rows[-1][2]:.2f}")

    bins = np.linspace(vlim[0], vlim[1], 160)
    axf.hist(vb, bins=bins, histtype="step", density=True, color="#888888",
             lw=1.4, label=f"f(v) start (t={rows[0][2]:.0f})")
    axf.hist(ve, bins=bins, histtype="step", density=True, color="#d1495b",
             lw=1.6, label=f"f(v) end (t={rows[-1][2]:.0f})")
    axf.axvline(V_B, color="k", lw=0.6, ls="--", alpha=0.4)
    axf.set_xlabel(r"$v_x$"); axf.set_ylabel("f(v)")
    axf.set_title("velocity distribution: bump -> plateau")
    axf.legend(fontsize=9); axf.grid(True, alpha=0.2)

    fig.suptitle("Bump-on-tail instability (self-excited from shot noise): the tail "
                 "beam traps into phase-space vortices,\nflattening the positive slope "
                 "into a quasilinear plateau", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    cmp_path = os.path.join(frames_dir, "begin_vs_end.png")
    fig.savefig(cmp_path, dpi=140)
    plt.close(fig)
    print(f"wrote {cmp_path}")

    # ---- video ----
    if not args.no_video:
        mp4 = os.path.join(frames_dir, "bump_on_tail_movie.mp4")
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
            sys.stderr.write("ffmpeg not found; skipping video (images still written)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
