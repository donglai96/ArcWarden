#!/usr/bin/env python3
"""Render the two-stream phase-space evolution: per-frame images, a begin-vs-end
comparison, and an mp4 video.

Input is the folder written by the `two_stream_movie` tool: one `frame_XXXX.csv`
(columns x,vx — ALL particles) per time, plus a `manifest.csv`
(frame,step,time,file). Particles keep their index across frames, so each one is
colored by the beam it *started* in (sign of its initial vx) — letting you watch
the two counter-streaming beams wind up into phase-space vortices.

Usage:
    python3 scripts/plot_phase_movie.py [frames_dir] [--fps 12] [--no-video]

Outputs (inside frames_dir):
    img/frame_XXXX.png   per-frame phase-space images
    begin_vs_end.png     first vs last frame, side by side
    two_stream_movie.mp4 the time evolution (needs ffmpeg)
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

BEAM_COLORS = ("#d1495b", "#2e6fb7")   # beam +v0 (red), beam -v0 (blue)
V0 = 1.0                               # beam drift in the two-stream tool


def find_frames_dir(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for c in (os.path.join(root, "phase_frames"),
              os.path.join(root, "build", "phase_frames"),
              "phase_frames"):
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "manifest.csv")):
            return c
    return os.path.join(root, "phase_frames")


def load_manifest(frames_dir):
    path = os.path.join(frames_dir, "manifest.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), int(r["step"]), float(r["time"]), r["file"]))
    return rows


def load_frame(frames_dir, fname):
    data = np.loadtxt(os.path.join(frames_dir, fname), delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def render_frame(ax, x, vx, beam, xlim, vlim, title):
    ax.clear()
    for b, color in ((True, BEAM_COLORS[0]), (False, BEAM_COLORS[1])):
        m = beam if b else ~beam
        ax.scatter(x[m], vx[m], s=1.5, c=color, alpha=0.35, linewidths=0,
                   rasterized=True)
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
            "  cmake --build build --target two_stream_movie && "
            "(cd build && ./two_stream_movie)\n")
        return 1

    rows = load_manifest(frames_dir)
    if not rows:
        sys.stderr.write("error: manifest has no frames\n")
        return 1

    # beam label + global axis limits from a first pass
    x0, vx0 = load_frame(frames_dir, rows[0][3])
    beam = vx0 > 0.0                         # initial beam membership, per particle
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for frame, step, time, fname in rows:
        x, vx = load_frame(frames_dir, fname)
        render_frame(ax, x, vx, beam, xlim, vlim,
                     f"Two-stream phase space   t = {time:6.2f}   (step {step})")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"frame_{frame:04d}.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {len(rows)} frame images to {img_dir}/")

    # ---- begin vs end comparison ----
    fig, (axb, axe) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    xb, vb = load_frame(frames_dir, rows[0][3])
    xe, ve = load_frame(frames_dir, rows[-1][3])
    render_frame(axb, xb, vb, beam, xlim, vlim, f"start   t = {rows[0][2]:.2f}")
    render_frame(axe, xe, ve, beam, xlim, vlim, f"end   t = {rows[-1][2]:.2f}")
    trap_b = np.mean(np.abs(vb) < 0.5 * V0)
    trap_e = np.mean(np.abs(ve) < 0.5 * V0)
    fig.suptitle("Two-stream instability: counter-streaming beams (left) wind into "
                 f"phase-space vortices (right)\ntrapped fraction (|$v_x$|<{0.5*V0:.1f}): "
                 f"{trap_b:.3f} -> {trap_e:.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cmp_path = os.path.join(frames_dir, "begin_vs_end.png")
    fig.savefig(cmp_path, dpi=140)
    plt.close(fig)
    print(f"wrote {cmp_path}  (trapped {trap_b:.3f} -> {trap_e:.3f})")

    # ---- video ----
    if not args.no_video:
        mp4 = os.path.join(frames_dir, "two_stream_movie.mp4")
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
