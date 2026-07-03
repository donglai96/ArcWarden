#!/usr/bin/env python3
"""Time-evolution video of whistler-pump Simulation 1 (An et al. 2019).

Per frame (sharing x): top = delta E_par (red) + delta E_perp (blue) waveforms;
bottom = f(x, v_parallel) phase-space density. Shows the trapped resonant island
forming and the Langmuir waves igniting in delta E_par near t_off.

Reads whistler_vid_phase.bin, whistler_vid_field.bin, whistler_vid.meta written
by tools/whistler_pump.cu. Renders PNG frames then an mp4 (needs ffmpeg).
Usage: python3 scripts/plot_whistler_video.py [--fps 15]
"""
import argparse, os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("--sim", type=int, default=2)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--outdir", default=None)
args = ap.parse_args()
pref = f"whistler_s{args.sim}_"
outdir = args.outdir or f"whistler_s{args.sim}_vid"
TAG = {1: "v_r/v_th=3.2 Langmuir", 2: "v_r/v_th=2.1 unipolar",
       3: "v_r/v_th=1.0 bipolar"}.get(args.sim, "")

m = dict(l.split() for l in open(pref + "vid.meta"))
nframe = int(m["nframe"]); nxb, nvb = int(m["nxb"]), int(m["nvb"])
nx = int(m["nx"]); Lx = float(m["Lx"]); vlo, vhi = float(m["vlo"]), float(m["vhi"])
toff = float(m["toff"]); dtf = float(m["dt_frame"]); vr = float(m["vr"])

phase = np.fromfile(pref + "vid_phase.bin", dtype=np.float32).reshape(nframe, nvb, nxb)
field = np.fromfile(pref + "vid_field.bin", dtype=np.float32).reshape(nframe, 3, nx)
xf = np.linspace(0, Lx, nx)

# fixed scales across frames
pmax = phase.max()
emax = np.abs(field[:, :2, :]).max() * 1.05

imgdir = os.path.join(outdir, "img")
os.makedirs(imgdir, exist_ok=True)
for k in range(nframe):
    t = k * dtf
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.025], height_ratios=[1, 1.6],
                          hspace=0.10, wspace=0.02)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    c1 = fig.add_subplot(gs[1, 1]); fig.add_subplot(gs[0, 1]).axis("off")

    a0.plot(xf, field[k, 0], color="#d1495b", lw=0.6, label=r"$\delta E_\parallel$")
    a0.plot(xf, field[k, 1], color="#3a6ea5", lw=0.6, label=r"$\delta E_\perp$")
    a0.set_ylim(-emax, emax); a0.set_ylabel(r"$\delta E$"); a0.grid(alpha=.2)
    a0.legend(loc="upper right", fontsize=8, ncol=2)
    pump = "PUMP ON" if t < toff else "pump off"
    a0.set_title(f"ArcWarden whistler-pump Sim {args.sim} ({TAG})   "
                 f"t = {t:6.1f} $\\omega_{{pe}}^{{-1}}$   [{pump}]")
    plt.setp(a0.get_xticklabels(), visible=False)

    im = a1.imshow(phase[k], origin="lower", extent=[0, Lx, vlo, vhi], aspect="auto",
                   cmap="turbo", norm=LogNorm(vmin=max(pmax*2e-4, 1), vmax=pmax))
    a1.axhline(vr, color="w", ls="--", lw=0.7)
    a1.set_ylabel(r"$v_\parallel/v_{th}$"); a1.set_xlabel("x")
    fig.colorbar(im, cax=c1, label="f")
    fig.savefig(os.path.join(imgdir, f"f{k:04d}.png"), dpi=110)
    plt.close(fig)
    if k % 20 == 0: print(f"frame {k}/{nframe}")

mp4 = os.path.join(outdir, f"whistler_s{args.sim}.mp4")
ff = shutil.which("ffmpeg")
if ff:
    cmd = [ff, "-y", "-framerate", str(args.fps), "-i", os.path.join(imgdir, "f%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0: print("wrote", mp4)
    else: sys.stderr.write(r.stderr[-800:])
else:
    print("ffmpeg not found; PNG frames in", imgdir)
