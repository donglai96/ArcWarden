#!/usr/bin/env python3
"""chirp1d diagnostics: dynamic spectra (the chirping money plot), energy
history, |B_perp|(x,t), and hot phase-space dumps.

Usage: plot_chirp1d.py <outdir>/<prefix> [--nperseg N] [--fmax F] [--db-floor D]

Reads <prefix>_meta.txt, _probes.bin, _energy.csv, _frames.bin, _phase.bin.
Probe records are float32 [nsample, nprobe, 4] = (By, Bz, Ey, Ez).
The complex signal s = By + i*Bz puts R-mode (whistler) power at f > 0
(numpy FFT basis e^{-iwt}; R-mode rotates as (By-iBz) ~ e^{-iwt}).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def stft_complex(s, dt, nperseg, noverlap):
    hop = nperseg - noverlap
    nwin = max(1, (len(s) - nperseg) // hop + 1)
    win = np.hanning(nperseg)
    S = np.empty((nperseg, nwin), dtype=complex)
    for j in range(nwin):
        seg = s[j * hop: j * hop + nperseg] * win
        S[:, j] = np.fft.fft(seg)
    f = np.fft.fftfreq(nperseg, d=dt) * 2 * np.pi   # angular frequency
    t = (np.arange(nwin) * hop + nperseg / 2) * dt
    return f, t, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="<outdir>/<prefix> (no suffix)")
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--fmax", type=float, default=1.1, help="max omega/wce0 shown")
    ap.add_argument("--db-floor", type=float, default=-13.0,
                    help="log10 power floor relative to max")
    args = ap.parse_args()

    pre = args.prefix
    meta = read_meta(pre + "_meta.txt")
    dt = float(meta["dt"])
    a = float(meta["a"])
    nx = int(meta["nx"])
    dx = float(meta["dx"])
    xc = float(meta["xc"])

    # ---- dynamic spectra at probes ----
    nprobe = int(meta.get("nprobe", 0))
    if nprobe > 0 and os.path.exists(pre + "_probes.bin"):
        probes = [float(v) for v in meta["probes"].split(",") if v]
        stride = int(meta["probe_stride"])
        dtp = stride * dt
        raw = np.fromfile(pre + "_probes.bin", dtype=np.float32)
        ns = len(raw) // (nprobe * 4)
        raw = raw[: ns * nprobe * 4].reshape(ns, nprobe, 4)

        nseg = max(64, min(args.nperseg, ns // 4))
        fig, axes = plt.subplots(nprobe, 1, figsize=(11, 2.6 * nprobe),
                                 sharex=True, squeeze=False)
        for k in range(nprobe):
            s = raw[:, k, 0].astype(float) + 1j * raw[:, k, 1].astype(float)
            f, t, S = stft_complex(s, dtp, nseg, int(0.875 * nseg))
            sel = (f >= 0) & (f <= args.fmax)
            P = np.abs(S[sel]) ** 2
            Pmax = P.max() if P.max() > 0 else 1.0
            ax = axes[k, 0]
            im = ax.pcolormesh(t, f[sel], np.log10(P / Pmax + 1e-300),
                               vmin=args.db_floor, vmax=0, cmap="turbo",
                               shading="auto")
            wce_local = 1.0 + a * probes[k] ** 2
            ax.axhline(0.5 * wce_local, color="w", ls="--", lw=0.7, alpha=0.7)
            ax.axhline(wce_local, color="w", ls=":", lw=0.7, alpha=0.7)
            ax.set_ylabel(r"$\omega/\Omega_{e0}$")
            ax.text(0.01, 0.95, f"h = {probes[k]:+.0f} $c/\\Omega_{{e0}}$",
                    transform=ax.transAxes, color="w", va="top", fontsize=9)
            fig.colorbar(im, ax=ax, pad=0.01, label=r"$\log_{10} P/P_{max}$")
        axes[-1, 0].set_xlabel(r"$t\,\Omega_{e0}$")
        fig.suptitle("chirp1d dynamic spectrum  (R-mode power)")
        fig.tight_layout()
        fig.savefig(pre + "_spec.png", dpi=140)
        print("wrote", pre + "_spec.png")

    # ---- energy history ----
    if os.path.exists(pre + "_energy.csv"):
        E = np.genfromtxt(pre + "_energy.csv", delimiter=",", names=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].semilogy(E["t"], E["WB"], label=r"$W_B$")
        ax[0].semilogy(E["t"], E["WE"], label=r"$W_E$", alpha=0.7)
        ax[0].semilogy(E["t"], E["Wcold"], label=r"$W_{cold}$", alpha=0.7)
        ax[0].set_xlabel(r"$t\,\Omega_{e0}$"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(E["t"], E["Whot"], label=r"$W_{hot}$")
        axb = ax[1].twinx()
        axb.plot(E["t"], E["bmax"], "r", alpha=0.6, label=r"max$|\delta B|/B_0$")
        ax[1].set_xlabel(r"$t\,\Omega_{e0}$")
        ax[1].legend(loc="upper left"); axb.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(pre + "_energy.png", dpi=140)
        print("wrote", pre + "_energy.png")

    # ---- |B_perp|(x, t) ----
    if os.path.exists(pre + "_frames.bin") and int(meta.get("frame_stride", 0)) > 0:
        dec = int(meta["frame_decim"])
        nptx = nx // dec
        raw = np.fromfile(pre + "_frames.bin", dtype=np.float32)
        nf = len(raw) // (2 * nptx)
        raw = raw[: nf * 2 * nptx].reshape(nf, 2, nptx)
        bp = np.hypot(raw[:, 0], raw[:, 1])
        tf = (np.arange(nf) + 1) * int(meta["frame_stride"]) * dt
        x = (np.arange(nptx) * dec * dx) - xc
        fig, ax = plt.subplots(figsize=(10, 4.6))
        im = ax.pcolormesh(x, tf, np.log10(bp + 1e-12), cmap="magma",
                           shading="auto")
        ax.set_xlabel(r"$h$  $(c/\Omega_{e0})$"); ax.set_ylabel(r"$t\,\Omega_{e0}$")
        fig.colorbar(im, ax=ax, label=r"$\log_{10}|\delta B_\perp|/B_{0,eq}$")
        fig.tight_layout()
        fig.savefig(pre + "_xt.png", dpi=140)
        print("wrote", pre + "_xt.png")

    # ---- hot phase space ----
    if os.path.exists(pre + "_phase.bin"):
        recs = []
        raw = np.fromfile(pre + "_phase.bin", dtype=np.float32)
        i = 0
        while i + 2 <= len(raw):
            t0, npk = raw[i], int(raw[i + 1])
            i += 2
            if i + 4 * npk > len(raw):
                break
            d = raw[i: i + 4 * npk].reshape(npk, 4)
            recs.append((t0, d))
            i += 4 * npk
        if recs:
            show = recs[:: max(1, len(recs) // 4)][-4:]
            fig, axes = plt.subplots(2, len(show), figsize=(3.4 * len(show), 6),
                                     squeeze=False)
            for j, (t0, d) in enumerate(show):
                axes[0, j].hist2d(d[:, 0], d[:, 1], bins=160,
                                  cmap="viridis",
                                  norm=matplotlib.colors.LogNorm())
                axes[0, j].set_title(f"t = {t0:.0f}", fontsize=9)
                axes[0, j].set_xlabel("h"); axes[0, j].set_ylabel(r"$u_\parallel$")
                eq = np.abs(d[:, 0]) < 100
                up = np.hypot(d[eq, 2], d[eq, 3])
                axes[1, j].hist2d(d[eq, 1], up, bins=140,
                                  cmap="viridis",
                                  norm=matplotlib.colors.LogNorm())
                axes[1, j].set_xlabel(r"$u_\parallel$")
                axes[1, j].set_ylabel(r"$u_\perp$ (|h|<100)")
            fig.tight_layout()
            fig.savefig(pre + "_phase.png", dpi=140)
            print("wrote", pre + "_phase.png")


if __name__ == "__main__":
    sys.exit(main())
