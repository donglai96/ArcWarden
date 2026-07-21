#!/usr/bin/env python3
"""Reproduce Chen et al., Phys. Plasmas 33, 072105 (2026) Fig. 1 from chirp2d runs.

3 columns (Case I incoherent whistler / II discrete chorus / III hiss-like
chorus) x 3 rows:
  (a) dB_phi/B_e0 waveform at lambda = +5 deg   (our By at the +116.4 probe)
  (b) dB/B_e0 = |By + iBz|/B_e0 envelope (log), peak time marked
  (c) STFT spectrogram, their recipe: window 76.8/We0 (256 samples at our
      1.5/wpe probe cadence), hop 2.4/We0 (8 samples), Hann, amplitude in
      dB/B_e0, color 1e-4..1e-2, omega/We0 in [0, 0.8].

Usage: plot_chen2026_fig1.py case1dir case2dir case3dir out.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_meta(d):
    meta = {}
    for line in open(f"{d}/meta.txt"):
        parts = line.split()
        if len(parts) == 2:
            k, v = parts
            meta.setdefault(k, []).append(float(v))
    return {k: (v[0] if len(v) == 1 else v) for k, v in meta.items()}


def load_probe(d, probe_offset=116.4):
    m = read_meta(d)
    npr = int(m["nprobe"])
    raw = np.fromfile(f"{d}/probe.bin", dtype=np.float32)
    raw = raw[: (len(raw) // (2 * npr)) * 2 * npr].reshape(-1, npr, 2)
    # pick the probe closest to the requested offset from the equator
    ix = np.array(m["probe_ix"], dtype=float)
    off = ix * m["dx"] - m["b0_xc"]
    p = int(np.argmin(np.abs(off - probe_offset)))
    t_wpe = (np.arange(raw.shape[0]) + 1) * m["probe_every"] * m["dt"]
    return m, t_wpe, raw[:, p, 0].astype(float), raw[:, p, 1].astype(float)


def stft(sig, nwin, hop):
    w = np.hanning(nwin)
    nfrm = (len(sig) - nwin) // hop + 1
    frames = np.stack([sig[i * hop : i * hop + nwin] * w for i in range(nfrm)])
    spec = np.fft.fft(frames, axis=1) * (2.0 / w.sum())  # amplitude calibration
    return spec


def main():
    dirs = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else "chen2026_fig1.png"
    titles = ["Case I: incoherent whistler", "Case II: discrete-element chorus",
              "Case III: hiss-like chorus"]
    # windows: prefer the PAPER's own Fig.-1 windows when the measured peak
    # falls inside them (the δf runs match their timeline); otherwise fall
    # back to a window around our peak (full-f runs fire earlier).
    paper_windows = [(10000, 17500), (5000, 10000), (0, 5000)]
    xlims = []
    for d, pw in zip(dirs, paper_windows):
        m, t_wpe, by, bz = load_probe(d)
        env = np.abs(by / m["wce"] + 1j * bz / m["wce"])
        envs = np.convolve(env, np.ones(256) / 256, mode="same")
        tpk = t_wpe[int(np.argmax(envs))] * m["wce"]
        xlims.append(pw if pw[0] <= tpk <= pw[1] else (0, t_wpe[-1] * m["wce"]))

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 9),
                             gridspec_kw=dict(height_ratios=[1, 1, 1.6]))
    for col, (d, title, xlim) in enumerate(zip(dirs, titles, xlims)):
        m, t_wpe, by, bz = load_probe(d)
        wce = m["wce"]
        t_oe = t_wpe * wce                     # We0 t
        byn, bzn = by / wce, bz / wce          # dB/B_e0
        env = np.abs(byn + 1j * bzn)

        ax = axes[0, col]
        ax.plot(t_oe, byn, lw=0.3, color="tab:blue")
        ax.set_ylim(-0.045, 0.045)
        ax.set_xlim(*xlim)
        ax.set_title(f"{title}\n" + r"$n_h/n_{e0}$=%g, $T_\perp/T_\parallel$=%g"
                     % (m["nh"], {0: 1.19, 1: 1.52, 2: 3.75}[col]), fontsize=11)
        if col == 0:
            ax.set_ylabel(r"$\delta B_\varphi/B_{e0}$")

        ax = axes[1, col]
        # short boxcar for the envelope trace (76.8/We0, their FFT window)
        k = 256
        envs = np.convolve(env, np.ones(k) / k, mode="same")
        ax.semilogy(t_oe, envs, lw=0.6, color="k")
        ipk = int(np.argmax(envs))
        ax.axvline(t_oe[ipk], color="g", ls="--", lw=1)
        ax.text(t_oe[ipk], 2e-3, r" $T$=%d, pk=%.3f" % (t_oe[ipk], envs[ipk]),
                fontsize=8, color="g")
        ax.set_ylim(3e-4, 1e-1)
        ax.set_xlim(*xlim)
        if col == 0:
            ax.set_ylabel(r"$\delta B/B_{e0}$")

        ax = axes[2, col]
        nwin, hop = 256, 8                     # 76.8/We0 window, 2.4/We0 hop
        spec = stft(byn + 1j * bzn, nwin, hop)
        dt_s = t_wpe[1] - t_wpe[0]
        freqs = np.fft.fftfreq(nwin, d=dt_s) * 2 * np.pi / wce   # omega/We0
        tt = (t_wpe[0] + (np.arange(spec.shape[0]) * hop + nwin / 2) * dt_s) * wce
        sel = (freqs >= 0) & (freqs <= 0.85)
        pw = np.abs(spec[:, sel]).T
        # jfilter=3 + ppc=3200 runs: per-bin noise ~3e-5, close to the paper's
        # 1e-5 floor; smooth 3 frames in t, color range near theirs
        kt = np.ones((1, 3)) / 3.0
        from scipy.ndimage import convolve as ndconv
        pw = ndconv(pw, kt, mode="nearest")
        pc = ax.pcolormesh(tt, freqs[sel], np.clip(pw, 1e-5, None),
                           norm=LogNorm(vmin=2e-4, vmax=2e-2),
                           cmap="jet", shading="auto")
        ax.set_ylim(0, 0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$\Omega_{e0} t$")
        if col == 0:
            ax.set_ylabel(r"$\omega/\Omega_{e0}$")
        if col == 2:
            fig.colorbar(pc, ax=ax, label=r"$\delta B/B_{e0}$", pad=0.02)

    m0 = read_meta(dirs[0])
    method = "delta-f (gcPIC-δf config: wd-noise source, cavity boundary)" \
        if m0.get("deltaf", 0) else "full-f"
    fig.suptitle(f"ArcWarden chirp2d {method} reproduction of Chen et al. PoP 33, 072105 (2026) Fig. 1"
                 r"  —  dipole $B_0(s)$, loss-cone hot e$^-$, probe $\lambda=+5^\circ$",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
