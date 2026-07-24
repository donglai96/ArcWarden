#!/usr/bin/env python3
"""G2.1 gate figure — Li/Bortnik/An 2019 replication in ArcWarden.

Panels:
  (a) spectrogram of By+iBz at a probe (w/We vs t)
  (b) f(vpar) snapshots (log) with the LB phase-velocity band marked:
      the Landau plateau must grow inside [Vp(0.2), Vp(0.5)] = [0.08, 0.10]c
  (c) band-integrated power vs time: LB / gap window / UB
  (d) f(vpar, t) change map (plateau carving history)

Usage: plot_li2019_band.py <rundir> <out.png>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob


def meta(d):
    m = {}
    for ln in open(f"{d}/meta.txt"):
        p = ln.split()
        if len(p) >= 2 and p[0] not in ("species", "probe_ix"):
            m[p[0]] = float(p[1])
    return m


def main():
    d, out = sys.argv[1], sys.argv[2]
    m = meta(d)
    wce, dt = m["wce"], m["dt"]
    nprobe = int(m["nprobe"])
    NB, VMAX = int(m["nb"]), m["vmax"]

    pr = np.fromfile(f"{d}/probe.bin", dtype=np.float32).reshape(-1, 2 * nprobe)
    sig = (pr[:, 2] + 1j * pr[:, 3]) / wce          # middle probe
    dt_s = m["probe_every"] * dt
    t_oe = np.arange(len(sig)) * dt_s * wce

    nwin, hop = 4096, 4096 // 16
    nseg = (len(sig) - nwin) // hop
    spec = np.empty((nseg, nwin), dtype=np.float32)
    win = np.hanning(nwin)
    for i in range(nseg):
        spec[i] = np.abs(np.fft.fft(sig[i * hop:i * hop + nwin] * win))
    freqs = np.fft.fftfreq(nwin, d=dt_s) * 2 * np.pi / wce
    tt = (np.arange(nseg) * hop + nwin / 2) * dt_s * wce
    sel = (freqs >= 0.05) & (freqs <= 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    pw = np.abs(spec[:, sel]).T
    pc = ax.pcolormesh(tt, freqs[sel], np.clip(pw / pw.max(), 1e-5, None),
                       norm=LogNorm(vmin=1e-4, vmax=1), cmap="jet",
                       shading="auto", rasterized=True)
    ax.axhline(0.5, color="w", ls=":", lw=1)
    ax.set_ylabel(r"$\omega/\Omega_e$"); ax.set_xlabel(r"$\Omega_e t$")
    ax.set_title("(a) spectrogram (middle probe)")
    fig.colorbar(pc, ax=ax)

    files = sorted(glob.glob(f"{d}/fhist_*.bin"))
    v = (np.arange(NB) + 0.5) / NB * 2 * VMAX - VMAX
    H = np.array([np.fromfile(f, dtype=np.float64) for f in files])
    H /= H[0].sum()
    th = m["fhist_every"] * dt * wce * (np.arange(len(files)) + 1)

    ax = axes[0, 1]
    for idx, cc in ((0, "k"), (len(H) // 4, "tab:blue"),
                    (len(H) // 2, "tab:green"), (len(H) - 1, "tab:red")):
        ax.semilogy(v, np.maximum(H[idx], 1e-12), color=cc, lw=1.2,
                    label=f"$\\Omega_e t$={th[idx]:.0f}")
    for s in (+1, -1):
        ax.axvspan(s * 0.08, s * 0.10, color="orange", alpha=0.25)
    ax.set_xlim(-0.3, 0.3); ax.set_ylim(1e-7, 1)
    ax.set_xlabel(r"$v_\parallel/c$"); ax.set_ylabel(r"$f(v_\parallel)$")
    ax.set_title("(b) plateau monitor (shade = $V_p$(LB) band)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    def band(lo, hi):
        s = (freqs >= lo) & (freqs <= hi)
        return np.sqrt((np.abs(spec[:, s]) ** 2).sum(axis=1))
    for lo, hi, lab, cc in ((0.20, 0.45, "LB 0.20-0.45", "tab:blue"),
                            (0.48, 0.56, "gap 0.48-0.56", "tab:red"),
                            (0.58, 0.80, "UB 0.58-0.80", "tab:green")):
        ax.semilogy(tt, band(lo, hi), color=cc, lw=1.4, label=lab)
    ax.set_xlabel(r"$\Omega_e t$"); ax.set_ylabel("band |B| (arb)")
    ax.set_title("(c) band power history")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    D = (H - H[0]) / H[0].max()
    pc = ax.pcolormesh(th, v, D.T, cmap="RdBu_r",
                       vmin=-np.abs(D).max(), vmax=np.abs(D).max(),
                       shading="auto", rasterized=True)
    for s in (+1, -1):
        ax.axhline(s * 0.08, color="k", ls=":", lw=0.7)
        ax.axhline(s * 0.10, color="k", ls=":", lw=0.7)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel(r"$\Omega_e t$"); ax.set_ylabel(r"$v_\parallel/c$")
    ax.set_title(r"(d) $\Delta f(v_\parallel, t)$ — plateau carving")
    fig.colorbar(pc, ax=ax)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
