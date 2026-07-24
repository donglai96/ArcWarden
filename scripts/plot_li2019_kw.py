#!/usr/bin/env python3
"""G2.1 gate figure, in Li/Bortnik/An 2019's OWN format (their Fig. 4).

Panels:
  (a) k-w magnetic spectra, FFT window T = 50-100 tau_gyro  (their 4b)
  (b) k-w magnetic spectra, FFT window T = 700-800 tau_gyro (their 4f)
      x = k mode number m (waves per box), y = omega/Omega_e
  (c) f(vpar) at T ~ 0, 150, 800 tau_gyro (their 4a/c/e cuts)
  (d) k-integrated omega spectrum early vs late + gap/LB ratio

Usage: plot_li2019_kw.py <rundir> <out.png>
"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def meta(d):
    m = {}
    for ln in open(f"{d}/meta.txt"):
        p = ln.split()
        if len(p) >= 2 and p[0] not in ("species", "probe_ix"):
            m[p[0]] = float(p[1])
    return m


def kw_window(B, i0, i1, dt_line, wce):
    W = B[i0:i1] * np.hanning(i1 - i0)[:, None]
    S = np.fft.fftshift(np.abs(np.fft.fft2(W)) ** 2, axes=(0, 1))
    w = np.fft.fftshift(np.fft.fftfreq(i1 - i0, d=dt_line)) * 2 * np.pi / wce
    return S, w


def main():
    d, out = sys.argv[1], sys.argv[2]
    m = meta(d)
    wce, dt, dx = m["wce"], m["dt"], m["dx"]
    nx = int(m["nx"])
    dt_line = m["bline_every"] * dt
    tau_g = 2 * np.pi / wce                       # gyro-period in 1/wpe units

    files = sorted(glob.glob(f"{d}/bline_*.bin"))
    n = len(files)
    B = np.empty((n, nx), dtype=np.complex64)
    for i, f in enumerate(files):
        a = np.fromfile(f, dtype=np.float32, count=2 * nx)
        B[i] = a[:nx] + 1j * a[nx:2 * nx]

    def frame(tg):                                # tau_gyro -> frame index
        return min(n - 1, int(tg * tau_g / dt_line))

    modes = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))   # k mode number

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (g0, g1) in zip(axes[0], ((50, 100), (700, 800))):
        i0, i1 = frame(g0), frame(g1)
        S, w = kw_window(B, i0, i1, dt_line, wce)
        selw = (w >= 0.02) & (w <= 1.05)
        pc = ax.pcolormesh(modes, w[selw], np.clip(S[selw] / S[selw].max(),
                           1e-6, None), norm=LogNorm(vmin=1e-5, vmax=1),
                           cmap="jet", shading="auto", rasterized=True)
        ax.axhline(0.5, color="w", ls=":", lw=1)
        ax.set_xlim(-25, 25)
        ax.set_xlabel("k mode number")
        ax.set_ylabel(r"$\omega/\Omega_e$")
        ax.set_title(f"k-$\\omega$ spectra, T = {g0}-{g1} $\\tau_{{gyro}}$")
        fig.colorbar(pc, ax=ax)

    # (c) f(vpar) snapshots
    ax = axes[1, 0]
    NB, VMAX = int(m["nb"]), m["vmax"]
    v = (np.arange(NB) + 0.5) / NB * 2 * VMAX - VMAX
    hfiles = sorted(glob.glob(f"{d}/fhist_*.bin"))
    fh_tg = m["fhist_every"] * dt / tau_g          # tau_gyro per fhist frame
    for tg, cc in ((fh_tg, "k"), (150, "tab:blue"), (400, "tab:green"),
                   (800, "tab:red")):
        i = min(len(hfiles) - 1, max(0, int(tg / fh_tg) - 1))
        h = np.fromfile(hfiles[i], dtype=np.float64)
        ax.semilogy(v, np.maximum(h / h.sum(), 1e-12), color=cc, lw=1.2,
                    label=f"T$\\approx${(i+1)*fh_tg:.0f} $\\tau_g$")
    for s in (+1, -1):
        ax.axvspan(s * 0.08, s * 0.10, color="orange", alpha=0.25)
    ax.set_xlim(-0.35, 0.35); ax.set_ylim(1e-7, 1e-1)
    ax.set_xlabel(r"$v_\parallel/c$"); ax.set_ylabel(r"$f(v_\parallel)$")
    ax.set_title("(c) plateau monitor (shade: $V_p$ band $\\pm$[0.08,0.10]c)")
    ax.legend(fontsize=8)

    # (d) k-integrated spectra early/late + gap metric
    ax = axes[1, 1]
    for (g0, g1), cc in (((50, 100), "tab:blue"), ((700, 800), "tab:red")):
        S, w = kw_window(B, frame(g0), frame(g1), dt_line, wce)
        selw = (w >= 0.02) & (w <= 1.05)
        P = S[selw].sum(axis=1)
        ax.semilogy(w[selw], P / P.max(), color=cc, lw=1.5,
                    label=f"T = {g0}-{g1} $\\tau_g$")
        if g0 == 700:
            wp = w[selw]
            def bp(lo, hi): return P[(wp >= lo) & (wp <= hi)].mean()
            lb, gap, ub = bp(0.25, 0.45), bp(0.47, 0.55), bp(0.58, 0.80)
            print(f"late window: LB {lb:.3e} gap {gap:.3e} UB {ub:.3e}  "
                  f"gap/LB={gap/lb:.2f} gap/UB={gap/ub:.2f}")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\omega/\Omega_e$"); ax.set_ylabel("power (norm)")
    ax.set_title("(d) k-integrated spectrum: early vs late")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
