#!/usr/bin/env python3
"""G-source screening verdict vs RSM_TEST_PLAN §3 bar (7 conditions).

Usage: g1_source_verdict.py DIR [--pref PREFIX]

Checks on the RSM-off screening run itself:
  1/2/3. connected rising ridge, sweep >=0.15, crossing 0.50  (spectrogram +
         x4_omega_stop element table — eyeball panel 1)
  4. upper shoulder 0.55-0.60 detectable (band power vs noise bands)
  5. NO natural valley 0.46-0.56 (V = P_gap/sqrt(P_lo*P_up) on the control)
  6. probe time-delay consistency (eq vs +-145/+-291 burst onset)
  7. amplitude scale (Bw/B0 from element envelope; printed, judged vs x4 base)
"""
import argparse, os, sys
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import load_probe, stft


def psd_bands(d, off, t0, t1, wce):
    m, t, by, bz = load_probe(d, off)
    sel = (t * wce >= t0) & (t * wce <= t1)
    dt_s = t[1] - t[0]
    fs = 1.0 / dt_s
    P = 0
    for s in (by[sel], bz[sel]):
        f, p = welch(s, fs=fs, nperseg=4096)
        P = P + p
    w = 2 * np.pi * f / wce
    bands = {"lo": (0.35, 0.45), "gap": (0.46, 0.56), "up": (0.55, 0.65),
             "sh": (0.55, 0.60), "hi": (0.65, 0.75)}
    out = {k: P[(w >= a) & (w <= b)].mean() for k, (a, b) in bands.items()}
    return w, P, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--pref", default=None)
    a = ap.parse_args()
    pref = a.pref or os.path.basename(a.dir.rstrip("/"))

    m, t, by, bz = load_probe(a.dir, 0.0)
    wce = m["wce"]
    sig = (by + 1j * bz) / wce
    nwin, hop = 1024, 64
    spec = stft(sig, nwin, hop)
    dt_s = t[1] - t[0]
    freqs = np.fft.fftfreq(nwin, d=dt_s) * 2 * np.pi / wce
    tt = (t[0] + (np.arange(spec.shape[0]) * hop + nwin / 2) * dt_s) * wce
    fsel = (freqs >= 0.05) & (freqs <= 0.9)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    ax = axs[0]
    pc = ax.pcolormesh(tt, freqs[fsel], np.log10(np.abs(spec[:, fsel].T) + 1e-8),
                       shading="auto", cmap="jet", vmin=-5)
    fig.colorbar(pc, ax=ax)
    for yv in (0.46, 0.50, 0.56, 0.60):
        ax.axhline(yv, color="w", lw=0.6, ls="--")
    ax.set(xlabel=r"t $\Omega_e$", ylabel=r"$\omega/\Omega_e$",
           title=f"{pref} equator STFT")

    # burst window = where |B| envelope > 20% of max
    env = np.abs(spec[:, fsel]).max(axis=1)
    thr = 0.2 * env.max()
    ion = np.where(env > thr)[0]
    t0, t1 = tt[ion[0]], tt[ion[-1]]
    print(f"# burst window [{t0:.0f}, {t1:.0f}] /We0  (env>20% max)")

    ax = axs[1]
    print("# equator band powers (burst window):")
    w, P, b = psd_bands(a.dir, 0.0, t0, t1, wce)
    ax.semilogy(w, P, lw=0.8)
    for k, (x0, x1) in {"lo": (0.35, 0.45), "gap": (0.46, 0.56),
                        "sh": (0.55, 0.60), "hi": (0.65, 0.75)}.items():
        ax.axvspan(x0, x1, alpha=0.12)
    ax.set(xlim=(0, 1.0), xlabel=r"$\omega/\Omega_e$",
           title="eq Welch PSD (burst)")
    V = b["gap"] / np.sqrt(b["lo"] * b["up"])
    print(f"  lo={b['lo']:.3e} gap={b['gap']:.3e} up={b['up']:.3e} "
          f"sh={b['sh']:.3e} hi(noise)={b['hi']:.3e}")
    print(f"  bar5 natural-valley V = gap/sqrt(lo*up) = {V:.3f}  "
          f"({'OK no valley' if V >= 0.7 else 'NATURAL VALLEY — FAIL'})")
    print(f"  bar4 upper shoulder: sh/lo = {b['sh']/b['lo']:.4f}  "
          f"sh/hi-noise = {b['sh']/b['hi']:.2f}")

    # 6. probe delay: burst onset vs |lat|
    ax = axs[2]
    print("# bar6 onset time (env>20% of its own max) vs probe:")
    for off, lab in ((0.0, "eq"), (145.2, "+145"), (-145.2, "-145"),
                     (291.0, "+291"), (-291.0, "-291")):
        mm, tp, py, pz = load_probe(a.dir, off)
        s2 = stft((py + 1j * pz) / wce, nwin, hop)
        e2 = np.abs(s2[:, fsel]).max(axis=1)
        tt2 = (tp[0] + (np.arange(s2.shape[0]) * hop + nwin / 2) * dt_s) * wce
        on = tt2[np.where(e2 > 0.2 * e2.max())[0][0]]
        ax.plot(tt2, e2 / wce, lw=0.7, label=f"{lab} on={on:.0f}")
        print(f"  {lab:5s}: onset {on:6.0f}  peak {e2.max()/wce:.2e}")
    ax.legend(fontsize=7); ax.set(xlabel=r"t $\Omega_e$", title=r"$|B_w|/B_0$ envelope")
    fig.savefig(f"{pref}_verdict.png", dpi=130)
    print(f"fig -> {pref}_verdict.png")


if __name__ == "__main__":
    main()
