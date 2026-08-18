#!/usr/bin/env python3
"""delta-f vs full-f cross-validation verdict (pre-registered gates in the
V4R6-XVAL deck headers): (1) linear growth rate of W_EM within 20%;
(2) same spectral morphology at matched W_EM; (3) delta-f wd_rms << 1
through the linear phase. Usage: xval_compare2d.py <ff_dir> <df_dir>"""
import csv
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ffd = sys.argv[1] if len(sys.argv) > 1 else "build/xval_ff"
dfd = sys.argv[2] if len(sys.argv) > 2 else "build/xval_df"


def energy(d):
    t, w, wd = [], [], []
    with open(f"{d}/energy.csv") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            w.append(float(row["W_EM"]))
            wd.append(float(row.get("wdrms_engine", 0)))
    return np.array(t), np.array(w), np.array(wd)


def growth_rate(t, w):
    # steepest sustained dlnW/dt over a 1000/wpe sliding window
    ln = np.log(w)
    best, t0b = 0.0, 0.0
    span = 1000.0
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + span)
        if j >= len(t):
            break
        g = (ln[j] - ln[i]) / (t[j] - t[i])
        if g > best:
            best, t0b = g, t[i]
    return best, t0b


def spec(d, wsel):
    meta = open(f"{d}/meta.txt").read()
    dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
    pev = int(re.search(r"probe_every (\d+)", meta).group(1))
    probes = [(float(a), float(b)) for a, b in
              re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
    dt_s = dt * pev
    raw = np.fromfile(f"{d}/probes.bin", dtype=np.float32)
    NP = len(probes)
    ns = raw.size // (NP * 6)
    pr = raw[:ns * NP * 6].reshape(ns, 6, NP)
    pidx = [i for i, (l, _) in enumerate(probes) if l == 5.0][0]
    sig = pr[:, 4, pidx].astype(np.float64)
    i0, i1 = int(wsel[0] / dt_s), min(ns, int(wsel[1] / dt_s))
    x = sig[i0:i1] * np.hanning(i1 - i0)
    F = np.abs(np.fft.rfft(x))**2
    fr = np.fft.rfftfreq(i1 - i0, dt_s) * 2 * np.pi / probes[pidx][1]
    return fr, F


tf, wf, _ = energy(ffd)
td, wd_, wdr = energy(dfd)
gf, t0f = growth_rate(tf, wf)
gd, t0d = growth_rate(td, wd_)
print(f"full-f: gamma_W = {gf:.3e} /wpe (window from t={t0f:.0f})")
print(f"delta-f: gamma_W = {gd:.3e} /wpe (window from t={t0d:.0f})")
print(f"ratio df/ff = {gd/gf:.3f}  -> gate |1-r|<0.2: "
      f"{'PASS' if abs(1 - gd/gf) < 0.2 else 'FAIL'}")
lin = td < t0d + 1000
print(f"delta-f wd_rms at end of linear window: {wdr[lin][-1]:.3e} "
      f"(gate << 1: {'PASS' if wdr[lin][-1] < 0.3 else 'FAIL'})")

# spectra in each arm's own growth window (matched dynamics, not clock)
frf, Ff = spec(ffd, (t0f, t0f + 1500))
frd, Fd = spec(dfd, (t0d, t0d + 1500))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
axs[0].semilogy(tf, wf, "k-", label="full-f")
axs[0].semilogy(td, wd_, "r-", label="δf")
for t0, g, c in [(t0f, gf, "k"), (t0d, gd, "r")]:
    tt = np.linspace(t0, t0 + 1000, 10)
    w0 = np.interp(t0, [tf, td][c == "r"], [wf, wd_][c == "r"])
    axs[0].semilogy(tt, w0 * np.exp(g * (tt - t0)), c + "--", lw=1)
axs[0].set_xlabel(r"$t$")
axs[0].set_ylabel(r"$W_{EM}$")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[0].set_title(f"growth: γ_W ff {gf:.2e}, δf {gd:.2e} (ratio {gd/gf:.2f})")
m = (frf > 0.05) & (frf < 0.9)
axs[1].semilogy(frf[m], Ff[m] / Ff[m].max(), "k-", lw=0.8, label="full-f")
md = (frd > 0.05) & (frd < 0.9)
axs[1].semilogy(frd[md], Fd[md] / Fd[md].max(), "r-", lw=0.8, alpha=0.8,
                label="δf")
axs[1].axvline(0.5, color="b", ls=":")
axs[1].set_xlabel(r"$\omega/\Omega_{e,\rm local}$")
axs[1].set_title("B1 spectrum, each arm's growth window (normalized)")
axs[1].legend()
axs[2].plot(td, wdr, "r-")
axs[2].axvspan(t0d, t0d + 1000, color="y", alpha=0.2)
axs[2].set_xlabel(r"$t$")
axs[2].set_ylabel("wd_rms")
axs[2].set_title("δf weight amplitude (shaded: linear window)")
axs[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{dfd}/xval_verdict.png", dpi=140)
print(f"wrote {dfd}/xval_verdict.png")
