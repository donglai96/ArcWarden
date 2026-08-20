#!/usr/bin/env python3
"""spec2d — THE canonical warden2d probe spectrogram tool (user audit
2026-08-19; supersedes ad-hoc inline spectra and quicklook2d for physics
claims). Rules baked in:

  1. FRAME INTEGRITY: probes*.bin read in whole (6,7)-float32 records only;
     partial tails reported and dropped.
  2. DETREND: per-window mean removal on B1 and By (post-prebalance static
     baselines are ~10x wave rms — leakage otherwise).
  3. SIGNAL: whistler = B1 + i*By, positive-frequency half of the complex
     FFT (R-mode; measured +/- power ratio 30-140 on P1 data).
  4. COLOR: fixed ABSOLUTE PSD scale across all panels of a figure —
     PSD = |Z|^2 * dt_s / NW (per unit omega/2pi), log10, limits either
     from --vmax or from the global max over panels; NO row whitening,
     NO per-panel scaling (they lift noise into fake bands and erase
     persistent gaps).
  5. RESOLUTION HONESTY: every figure carries BOTH N=1024 and N=4096
     rows; the axis label states df. Horizontal combs at the grid spacing
     are FFT rasterization, not physics.
  6. AXES: left axis common omega/wpe, right axis local omega/Omega_e.
  7. BAND NUMBERS: printed comparisons use bandwidth-normalised mean PSD
     (not bin sums) for LB 0.20-0.45 / gap 0.45-0.55 / UB 0.55-0.75 local.

Usage: spec2d.py <outdir> [--line=1|2] [--lam=5] [--vmax=auto] [out.png]
"""
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1]
line = 1
lam_want = 5.0
vmax_cli = None
png = None
for a in sys.argv[2:]:
    if a.startswith("--line="): line = int(a[7:])
    elif a.startswith("--lam="): lam_want = float(a[6:])
    elif a.startswith("--vmax="): vmax_cli = None if a[7:] == "auto" else float(a[7:])
    elif not a.startswith("-"): png = a
if png is None:
    png = f"{outdir}/spec2d_L{line}_lam{lam_want:+.0f}.png"

meta = open(f"{outdir}/meta.txt").read()
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
pev = int(re.search(r"probe_every (\d+)", meta).group(1))
dt_s = dt * pev
if line == 1:
    tab = re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)
    fn, tag = "probes.bin", "L0"
else:
    tab = re.findall(r"^\s+L2\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)
    fn, tag = "probes2.bin", re.search(r"probe2_L ([\d.]+)", meta).group(1)
lams = [float(a) for a, _ in tab]
ip = lams.index(lam_want)
wce = float(tab[ip][1])

# ---- 1. frame-integrity read ----------------------------------------------
raw = np.fromfile(f"{outdir}/{fn}", dtype=np.float32)
rec = 6 * 7
ns = raw.size // rec
tail = raw.size - ns * rec
if tail:
    print(f"NOTE: {fn} has a partial tail of {tail} floats — dropped")
pr = raw[:ns * rec].reshape(ns, 6, 7)
b1 = pr[:, 4, ip].astype(np.float64)
by = pr[:, 5, ip].astype(np.float64)
print(f"{fn}: {ns} complete records = steps {ns*pev}, t_end {ns*dt_s:.0f}/wpe")

# ---- 2-4. complex STFT with per-window detrend, absolute PSD ---------------
def cstft(NW):
    HOP = max(NW // 16, 16)
    nwin = (ns - NW) // HOP
    w = np.hanning(NW)
    wnorm = (w ** 2).sum()
    nf = NW // 2
    S = np.zeros((nf, nwin))
    for i in range(nwin):
        s1 = b1[i*HOP:i*HOP+NW]; s2 = by[i*HOP:i*HOP+NW]
        seg = (s1 - s1.mean()) + 1j * (s2 - s2.mean())
        Z = np.fft.fft(seg * w)
        S[:, i] = (np.abs(Z[:nf]) ** 2) * dt_s / wnorm     # PSD per (1/wpe)
    om = np.fft.fftfreq(NW, dt_s)[:nf] * 2 * np.pi          # omega/wpe
    tt = (np.arange(nwin) * HOP + NW / 2) * dt_s
    return om, tt, S

panels = [(1024,), (4096,)]
res = [cstft(n[0]) for n in panels]
vmax = vmax_cli
if vmax is None:
    vmax = max(np.log10(S.max() + 1e-30) for _, _, S in res)

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
for ax, (NW,), (om, tt, S) in zip(axes, panels, res):
    P = np.log10(S + 1e-30)
    pc = ax.pcolormesh(tt, om, P, cmap="turbo", vmin=vmax - 5, vmax=vmax)
    ax.set_ylim(0.01, 0.95 * wce)
    ax.axhline(0.5 * wce, color="w", ls="--", lw=0.8)
    df = 2 * np.pi / (NW * dt_s)
    ax.set_ylabel("ω/ωpe")
    ax.set_title(f"{tag} λ={lam_want:+.0f}°  B1+iBy (+freq)  N={NW}  "
                 f"δω={df:.4f} ωpe = {df/wce:.3f} Ωe_local  [abs PSD, log10]")
    ax2 = ax.secondary_yaxis("right", functions=(lambda w_, c=wce: w_ / c,
                                                 lambda w_, c=wce: w_ * c))
    ax2.set_ylabel("ω/Ωe_local")
    fig.colorbar(pc, ax=ax, pad=0.08)
axes[1].set_xlabel("t [1/ωpe]")
fig.tight_layout()
fig.savefig(png, dpi=110)

# ---- 7. bandwidth-normalised band table ------------------------------------
om, tt, S = res[0]
half = S[:, S.shape[1]//2:]
def mean_psd(a, b):
    sel = (om >= a * wce) & (om < b * wce)
    return half[sel].mean()
lb, gp, ub = mean_psd(0.20, 0.45), mean_psd(0.45, 0.55), mean_psd(0.55, 0.75)
print(f"late-half MEAN PSD (bandwidth-normalised): "
      f"LB {lb:.3e}  gap {gp:.3e}  UB {ub:.3e}")
print(f"  gap/LB = {gp/lb:.3f}   UB/LB = {ub/lb:.3f}")
print("wrote", png)
