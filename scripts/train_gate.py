#!/usr/bin/env python3
"""P4-TRAIN gate — FROZEN 2026-08-20 BEFORE the p4_train data existed.

Deck contract: [antenna] tper = 2100, toff = 600, 3 pulses at t = 0, 2100,
4200 (+ trmp 200), run t = 7500.

Readouts (deck warden2d_p4_train.ini pre-registration):
  P-C1 each pulse spawns ONE riser: for >= 2 of 3 pulse windows
       [k*tper + 400, (k+1)*tper + 900] at the +-5 deg stations, the
       ridge (N=1024 STFT, same estimator as element_gate) rises by
       >= 0.10 Omega_e_local with Spearman rho >= 0.7 within the window.
       (Window bars looser than P-B2's single-element run: each element
       has only ~tper of clean track before the next trigger.)
  P-C2 inter-element separation: at +5 deg, mean PSD in the birth band
       (0.27-0.33 local) during the last 300/wpe of each inter-pulse gap
       drops >= 10x below that pulse's element-peak PSD in the same band.
  P-C3 validity: multi-criteria battery — run completed OR healthy()
       emergency stop (report which); wdrms trace recorded, gauss_res
       < 1e-5 while running; finite energies.

Usage: train_gate.py <outdir> [tper] [npulse] [out.png]
"""
import re
import sys

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1]
tper = float(sys.argv[2]) if len(sys.argv) > 2 else 2100.0
npulse = int(sys.argv[3]) if len(sys.argv) > 3 else 3
png = sys.argv[4] if len(sys.argv) > 4 else f"{outdir}/train_gate.png"

meta = open(f"{outdir}/meta.txt").read()
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
pev = int(re.search(r"probe_every (\d+)", meta).group(1))
probes = [(float(a), float(b)) for a, b in
          re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
lams = [p[0] for p in probes]
wce = {p[0]: p[1] for p in probes}
dt_s = dt * pev

raw = np.fromfile(f"{outdir}/probes.bin", dtype=np.float32)
NP, NC = len(probes), 6
ns = raw.size // (NP * NC)
pr = raw[:ns * NP * NC].reshape(ns, NC, NP)
t = (np.arange(ns) + 1) * dt_s
en = np.genfromtxt(f"{outdir}/energy.csv", delimiter=",", names=True)

NW = 1024
hop = NW // 4


def stft(station_lam):
    ip = lams.index(station_lam)
    sig = (pr[:, 3, ip] + 1j * pr[:, 4, ip]).astype(np.complex64)
    nwin = (ns - NW) // hop
    freqs = np.fft.fftfreq(NW, dt_s) * 2 * np.pi
    S = np.empty((nwin, NW))
    tt = np.empty(nwin)
    for j in range(nwin):
        seg = sig[j * hop: j * hop + NW]
        S[j] = np.abs(np.fft.fft((seg - seg.mean()) * np.hanning(NW))) ** 2
        tt[j] = t[j * hop + NW // 2]
    return tt, freqs, S


# ---- P-C1 per-pulse risers -------------------------------------------------
c1_hits, ridge_tracks = [], {}
for lam in (5.0, -5.0):
    tt, freqs, S = stft(lam)
    sel = (freqs > 0.15 * wce[lam]) & (freqs < 0.68 * wce[lam])
    fsel = freqs[sel]
    for k in range(npulse):
        w0, w1 = k * tper + 400, (k + 1) * tper + 900
        m = (tt >= w0) & (tt < w1)
        if m.sum() < 5:
            continue
        fr = fsel[S[m][:, sel].argmax(axis=1)] / wce[lam]
        rise = np.percentile(fr, 90) - np.percentile(fr, 10)
        rho = spearmanr(tt[m], fr).statistic
        ok = (rise >= 0.10) and (rho >= 0.7)
        c1_hits.append((lam, k, rise, rho, ok))
        ridge_tracks[(lam, k)] = (tt[m], fr)
pulse_ok = [any(ok for l_, k_, r_, s_, ok in c1_hits if k_ == k)
            for k in range(npulse)]
c1 = sum(pulse_ok) >= 2

# ---- P-C2 inter-element separation ----------------------------------------
tt, freqs, S = stft(5.0)
sel = (freqs > 0.27 * wce[5.0]) & (freqs < 0.33 * wce[5.0])
band = S[:, sel].mean(axis=1)
c2_ratios = []
for k in range(npulse):
    mpk = (tt >= k * tper) & (tt < (k + 1) * tper + 400)
    gap0, gap1 = (k + 1) * tper - 300, (k + 1) * tper
    mgap = (tt >= gap0) & (tt < gap1)
    if mpk.sum() and mgap.sum():
        c2_ratios.append(band[mpk].max() / max(band[mgap].mean(), 1e-300))
c2 = all(r >= 10 for r in c2_ratios) and len(c2_ratios) >= 2

# ---- P-C3 validity ---------------------------------------------------------
wd_max = np.nanmax(en["wdrms_engine"])
gr_max = np.nanmax(en["gauss_res"])
finite = np.isfinite(en["W_EM"]).all()
completed = en["t"][-1] >= 7400
c3 = finite and (gr_max < 1e-5)

# ---- figure ----------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
a = ax[0, 0]
ip = lams.index(5.0)
ext = [tt[0], tt[-1], 0, freqs[sel := (freqs > 0) & (freqs < 0.8 * wce[5.0])].max() / wce[5.0]]
a.imshow(np.log10(S[:, sel].T + 1e-300), origin="lower", aspect="auto",
         extent=ext, cmap="turbo",
         vmax=np.log10(S[:, sel].max()), vmin=np.log10(S[:, sel].max()) - 5)
for k in range(1, npulse + 1):
    a.axvline(k * tper, color="w", ls=":", lw=0.8)
a.axhline(0.5, color="w", ls="--", lw=0.8)
a.set_title("+5° STFT (N=1024), pulse boundaries dotted")
a.set_xlabel("t"); a.set_ylabel("ω/Ωe_local")
a = ax[0, 1]
for (lam, k), (tk, fk) in ridge_tracks.items():
    if lam == 5.0:
        a.plot(tk, fk, "o-", ms=2, label=f"pulse {k}")
a.axhline(0.5, color="r", ls="--"); a.set_ylim(0.15, 0.7)
a.legend(fontsize=8); a.set_title("per-pulse ridge tracks (+5°)")
a = ax[1, 0]
babs = np.sqrt((pr[:, 3:6, :] ** 2).sum(axis=1)) / 0.2
for lam in (5.0, 10.0, 20.0):
    a.semilogy(t[::10], babs[::10, lams.index(lam)], label=f"+{lam:g}°", lw=0.7)
a.legend(fontsize=8); a.set_title("envelopes"); a.set_xlabel("t")
a = ax[1, 1]
a.plot(en["t"], en["wdrms_engine"], label="wdrms")
a.axhline(0.25, color="r", ls="--")
a2 = a.twinx(); a2.semilogy(en["t"], en["gauss_res"], "C2", label="gauss_res")
a.set_title(f"validity (completed={completed})"); a.set_xlabel("t")
fig.suptitle(f"P4-TRAIN gate: C1 pulses {pulse_ok} [{c1}]  "
             f"C2 ratios {['%.0f' % r for r in c2_ratios]} [{c2}]  "
             f"C3 wd={wd_max:.3f} gr={gr_max:.1e} done={completed} [{c3}]")
fig.tight_layout(); fig.savefig(png, dpi=110)

print(f"P-C1 per-pulse risers : {'PASS' if c1 else 'FAIL'}  "
      + "  ".join(f"p{k}:{'riser' if ok else 'no'}" for k, ok in enumerate(pulse_ok)))
for l_, k_, r_, s_, ok in c1_hits:
    print(f"   lam {l_:+.0f} pulse {k_}: rise {r_:.3f}, spearman {s_:.2f} "
          f"-> {'ok' if ok else 'x'}")
print(f"P-C2 separation      : {'PASS' if c2 else 'FAIL'}  ratios "
      + " ".join(f"{r:.1f}" for r in c2_ratios))
print(f"P-C3 validity        : {'PASS' if c3 else 'FAIL'}  wdrms {wd_max:.3f}"
      f" gauss {gr_max:.2e} completed {completed}")
print(f"figure: {png}")
