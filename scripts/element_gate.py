#!/usr/bin/env python3
"""P4-EL1 element gate — FROZEN 2026-08-20 BEFORE the ARM-1 data existed.

Readouts (deck warden2d_p4_el1.ini pre-registration):
  P-B1 seed      : dB/B0eq at +-5 deg in [3e-3, 6e-3] at t ~ 800
                   (window 700-900, mean of the two stations).
  P-B2 element   : ridge track at +5 deg (N=1024 STFT of B1+iBy, +freq):
                   per-window peak frequency within 0.15-0.65 local;
                   require (a) total rise >= 0.13 Omega_e_local from the
                   trigger band (0.30 +- 0.03) i.e. endpoint >= 0.45 local
                   at ANY station incl +-10; (b) monotonic within noise
                   (Spearman rho of ridge vs t >= 0.8 over the rise
                   interval); (c) median instantaneous ridge width
                   (-10 dB from ridge peak) < 0.10 Omega_e.
  P-B3 transport : mean dB/B0eq over the last 500/wpe >= 3e-3 at +-10 deg
                   AND >= 1e-3 at +-20 deg.
  P-B4 validity  : max wdrms < 0.25, max gauss_res < 1e-5, energy.csv
                   finite throughout.

Usage: element_gate.py <outdir> [out.png]
"""
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1]
png = sys.argv[2] if len(sys.argv) > 2 else f"{outdir}/element_gate.png"

meta = open(f"{outdir}/meta.txt").read()
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
pev = int(re.search(r"probe_every (\d+)", meta).group(1))
B0eq = 0.2
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
babs = np.sqrt((pr[:, 3:6, :] ** 2).sum(axis=1)) / B0eq

en = np.genfromtxt(f"{outdir}/energy.csv", delimiter=",", names=True)

# ---- P-B1 seed -------------------------------------------------------------
m = (t >= 700) & (t < 900)
i5 = [lams.index(5.0), lams.index(-5.0)]
seed = float(babs[m][:, i5].mean())
b1 = 3e-3 <= seed <= 6e-3

# ---- P-B2 ridge track ------------------------------------------------------
NW = 1024


def ridge(station_lam):
    ip = lams.index(station_lam)
    sig = (pr[:, 3, ip] + 1j * pr[:, 4, ip]).astype(np.complex64)
    hop = NW // 4
    nwin = (ns - NW) // hop
    fr, tt, wid = [], [], []
    freqs = np.fft.fftfreq(NW, dt_s) * 2 * np.pi
    sel = (freqs > 0.15 * wce[station_lam]) & (freqs < 0.68 * wce[station_lam])
    for j in range(nwin):
        seg = sig[j * hop: j * hop + NW]
        seg = seg - seg.mean()
        S = np.abs(np.fft.fft(seg * np.hanning(NW))) ** 2
        Ssel = S[sel]
        pk = Ssel.max()
        fr.append(freqs[sel][Ssel.argmax()] / wce[station_lam])
        above = freqs[sel][Ssel > pk * 0.1]
        wid.append((above.max() - above.min()) / wce[station_lam]
                   if len(above) > 1 else 0.0)
        tt.append(t[j * hop + NW // 2])
    return np.array(tt), np.array(fr), np.array(wid), \
        np.array([np.abs(sig[j * hop: j * hop + NW]).mean()
                  for j in range(nwin)])


results = {}
for lam in (5.0, 10.0, -5.0, -10.0):
    results[lam] = ridge(lam)

# element interval: from first window with envelope > 20% of station max
best_end, best_lam = 0.0, None
rise_ok = mono_ok = width_ok = False
for lam, (tt, fr, wid, env) in results.items():
    on = env > 0.2 * env.max()
    if on.sum() < 6:
        continue
    fr_on, tt_on, wid_on = fr[on], tt[on], wid[on]
    end = np.percentile(fr_on, 95)
    if end > best_end:
        best_end, best_lam = end, lam
        from scipy.stats import spearmanr
        rho = spearmanr(tt_on, fr_on).statistic
        rise_ok = end >= 0.45
        mono_ok = rho >= 0.8
        width_ok = np.median(wid_on) < 0.10
b2 = rise_ok and mono_ok and width_ok

# ---- P-B3 transport --------------------------------------------------------
mend = t >= t[-1] - 500
a10 = babs[mend][:, [lams.index(10.0), lams.index(-10.0)]].mean()
a20 = babs[mend][:, [lams.index(20.0), lams.index(-20.0)]].mean()
b3 = (a10 >= 3e-3) and (a20 >= 1e-3)

# ---- P-B4 validity ---------------------------------------------------------
wd_max = np.nanmax(en["wdrms_engine"])
gr_max = np.nanmax(en["gauss_res"])
finite = np.isfinite(en["W_EM"]).all()
b4 = (wd_max < 0.25) and (gr_max < 1e-5) and finite

# ---- figure ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
a = ax[0, 0]
for lam in (5.0, 10.0):
    tt, fr, wid, env = results[lam]
    on = env > 0.2 * env.max()
    a.plot(tt[on], fr[on], "o-", ms=3, label=f"+{lam:g}° ridge")
a.axhline(0.45, color="r", ls="--", alpha=0.6)
a.axhspan(0.27, 0.33, color="g", alpha=0.15, label="trigger band")
a.set_ylim(0.1, 0.7); a.set_xlabel("t"); a.set_ylabel("ridge ω/Ωe_local")
a.legend(fontsize=8); a.set_title("P-B2 ridge track")
a = ax[0, 1]
for lam in (0.0, 5.0, 10.0, 20.0):
    a.semilogy(t[::10], babs[::10, lams.index(lam)], label=f"+{lam:g}°")
a.axhline(3e-3, color="k", ls=":", alpha=0.5)
a.set_xlabel("t"); a.set_ylabel("dB/B0eq"); a.legend(fontsize=8)
a.set_title("P-B1/B3 envelopes")
a = ax[1, 0]
a.plot(en["t"], en["wdrms_engine"]); a.axhline(0.25, color="r", ls="--")
a.set_xlabel("t"); a.set_title("P-B4 wdrms")
a = ax[1, 1]
a.semilogy(en["t"], en["gauss_res"]); a.axhline(1e-5, color="r", ls="--")
a.set_xlabel("t"); a.set_title("P-B4 gauss_res")
fig.suptitle(f"P4-EL1 gate: B1 seed={seed:.2e} [{b1}]  "
             f"B2 rise {best_end:.3f}@{best_lam}° mono={mono_ok} "
             f"width={width_ok} [{b2}]  B3 10°={a10:.2e} 20°={a20:.2e} "
             f"[{b3}]  B4 wd={wd_max:.3f} gr={gr_max:.1e} [{b4}]")
fig.tight_layout(); fig.savefig(png, dpi=110)

for tag, ok, txt in [
        ("P-B1 seed", b1, f"{seed:.3e} (window 3e-3..6e-3)"),
        ("P-B2 element", b2, f"end {best_end:.3f} local @ {best_lam}° "
         f"(>=0.45: {rise_ok}), mono {mono_ok}, width {width_ok}"),
        ("P-B3 transport", b3, f"10°: {a10:.3e} (>=3e-3), 20°: {a20:.3e} (>=1e-3)"),
        ("P-B4 validity", b4, f"wdrms {wd_max:.3f} (<0.25), "
         f"gauss {gr_max:.2e} (<1e-5), finite {finite}")]:
    print(f"{tag:15s}: {'PASS' if ok else 'FAIL'}  {txt}")
print(f"figure: {png}")
