"""Measured riser sweep rates (ridge tracking on probe STFT) vs the Omura
nonlinear theory optimum dw/dt = 0.4 s0 w Omega_w / s1 at the equator
(S = -0.4; relativistic V_R; cold parallel whistler dispersion, local
wce from meta; local wpe = 1 by construction nc+nh = 1)."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
outdir = sys.argv[1] if len(sys.argv) > 1 else "build/v4r5"
meta = open(f"{outdir}/meta.txt").read()
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
pev = int(re.search(r"probe_every (\d+)", meta).group(1))
probes = [(float(a), float(b)) for a, b in
          re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
dt_s = dt * pev
raw = np.fromfile(f"{outdir}/probes.bin", dtype=np.float32)
NP = len(probes)
ns = raw.size // (NP * 6)
pr = raw[:ns * NP * 6].reshape(ns, 6, NP)

pidx = [i for i, (l, _) in enumerate(probes) if l == 5.0][0]
wce = probes[pidx][1]
b1, by = pr[:, 4, pidx], pr[:, 5, pidx]

# ---- STFT ridge ----
# NW=384 RETIRED (user audit 2026-08-19): at dt_s=0.6 it gave df=0.133 We
# — only ~3 independent bins across the band while the sweep gate was
# 0.05 We, so single-bin hops were scored as risers; every chirp-rate/Omura
# ratio produced with NW=384 is non-quantitative. NW=1024 (df=0.05 We) and
# the span gate below now requires >= 3 grid cells of genuine sweep.
NW, HOP = 1024, 24
w = np.hanning(NW)
nwin = (ns - NW) // HOP
fr = np.fft.rfftfreq(NW, dt_s) * 2 * np.pi          # omega in wpe units
S = np.zeros((NW // 2 + 1, nwin))
for i in range(nwin):
    S[:, i] = np.abs(np.fft.rfft(b1[i * HOP:i * HOP + NW] * w))**2
tw = (np.arange(nwin) * HOP + NW / 2) * dt_s
band = (fr > 0.15 * wce) & (fr < 0.55 * wce)
fb = fr[band]
Sb = S[band]
ridge_w = fb[np.argmax(Sb, axis=0)]                  # peak omega per column
ridge_p = Sb.max(axis=0)
alive = ridge_p > np.percentile(ridge_p, 70)         # strong-emission columns

# contiguous alive segments, keep monotone-ish rising stretches >= 8 columns
segs = []
i = 0
while i < nwin:
    if not alive[i]:
        i += 1
        continue
    j = i
    while j + 1 < nwin and alive[j + 1]:
        j += 1
    if j - i >= 8:
        segs.append((i, j))
    i = j + 1

fits = []
for i0, j0 in segs:
    t = tw[i0:j0 + 1]
    om = ridge_w[i0:j0 + 1]
    # split segment at large downward jumps (element boundaries)
    brk = np.where(np.diff(om) < -0.06 * wce)[0]
    parts = np.split(np.arange(len(t)), brk + 1)
    for p in parts:
        if len(p) < 8:
            continue
        A = np.polyfit(t[p], om[p], 1)
        rate = A[0]                                   # dw/dt, wpe^2
        if rate <= 0:
            continue
        span = om[p].max() - om[p].min()
        # a real sweep must exceed BOTH 0.05 We and 3 FFT grid cells
        # (grid-hop immunity, user audit 2026-08-19)
        dgrid = 2 * np.pi / (NW * dt_s)
        if span < max(0.05 * wce, 3 * dgrid):
            continue
        # B_w in the same window: bandpassed transverse amplitude
        s0i, s1i = i0 * HOP + p[0] * HOP, i0 * HOP + p[-1] * HOP + NW
        F1, Fy = np.fft.rfft(b1[s0i:s1i]), np.fft.rfft(by[s0i:s1i])
        fw = np.fft.rfftfreq(s1i - s0i, dt_s) * 2 * np.pi
        keep = (fw > 0.15 * wce) & (fw < 0.55 * wce)
        bw = np.sqrt(2 * (np.mean(np.abs(np.fft.irfft(F1 * keep, s1i - s0i))**2)
                          + np.mean(np.abs(np.fft.irfft(Fy * keep, s1i - s0i))**2)))
        fits.append((t[p][0], t[p][-1], rate, np.mean(om[p]), bw))

# ---- Omura theory ----
def theory_rate(om, bw, wce, vperp0=0.30):
    xi2 = om * (wce - om)                             # wpe = 1
    xi = np.sqrt(xi2)
    dl = 1.0 / np.sqrt(1.0 + xi2)
    k = om / (xi * dl)
    gam = 1.0
    for _ in range(20):                               # relativistic V_R
        vr = (om - wce / gam) / k
        gam = 1.0 / np.sqrt(max(1e-9, 1.0 - vr**2 - vperp0**2))
    dk = 1e-4
    om_of_k = lambda kk: None
    # numerical group velocity from the cold parallel dispersion k(om)
    dom = 1e-5
    k2 = (om + dom) / (np.sqrt((om + dom) * (wce - om - dom)) /
                       np.sqrt(1 + (om + dom) * (wce - om - dom)))
    vg = dom / (k2 - k)
    s0 = dl * vperp0 / xi
    s1 = gam * (1.0 - vr / vg)**2
    return 0.4 * s0 * om * bw / s1, vr, vg, s0, s1, gam

print(f"probe +5deg: wce_local = {wce:.4f} (wpe=1)   risers kept: {len(fits)}")
print(f"{'t range':>16} {'dw/dt meas':>12} {'<w>/wce':>8} {'Bw':>9} "
      f"{'dw/dt theo':>12} {'meas/theo':>10}")
rows = []
for t0, t1, rate, omm, bw in fits:
    th, vr, vg, s0, s1, gam = theory_rate(omm, bw, wce)
    rows.append((t0, t1, rate, omm, bw, th))
    print(f"[{t0:6.0f},{t1:6.0f}] {rate:12.3e} {omm/wce:8.3f} {bw:9.2e} "
          f"{th:12.3e} {rate/th:10.2f}")
m = np.array([[r[2], r[5]] for r in rows])
if len(m):
    print(f"\nmean measured {m[:,0].mean():.3e} wpe^2 = "
          f"{m[:,0].mean()/wce**2:.3e} wce^2")
    print(f"mean theory   {m[:,1].mean():.3e} wpe^2 = "
          f"{m[:,1].mean()/wce**2:.3e} wce^2")
    print(f"ratio mean {np.mean(m[:,0]/m[:,1]):.2f}  "
          f"median {np.median(m[:,0]/m[:,1]):.2f}")
ex = theory_rate(0.3 * wce, 5e-3, wce)
print(f"\nreference point w=0.3wce Bw=5e-3: VR={ex[1]:.3f}c Vg={ex[2]:.3f}c "
      f"s0={ex[3]:.2f} s1={ex[4]:.2f} gam={ex[5]:.3f}")

# ---- figure ----
fig, ax = plt.subplots(figsize=(13, 5.5))
mplot = (fr > 0.08 * wce) & (fr < 0.85 * wce)
v = np.log10(S[mplot] + 1e-24)
ax.pcolormesh(tw, fr[mplot] / wce, v, cmap="turbo",
              vmin=np.percentile(v, 60), vmax=np.percentile(v, 99.8),
              shading="auto", rasterized=True)
ok = alive & (ridge_w > 0.15 * wce)
ax.plot(tw[ok], ridge_w[ok] / wce, "w.", ms=2, alpha=0.6)
for t0, t1, rate, omm, bw, th in rows:
    tt = np.array([t0, t1])
    ax.plot(tt, (omm + rate * (tt - 0.5 * (t0 + t1))) / wce, "w-", lw=2.2)
    ax.plot(tt, (omm + th * (tt - 0.5 * (t0 + t1))) / wce, "m--", lw=2.2)
ax.axhline(0.5, color="w", ls=":", lw=0.8)
ax.set_xlabel(r"$t$ [$1/\omega_{pe}$]")
ax.set_ylabel(r"$\omega/\Omega_{e,\rm local}$")
ax.set_title("V4R5-A5 riser sweep rates, +5° probe — white: measured ridge fit, "
             "magenta dashed: Omura S=−0.4 optimum at same $B_w$")
fig.tight_layout()
fig.savefig(f"{outdir}/chirp_rate_vs_theory.png", dpi=145)
print(f"wrote {outdir}/chirp_rate_vs_theory.png")
