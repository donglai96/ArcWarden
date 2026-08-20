#!/usr/bin/env python3
"""Sharp time-frequency analysis of a probe signal: (1) fine-hop STFT,
(2) REASSIGNED spectrogram (Auger-Flandrin: energy moved to the local
instantaneous frequency/group delay — sharpens ridges well below the
naive df = 2pi/T bin width), (3) Hilbert instantaneous frequency of the
band-passed signal where the envelope is strong. Probe cadence gives
Nyquist = pi/(dt*probe_every) = 5.2 wpe >> Omega_e: sampling is never
the limit; window time-bandwidth is — reassignment attacks exactly that.

Usage: chorus_sharp2d.py <outdir> [lam_deg] [t0] [t1] [out.png]
"""
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1] if len(sys.argv) > 1 else "build/v4r5"
lam_want = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else 3500.0
T1 = float(sys.argv[4]) if len(sys.argv) > 4 else 7500.0
png = sys.argv[5] if len(sys.argv) > 5 else f"{outdir}/chorus_sharp.png"

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
pidx = [i for i, (l, _) in enumerate(probes) if l == lam_want][0]
wce = probes[pidx][1]
sig = pr[:, 4, pidx].astype(np.float64)

i0, i1 = int(T0 / dt_s), min(ns, int(T1 / dt_s))
x = sig[i0:i1]
nx = len(x)

NW, HOP = 512, 4
n = np.arange(NW)
w = np.hanning(NW)
dw = np.gradient(w)                       # per-sample derivative window
tw = (n - NW / 2) * w
nwin = (nx - NW) // HOP
F = np.fft.rfftfreq(NW) * 2 * np.pi       # rad/sample
nf = len(F)
Sw = np.zeros((nf, nwin), complex)
Sd = np.zeros_like(Sw)
St = np.zeros_like(Sw)
for i in range(nwin):
    seg = x[i * HOP:i * HOP + NW]
    Sw[:, i] = np.fft.rfft(seg * w)
    Sd[:, i] = np.fft.rfft(seg * dw)
    St[:, i] = np.fft.rfft(seg * tw)
P = np.abs(Sw)**2
eps = P.max() * 1e-12

# reassignment coordinates (rad/sample and samples). SIGN FIX 2026-08-19
# (user audit): with NumPy's e^{-i w t} forward-FFT convention the operators
# are w_hat = F - Im(Sd conj(Sw))/P and t_hat = center + Re(St conj(Sw))/P;
# the previous signs scattered energy AWAY from ridges — all reassigned
# spectra/trajectories produced before this date must be recomputed.
with np.errstate(divide="ignore", invalid="ignore"):
    w_hat = F[:, None] - np.imag(Sd * np.conj(Sw)) / (P + eps)
    t_hat = (np.arange(nwin) * HOP + NW / 2)[None, :] \
        + np.real(St * np.conj(Sw)) / (P + eps)

# physical units
om_grid = F / dt_s / wce                  # w/wce rows of the plain STFT
om_hat = w_hat / dt_s / wce
tt_hat = (i0 + t_hat) * dt_s
tt = (i0 + np.arange(nwin) * HOP + NW / 2) * dt_s

# accumulate reassigned energy on a fine grid
fbins = np.linspace(0.05, 0.85, 320)
tbins = np.linspace(T0, T1, 900)
sel = (P > np.percentile(P, 92)) & (om_hat > 0.05) & (om_hat < 0.85)
H, _, _ = np.histogram2d(tt_hat[sel], om_hat[sel], bins=[tbins, fbins],
                         weights=P[sel])

# Hilbert instantaneous frequency of the LB band
X = np.fft.fft(x)
fw = np.fft.fftfreq(nx, dt_s) * 2 * np.pi
mask = (fw > 0.12 * wce) & (fw < 0.58 * wce)   # positive freqs only -> analytic
an = np.fft.ifft(X * mask) * 2
env = np.abs(an)
ph = np.unwrap(np.angle(an))
iff = np.gradient(ph) / dt_s / wce
ks = 41
iff_s = np.convolve(iff, np.ones(ks) / ks, "same")
tif = (i0 + np.arange(nx)) * dt_s
strong = env > 1.3 * np.median(env)

fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
mplot = (om_grid > 0.05) & (om_grid < 0.85)
v = np.log10(P[mplot] + 1e-24)
pc = axs[0].pcolormesh(tt, om_grid[mplot], v, cmap="turbo",
                       vmin=np.percentile(v, 60), vmax=np.percentile(v, 99.9),
                       shading="auto", rasterized=True)
axs[0].set_title(f"plain STFT, win {NW*dt_s:.0f}/ωpe, hop {HOP*dt_s:.1f}/ωpe "
                 f"(df = {2*np.pi/(NW*dt_s)/wce:.3f} Ωe)")
plt.colorbar(pc, ax=axs[0])
vH = np.log10(H.T + H.max() * 1e-6)
pc = axs[1].pcolormesh(0.5 * (tbins[1:] + tbins[:-1]),
                       0.5 * (fbins[1:] + fbins[:-1]), vH, cmap="turbo",
                       vmin=np.percentile(vH, 75), vmax=vH.max(),
                       shading="auto", rasterized=True)
axs[1].set_title("REASSIGNED spectrogram (same windows, energy at local "
                 "instantaneous frequency)")
plt.colorbar(pc, ax=axs[1])
sc = axs[2].scatter(tif[strong][::9], iff_s[strong][::9], s=2.5,
                    c=np.log10(env[strong][::9]), cmap="turbo")
axs[2].set_ylim(0.05, 0.85)
axs[2].set_title("Hilbert instantaneous frequency of 0.12–0.58 Ωe band "
                 "(envelope > 1.3× median, color = log amplitude)")
plt.colorbar(sc, ax=axs[2])
for ax in axs:
    ax.axhline(0.5, color="w", ls=":", lw=0.9)
    ax.set_ylabel(r"$\omega/\Omega_{e,\rm local}$")
axs[2].set_xlabel(r"$t$ [$1/\omega_{pe}$]")
fig.suptitle(f"probe {lam_want:+.0f}° — sampling Δt {dt_s:.1f}/ωpe "
             f"(Nyquist {np.pi/dt_s:.1f} ωpe = {np.pi/dt_s/wce:.0f} Ωe: "
             "not the limit)", y=0.995)
fig.tight_layout()
fig.savefig(png, dpi=140)
print(f"wrote {png}")
