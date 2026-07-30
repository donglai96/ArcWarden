#!/usr/bin/env python3
"""REFRESH Gate 3 A/B: closed vs refreshed Chen-case2 (x=damping, same seed).

Panels: (a,b) equator-probe STFT spectrograms; (c) |B_perp| envelopes;
(d) sliding-window ACF period of the envelope (the judge: closed should
stretch/die late, refreshed should stay stationary); (e) WB(t) + refresh
injection rate.  Usage: plot_gate3_refresh.py <closed_dir> <refresh_dir> <out.png>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WCE = 0.2          # wpe units
PROBE_EVERY = 10
DT = 0.15
NPROBE = 5
EQ = 2             # equator probe index

def load_probe(d):
    x = np.fromfile(f'{d}/probe.bin', dtype=np.float32)
    n = x.size // (2 * NPROBE)
    x = x[:n * 2 * NPROBE].reshape(n, NPROBE, 2)
    t = np.arange(n) * PROBE_EVERY * DT * WCE          # time in 1/We0
    return t, x[:, EQ, 0] + 1j * x[:, EQ, 1]           # By + iBz

def stft(sig, t, nwin=2048, nhop=256):
    wins, times = [], []
    w = np.hanning(nwin)
    for i0 in range(0, len(sig) - nwin, nhop):
        wins.append(np.fft.fft(w * sig[i0:i0 + nwin]))
        times.append(t[i0 + nwin // 2])
    S = np.abs(np.array(wins)).T ** 2
    f = np.fft.fftfreq(nwin, d=PROBE_EVERY * DT)       # wpe units
    om = 2 * np.pi * f / WCE                           # w/We0
    sel = (om >= 0.05) & (om <= 0.85)
    return np.array(times), om[sel], S[sel]

def envelope(sig, t, tau=25.0):
    # |B| smoothed over tau/We0
    dt = t[1] - t[0]
    k = max(1, int(tau / dt))
    ker = np.ones(k) / k
    return np.convolve(np.abs(sig), ker, mode='same')

def acf_period(env, t, twin=2500.0, thop=250.0, pmin=200.0, pmax=2500.0):
    """sliding-window ACF: first significant peak lag -> period(t)."""
    dt = t[1] - t[0]
    nw, nh = int(twin / dt), int(thop / dt)
    out_t, out_p = [], []
    for i0 in range(0, len(env) - nw, nh):
        seg = env[i0:i0 + nw] - env[i0:i0 + nw].mean()
        if seg.std() < 1e-12:
            continue
        ac = np.correlate(seg, seg, 'full')[nw - 1:]
        ac /= ac[0]
        lags = np.arange(nw) * dt
        sel = (lags >= pmin) & (lags <= pmax)
        acs, ls = ac[sel], lags[sel]
        k = np.argmax(acs)
        if acs[k] > 0.15:                              # significant periodicity
            out_t.append(t[i0 + nw // 2]); out_p.append(ls[k])
    return np.array(out_t), np.array(out_p)

def load_energy(d):
    a = np.genfromtxt(f'{d}/energy.csv', delimiter=',', names=True)
    return a['time'] * WCE, a['WB']

closed, refresh, out = sys.argv[1], sys.argv[2], sys.argv[3]
tc, sc = load_probe(closed)
tr, sr = load_probe(refresh)

fig, axs = plt.subplots(5, 1, figsize=(13, 16),
                        gridspec_kw={'height_ratios': [2, 2, 1.2, 1.2, 1.2]})

for ax, (t, s), lab in [(axs[0], (tc, sc), 'closed'),
                        (axs[1], (tr, sr), 'refreshed')]:
    ts, om, S = stft(s, t)
    ax.pcolormesh(ts, om, np.log10(S + 1e-16), cmap='jet',
                  vmin=np.log10(S.max()) - 5, vmax=np.log10(S.max()))
    ax.set_ylabel(r'$\omega/\Omega_{e0}$')
    ax.set_title(f'{lab}: equator probe $B_y+iB_z$ spectrogram')
    ax.axhline(0.5, color='w', lw=0.5, ls='--')

ec, er = envelope(sc, tc), envelope(sr, tr)
axs[2].semilogy(tc, ec, 'k', lw=0.7, label='closed')
axs[2].semilogy(tr, er, 'r', lw=0.7, alpha=0.75, label='refreshed')
axs[2].set_ylabel(r'$|B_\perp|$ env'); axs[2].legend(loc='lower right')
axs[2].set_title('wave envelope at the equator')

tc2, pc = acf_period(ec, tc)
tr2, pr = acf_period(er, tr)
axs[3].plot(tc2, pc, 'ko-', ms=3, label='closed')
axs[3].plot(tr2, pr, 'ro-', ms=3, alpha=0.75, label='refreshed')
axs[3].set_ylabel(r'ACF period [$1/\Omega_{e0}$]'); axs[3].legend()
axs[3].set_title('sliding-window (2500/$\\Omega_{e0}$) ACF period of the envelope')

tec, wbc = load_energy(closed)
ter, wbr = load_energy(refresh)
axs[4].semilogy(tec, wbc, 'k', label='closed WB')
axs[4].semilogy(ter, wbr, 'r', alpha=0.75, label='refreshed WB')
try:
    rc = np.genfromtxt(f'{refresh}/refresh.csv', delimiter=',', names=True)
    ax2 = axs[4].twinx()
    ax2.plot(rc['time'] * WCE, rc['redraws'] / 2000.0, 'b', lw=0.6, alpha=0.6)
    ax2.set_ylabel('redraws/step', color='b')
except OSError:
    pass
axs[4].set_ylabel('WB'); axs[4].set_xlabel(r'$t\,\Omega_{e0}$')
axs[4].legend(loc='upper left'); axs[4].set_title('field energy + refresh rate')

for ax in axs: ax.set_xlim(0, 10000)
plt.tight_layout()
plt.savefig(out, dpi=130)
print('wrote', out)
for lab, t2, p in [('closed', tc2, pc), ('refreshed', tr2, pr)]:
    if len(p):
        print(f'{lab}: ACF periods first-half median '
              f'{np.median(p[t2 < 5000]):.0f}, second-half median '
              f'{np.median(p[t2 >= 5000]):.0f} /We0  (n={len(p)})')
