#!/usr/bin/env python3
"""Element-period + sweep-rate + gating analysis for the Chen 2026 lre scan.

Three measurements, all on the equatorial probe unless noted:

1. ACF period: autocorrelation of the smoothed envelope over a window --
   objective repetition period, no peak-picking thresholds. x10 gives
   894/Oe; x5 gives 2795 (eq) / 2446 (S10, with clean harmonics).

2. Sweep rate vs Omura 2021 Eq. (88): the spectral ridge is split into
   monotonic rising segments (cut at sharp down-jumps); each segment gets a
   linear fit dw/dt and the top-20% envelope amplitude of its window, and is
   compared against 0.4*s0*w*Bw/s1 (s0 = chi*Uperp0/xi, s1 = gam*(1-VR/Vg)^2).
   Median meas/theory = 0.72 (x10) and 0.73 (x5): identical coefficient,
   lre-independent as Eq. 88 predicts; effective S ~ -0.29.

3. Gating evidence from inter-element gaps: minimum envelope vs B_th,
   fraction of gap spent super-threshold, and the effective growth rate of
   the recovery climb. x10 rides the threshold (min ~ 0.8x th) ->
   threshold-gated; x5 waits 8x ABOVE threshold with gamma suppressed 10x
   (6.5e-4 vs 5.8e-3) -> refill-gated (bounce return flow is the gate).

Usage: analyze_chen2026_periods.py  (paths and B_th values are hardwired to
the giant-run directories; edit RUNS below for new cases)
"""
import sys
import numpy as np

sys.path.insert(0, sys.path[0])
from plot_chen2026_fig1 import load_probe, stft
from omura_threshold import branch, Uperp0
from scipy.signal import find_peaks, medfilt

RUNS = {
    # name: (dir, B_th at w=0.25 from omura_threshold.py, acf window, gap peak params)
    "x10": ("build/chen2026_case2_giant", 1.24e-3, (1000, 10000),
            dict(pk_h=2.5e-3, pk_prom=1.5e-3, min_dist=700, t0=800)),
    "x5": ("build/chen2026_case2_l5", 7.8e-5, (3500, 15000),
           dict(pk_h=2.5e-3, pk_prom=1.5e-3, min_dist=1200, t0=3200)),
}


def envelope(d, poff=0.0, k=512):
    m, t, by, bz = load_probe(d, poff)
    wce = m["wce"]
    e = np.convolve(np.abs(by + 1j * bz) / wce, np.ones(k) / k, mode="same")
    return t * wce, e, m


def acf_period(toe, e, t0, t1):
    sel = (toe >= t0) & (toe <= t1)
    x = e[sel] - e[sel].mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    dt_oe = toe[1] - toe[0]
    i0 = int(300 / dt_oe)
    pk, _ = find_peaks(ac[i0:], height=0.05, prominence=0.03)
    return [((i0 + p) * dt_oe, ac[i0 + p]) for p in pk[:4]]


def ridge_segments(d, poff=0.0, nwin=1024, floor=1.5e-3,
                   min_dur=200, min_span=0.06):
    m, t, by, bz = load_probe(d, poff)
    wce = m["wce"]
    sig = (by + 1j * bz) / wce
    hop = nwin // 16
    spec = stft(sig, nwin, hop)
    dt_s = t[1] - t[0]
    freqs = np.fft.fftfreq(nwin, d=dt_s) * 2 * np.pi / wce
    tt = (t[0] + (np.arange(spec.shape[0]) * hop + nwin / 2) * dt_s) * wce
    sel = (freqs >= 0.15) & (freqs <= 0.68)
    P = np.abs(spec[:, sel])
    fr = freqs[sel]
    j = np.argmax(P, axis=1)
    ok = P[np.arange(len(j)), j] > floor
    rt, rf = tt[ok], medfilt(fr[j[ok]], 5)
    toe, env = t * wce, np.abs(sig)

    cuts = np.where((np.diff(rf) < -0.04) | (np.diff(rt) > 400))[0]
    segs, start = [], 0
    for c in list(cuts) + [len(rf) - 1]:
        s = slice(start, c + 1)
        start = c + 1
        t_, f_ = rt[s], rf[s]
        if len(t_) < 6 or t_[-1] - t_[0] < min_dur:
            continue
        A = np.polyfit(t_, f_, 1)
        r2 = 1 - (f_ - np.polyval(A, t_)).var() / f_.var() if f_.var() > 0 else 0
        if A[0] > 0 and f_.max() - f_.min() >= min_span and r2 > 0.5:
            esel = (toe >= t_[0]) & (toe <= t_[-1])
            bw = np.mean(np.sort(env[esel])[-max(1, int(0.2 * esel.sum())):])
            segs.append(dict(t0=t_[0], t1=t_[-1], slope=A[0],
                             w0=f_.mean(), r2=r2, bw=bw))
    return segs


def sweep_theory(wt, bw):
    k, Vg, xi, chi, Vp, VR, gam, s2 = branch(wt)
    s0 = chi * Uperp0 / xi
    s1 = gam * (1 - VR / Vg) ** 2
    return 0.4 * s0 * wt * bw / s1


def gap_analysis(toe, e, th, pk_h, pk_prom, min_dist, t0):
    dt_oe = toe[1] - toe[0]
    pk, _ = find_peaks(e, height=pk_h, distance=int(min_dist / dt_oe),
                       prominence=pk_prom)
    pk = pk[toe[pk] > t0]
    out = []
    for a, b in zip(pk[:-1], pk[1:]):
        seg = e[a:b]
        i_min = a + int(np.argmin(seg))
        climb, tt = e[i_min:b], toe[i_min:b]
        sel = climb > 1.5 * e[i_min]
        gam = (np.polyfit(tt[sel], np.log(climb[sel]), 1)[0]
               if sel.sum() > 10 else np.nan)
        out.append(dict(ta=toe[a], tb=toe[b], emin=e[i_min],
                        ratio=e[i_min] / th, frac=(seg > th).mean(), gam=gam))
    return out


def main():
    for name, (d, th, (a0, a1), gp) in RUNS.items():
        toe, e, _ = envelope(d)
        print(f"\n=== {name} ({d}) ===")
        acf = acf_period(toe, e, a0, a1)
        print("ACF peaks (lag, value):",
              ", ".join(f"{l:.0f} ({v:.2f})" for l, v in acf))
        segs = ridge_segments(d)
        ratios = [s["slope"] / sweep_theory(s["w0"], s["bw"]) for s in segs]
        print(f"sweep-rate segments: {len(segs)}, "
              f"median meas/Eq88 = {np.median(ratios):.2f}, "
              f"range {min(ratios):.2f}-{max(ratios):.2f}")
        for g in gap_analysis(toe, e, th, **gp):
            print(f"  gap {g['ta']:6.0f}->{g['tb']:6.0f}: "
                  f"min={g['emin']:.1e} ({g['ratio']:.1f}x th), "
                  f"above-th {g['frac'] * 100:3.0f}% of gap, "
                  f"gamma_climb={g['gam']:.1e}")


if __name__ == "__main__":
    main()
