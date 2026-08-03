#!/usr/bin/env python3
"""Per-element chirp termination frequency (omega_stop) statistics.

Gap-as-statistics test (x4×RSM program, docs/X4_RSM_EPAR_PLAN.md §6): if the
m1-carved Landau plateau gates the chirp at 0.5 fce, the ensemble of element
termination frequencies omega_stop should pile up at/below ~0.45 in RSM runs
while control elements cross 0.5. One number per element, comparable across
runs and seeds — robust to element-to-element variance in a way single
spectrograms are not.

Usage: x4_omega_stop.py DIR [DIR ...]   (equator probe of each run)
"""
import sys
import numpy as np
from scipy.signal import medfilt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import load_probe, stft


def elements(d, nwin=1024, floor=1.2e-3, fband=(0.13, 0.72),
             tgap=400.0, fdrop=0.05, min_dur=250.0, min_span=0.05):
    """Ridge-tracked elements at the equator probe.

    Returns list of dicts with t0/t1 (We0 units), w0 (start freq), wmax,
    wstop (median ridge freq over the last 5 frames — the termination), bw
    (top-20% envelope Bw/B0 over the element)."""
    m, t, by, bz = load_probe(d, 0.0)
    wce = m["wce"]
    sig = (by + 1j * bz) / wce
    hop = nwin // 16
    spec = stft(sig, nwin, hop)
    dt_s = t[1] - t[0]
    freqs = np.fft.fftfreq(nwin, d=dt_s) * 2 * np.pi / wce
    tt = (t[0] + (np.arange(spec.shape[0]) * hop + nwin / 2) * dt_s) * wce
    sel = (freqs >= fband[0]) & (freqs <= fband[1])
    P = np.abs(spec[:, sel])
    fr = freqs[sel]
    j = np.argmax(P, axis=1)
    ok = P[np.arange(len(j)), j] > floor
    if ok.sum() < 10:
        return []
    rt, rf = tt[ok], medfilt(fr[j[ok]], 5)
    toe, env = t * wce, np.abs(sig)

    cuts = np.where((np.diff(rf) < -fdrop) | (np.diff(rt) > tgap))[0]
    out, start = [], 0
    for c in list(cuts) + [len(rf) - 1]:
        s = slice(start, c + 1)
        start = c + 1
        t_, f_ = rt[s], rf[s]
        if len(t_) < 6 or t_[-1] - t_[0] < min_dur:
            continue
        if f_.max() - f_.min() < min_span:
            continue
        A = np.polyfit(t_, f_, 1)
        if A[0] <= 0:            # risers only
            continue
        esel = (toe >= t_[0]) & (toe <= t_[-1])
        bw = np.mean(np.sort(env[esel])[-max(1, int(0.2 * esel.sum())):])
        out.append(dict(t0=t_[0], t1=t_[-1], w0=float(f_[:5].mean()),
                        wmax=float(f_.max()), wstop=float(np.median(f_[-5:])),
                        bw=float(bw), sweep=float(A[0])))
    return out


def main():
    print(f"{'run':28s} {'t0':>6s} {'t1':>6s} {'w0':>6s} {'wmax':>6s} "
          f"{'wstop':>6s} {'Bw/B0':>8s}")
    for d in sys.argv[1:]:
        els = elements(d)
        if not els:
            print(f"{d:28s}  -- no elements above floor --")
            continue
        for e in els:
            print(f"{d:28s} {e['t0']:6.0f} {e['t1']:6.0f} {e['w0']:6.3f} "
                  f"{e['wmax']:6.3f} {e['wstop']:6.3f} {e['bw']:8.2e}")
        wm = [e["wmax"] for e in els]
        print(f"{d:28s}  N={len(els)}  max(wmax)={max(wm):.3f}  "
              f"mean(wstop)={np.mean([e['wstop'] for e in els]):.3f}")


if __name__ == "__main__":
    main()
