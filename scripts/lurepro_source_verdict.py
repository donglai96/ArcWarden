#!/usr/bin/env python3
"""Lu-repro source verdict — PLAN_LUMORPH.md 2026-08-07 revision.

Dual-location criteria:
  equator  : birth time/freq, flood check (W10), amplitude RECORDED only
  |λ|=5°   (|s|=116.4) : MAIN judge — sweep Δω, ω_max, 3-nwin 0.55, Bw/B0(s)
  |λ|=7.5° (|s|=175.1) : confirmation — arrival ordering, amplification,
                          single-convective-packet continuity
Direction pre-registration: outward Poynting flux of the LOW-band packets
(0.28–0.48) during the growth phase decides the hemisphere BEFORE any
high-band numbers are looked at; paired RSM must reuse it.
Dual normalization: every frequency quoted as ω/Ωe,eq AND ω/Ωe(s); every
amplitude as Bw/Beq AND Bw/B0(s).  B0(s)/Beq = 1 + 4.5 (s/lre)^2 (code
dipole profile; 5° = 1.0347, 7.5° = 1.0794 at lre=1330.5).

Success conditions (per seed, all six):
  1 no flood by t2000 (W10 < 0.20 at equator)
  2 a space-time continuous outward element exists
  3 at 5°: Δω >= 0.15 Ωe,eq and ω_max >= 0.55 Ωe,eq
  4 three STFT windows all confirm ω_max >= 0.55
  5 5°→7.5° arrival + amplification ordering consistent with outward
  6 the high-frequency structure is ONE convective packet (5°/7.5°
    envelope cross-correlation peak at positive lag ~ Δs/vg, r >= 0.5)
Secondary metric: whether ω_max reaches 0.60 / 0.65 (reported, not gated).

Usage: python3 lurepro_source_verdict.py <dir>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import read_meta
from lumorph_gate import spec, w10_occupancy, follow, element_from_traj

S5, S75 = 116.4, 175.1


def probe_pair(d, offset):
    m = read_meta(d)
    npr = int(m["nprobe"])
    ix = np.array(m["probe_ix"], dtype=float)
    off = ix * m["dx"] - m["b0_xc"]
    p = int(np.argmin(np.abs(off - offset)))
    rb = np.fromfile(f"{d}/probe.bin", dtype=np.float32)
    rb = rb[: (len(rb) // (2 * npr)) * 2 * npr].reshape(-1, npr, 2)
    re = np.fromfile(f"{d}/probe_e.bin", dtype=np.float32)
    re = re[: (len(re) // (2 * npr)) * 2 * npr].reshape(-1, npr, 2)
    nt = min(len(rb), len(re))
    t = (np.arange(nt) + 1) * m["probe_every"] * m["dt"]
    return (m, t, rb[:nt, p, 0].astype(float), rb[:nt, p, 1].astype(float),
            re[:nt, p, 0].astype(float), re[:nt, p, 1].astype(float))


def bandpass(sig, t, wce, w1, w2):
    n = len(sig)
    F = np.fft.fft(sig)
    f = np.fft.fftfreq(n, d=t[1] - t[0]) * 2 * np.pi / wce
    F[(np.abs(f) < w1) | (np.abs(f) > w2)] = 0
    return np.real(np.fft.ifft(F)) if np.isrealobj(sig) else np.fft.ifft(F)


def envelope(z, t, wce, w1, w2):
    n = len(z)
    F = np.fft.fft(z)
    f = np.fft.fftfreq(n, d=t[1] - t[0]) * 2 * np.pi / wce
    F[(f < w1) | (f > w2)] = 0
    env = np.abs(np.fft.ifft(F))
    nbox = max(1, int(60 / wce / (t[1] - t[0])))
    return np.convolve(env, np.ones(nbox) / nbox, mode="same")


def direction(d):
    """pre-registered hemisphere from LOW-band outward Poynting at ±5°."""
    out = {}
    for sgn in (+1, -1):
        m, t, by, bz, ey, ez = probe_pair(d, sgn * S5)
        wce = m["wce"]
        fb = lambda s: bandpass(s, t, wce, 0.28, 0.48)
        by, bz, ey, ez = fb(by), fb(bz), fb(ey), fb(ez)
        sx = ey * bz - ez * by
        # growth window: from when low-band |B| env passes 10% of its max
        env = np.hypot(by, bz)
        i0 = np.argmax(env > 0.1 * env.max())
        w = slice(i0, min(len(t), i0 + int(1000 / wce / (t[1] - t[0]))))
        out[sgn] = sgn * np.mean(sx[w])       # >0 = outward
    dom = +1 if out[+1] >= out[-1] else -1
    print(f"[dir] outward low-band Poynting: north {out[+1]:+.3e}  "
          f"south {out[-1]:+.3e}  -> MAIN hemisphere = "
          f"{'NORTH(+)' if dom > 0 else 'SOUTH(-)'} (pre-registered)")
    return dom


def main(d):
    m0 = read_meta(d)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    b5 = 1 + 4.5 * (S5 / lre) ** 2
    b75 = 1 + 4.5 * (S75 / lre) ** 2

    dom = direction(d)

    # ---- equator ----
    m, t, by, bz, *_ = probe_pair(d, 0.0)
    z = by + 1j * bz
    P, fs, tt = spec(z, t, wce, 1024)
    w10, nstrong = w10_occupancy(P, fs, tt, 0, 2000)
    c1 = w10 < 0.20
    tr = follow(P, fs, tt, 0.30, 200)
    lo, hi = element_from_traj(tr)
    eq_birth_t, eq_birth_f = tr[lo, 0], tr[lo, 1]
    eq_amp = envelope(z, t, wce, 0.15, 0.80).max() / wce
    print(f"[eq] W10={w10:.3f} ({'PASS' if c1 else 'FLOOD FAIL'})  "
          f"birth t={eq_birth_t:.0f} f={eq_birth_f:.3f}  "
          f"amp(recorded)={eq_amp:.2e} B_eq")

    # ---- 5° main ----
    m, t, by, bz, *_ = probe_pair(d, dom * S5)
    z5 = by + 1j * bz
    tops, els = [], {}
    for nwin in (1024, 1536, 2048):
        P, fs, tt = spec(z5, t, wce, nwin)
        tr = follow(P, fs, tt, 0.30, 200)
        lo, hi = element_from_traj(tr)
        el = tr[lo:hi + 1]
        tops.append(el[:, 1].max())
        els[nwin] = el
    el = els[1024]
    birth5, top5 = el[0, 1], max(tops)
    dw = el[:, 1].max() - el[0, 1]
    c2 = (el[-1, 0] - el[0, 0]) >= 300
    c3 = (dw >= 0.15) and (min(tops) >= 0.55)
    c4 = min(tops) >= 0.55
    e5 = envelope(z5, t, wce, 0.40, 0.80)
    amp5 = e5.max() / wce
    c6amp = 3e-3 <= amp5 / b5 <= 1e-2
    print(f"[5°]  element t[{el[0,0]:.0f},{el[-1,0]:.0f}] birth {birth5:.3f} "
          f"Δω={dw:.3f} (bar>=0.15)")
    print(f"[5°]  ω_max(nwin 1024/1536/2048) = "
          + "/".join(f"{x:.4f}" for x in tops)
          + f"  ->  /Ωe,eq {top5:.3f} | /Ωe(5°) {top5/b5:.3f}")
    print(f"[5°]  amp = {amp5:.2e} B_eq | {amp5/b5:.2e} B0(5°)  "
          f"(criterion-6 window on B0(s): {'PASS' if c6amp else 'FAIL'})")
    print(f"[5°]  cond3 (Δω & ω_max>=0.55): {'PASS' if c3 else 'FAIL'}   "
          f"cond4 (3-nwin all >=0.55): {'PASS' if c4 else 'FAIL'}")
    print(f"      secondary: reaches 0.60: {'Y' if top5>=0.60 else 'N'}  "
          f"0.65: {'Y' if top5>=0.65 else 'N'}")

    # ---- 7.5° confirmation ----
    m, t7, by, bz, *_ = probe_pair(d, dom * S75)
    z7 = by + 1j * bz
    e7 = envelope(z7, t7, wce, 0.40, 0.80)
    t5o = t[np.argmax(e5 > 0.3 * e5.max())] * wce
    t7o = t7[np.argmax(e7 > 0.3 * e7.max())] * wce
    amp7 = e7.max() / wce
    c5 = (t7o >= t5o - 30) and (amp7 >= amp5)
    print(f"[7.5°] onset {t7o:.0f} vs 5° {t5o:.0f} (lag {t7o-t5o:+.0f})  "
          f"amp {amp7:.2e} B_eq | {amp7/b75:.2e} B0(7.5°)  "
          f"outward ordering+amplification: {'PASS' if c5 else 'FAIL'}")
    # packet continuity: cross-correlate hi-band envelopes, window around
    # the 5° dominant packet
    n = min(len(e5), len(e7))
    a, b = e5[:n] - e5[:n].mean(), e7[:n] - e7[:n].mean()
    ipk = int(np.argmax(e5[:n]))
    hw = int(300 / wce / (t[1] - t[0]))
    sl = slice(max(0, ipk - hw), min(n, ipk + hw))
    lags = np.arange(0, int(400 / wce / (t[1] - t[0])))
    cc = [np.corrcoef(a[sl], np.roll(b, -L)[sl])[0, 1] for L in lags]
    Lbest = int(lags[int(np.argmax(cc))])
    lag_Oe = Lbest * (t[1] - t[0]) * wce
    r = max(cc)
    c6 = (r >= 0.5) and (0 < lag_Oe < 400)
    print(f"[7.5°] packet continuity: max corr r={r:.2f} at lag "
          f"{lag_Oe:.0f}/Ωe (expect ~Δs/vg~120)  -> "
          f"{'PASS' if c6 else 'FAIL'}")

    ok = all((c1, c2, c3, c4, c5, c6))
    print(f"\n=== {d}: {'SEED PASS' if ok else 'SEED FAIL'} "
          f"(1flood {c1} 2elem {c2} 3sweep {c3} 4nwin {c4} 5order {c5} "
          f"6packet {c6}) ===")


if __name__ == "__main__":
    main(sys.argv[1])
