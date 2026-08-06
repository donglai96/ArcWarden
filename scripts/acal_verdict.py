#!/usr/bin/env python3
"""A-cal verdict (PLAN_TWO_TRACK v2.1 execution step 4): D120 antenna
amplitude calibration.

Measurement discipline (user review round 3, hard gate):
  - NO near-field numbers: the equator probe sits inside the antenna
    column (sigma = 4 cells ~ 1 c/wpe) — reported only as a flagged
    reference, never quoted as dB_trig.
  - Quote the POST-OFF packet at the near-source probes x = +-145.2
    c/wpe: complex demodulation of z(t) = By + i Bz against exp(+i w0 t)
    (R-mode whistler rotates as exp(-i w0 t)), boxcar low-pass over 2
    drive periods, envelope median over the passage window
    [toff + x/vg - 300, toff + x/vg + 300].
  - Linearity: dB(a10)/dB(a05) vs amp ratio 2.0 (quote deviation; the
    pre-burst hot noise floor is measured in the same band from the
    pre-arrival window and quoted alongside).

Usage: python3 acal_verdict.py <dir_a05> <dir_a10>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import read_meta


def probe_series(d, offset):
    m = read_meta(d)
    npr = int(m["nprobe"])
    raw = np.fromfile(f"{d}/probe.bin", dtype=np.float32)
    raw = raw[: (len(raw) // (2 * npr)) * 2 * npr].reshape(-1, npr, 2)
    ix = np.array(m["probe_ix"], dtype=float)
    off = ix * m["dx"] - m["b0_xc"]
    p = int(np.argmin(np.abs(off - offset)))
    t = (np.arange(raw.shape[0]) + 1) * m["probe_every"] * m["dt"]
    return m, t, raw[:, p, 0].astype(float) + 1j * raw[:, p, 1].astype(float)


def envelope(t, z, w0):
    # z = By + i Bz rotates as exp(+i w0 t) (measured: the +w0 FFT line
    # carries the drive, -w0 is 10x down) -> demodulate with exp(-i w0 t),
    # then boxcar over 2 drive periods kills the counter-rotating residue
    # (a ~+-0.025 wpe band-pass around w0 against the hot broadband floor)
    base = z * np.exp(-1j * w0 * t)
    nbox = max(1, int(round(2 * 2 * np.pi / w0 / (t[1] - t[0]))))
    k = np.ones(nbox) / nbox
    return np.abs(np.convolve(base, k, mode="same"))


def arm(d):
    m = read_meta(d)
    w0, toff, amp = m["ant_w0"], m["ant_toff"], m["ant_amp"]
    trmp = m.get("ant_trmp", 250.0)   # not in meta.txt; deck value
    b0 = m["wce"]
    # vg at w0 for the cold whistler (wpe = 1 units)
    wt = w0 / m["wce"]
    x2 = wt / (1.0 - wt)
    vg = m["wce"] * 2.0 * np.sqrt(x2) / (1.0 + x2) ** 2
    out = {}
    for offset in (-145.2, 145.2):
        _, t, z = probe_series(d, offset)
        env = envelope(t, z, w0)
        # steady-train passage: the segment emitted in [trmp, toff] passes
        # x at [x/vg + trmp, x/vg + toff] — entirely AFTER toff here
        # (x/vg ~ 1118 > toff - trmp), so the antenna is already silent
        # (post-off discipline holds; bline check: packet at +-560 cells
        # peaks t~1500-2000 at 3.7e-4, matches this window)
        t_travel = abs(offset) / vg
        w1, w2 = t_travel + trmp + 100, t_travel + toff
        assert w1 > toff, "passage window overlaps the driven phase"
        win = (t > w1) & (t < w2)
        pre = (t > 200) & (t < t_travel - 200)  # before first arrival
        out[offset] = (np.median(env[win]) / b0,
                       np.median(env[pre]) / b0 if pre.any() else np.nan)
    # flagged near-field reference only
    _, t, z = probe_series(d, 0.0)
    env = envelope(t, z, w0)
    win = (t > toff - 600) & (t < toff)
    out["eq_nearfield_REF_ONLY"] = np.median(env[win]) / b0
    return amp, vg, out


def main(d05, d10):
    a05, vg, r05 = arm(d05)
    a10, _, r10 = arm(d10)
    print(f"vg(w0)/c = {vg:.4f}")
    print(f"{'probe':>26s} {'dB/B0 a05':>12s} {'noise':>10s} "
          f"{'dB/B0 a10':>12s} {'noise':>10s} {'ratio':>7s}")
    ratios = []
    for k in (-145.2, 145.2):
        v5, n5 = r05[k]
        v10, n10 = r10[k]
        ratios.append(v10 / v5)
        print(f"{k:>26} {v5:12.3e} {n5:10.2e} {v10:12.3e} {n10:10.2e} "
              f"{v10 / v5:7.3f}")
    print(f"{'eq (near-field, REF ONLY)':>26s} "
          f"{r05['eq_nearfield_REF_ONLY']:12.3e} {'':10s} "
          f"{r10['eq_nearfield_REF_ONLY']:12.3e}")
    amp_ratio = a10 / a05
    # per-probe ratios carry the coherent-interference systematic of the
    # SAME-SEED hot background with the packet (~2N/A ~ 10-15% here); the
    # geometric mean over the +-x pair cancels it to first order, so the
    # linearity bar applies to the mean and the per-probe spread is quoted
    # as the systematic.
    gm = float(np.sqrt(ratios[0] * ratios[1]))
    spread = max(abs(r / gm - 1.0) for r in ratios)
    dev = abs(gm / amp_ratio - 1.0)
    g05 = float(np.sqrt(r05[-145.2][0] * r05[145.2][0]))
    g10 = float(np.sqrt(r10[-145.2][0] * r10[145.2][0]))
    print(f"\nprobe-mean dB_trig/B0: a05 = {g05:.3e}   a10 = {g10:.3e}")
    print(f"amp ratio = {amp_ratio:.3f}; probe-mean envelope ratio = {gm:.3f} "
          f"(per-probe {ratios[0]:.3f}/{ratios[1]:.3f}, interference "
          f"systematic +-{100 * spread:.0f}%)")
    print(f"linearity dev of the mean = {100 * dev:.1f}%  ->  "
          f"{'PASS' if dev < 0.10 else 'FAIL'} (10% bar)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
