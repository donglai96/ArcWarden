#!/usr/bin/env python3
"""Phase-4 paired ctrl/RSM verdict (PLAN_LUMORPH §4 + amendment A5).

FROZEN BEFORE any RSM arm data was examined (2026-08-08, runs launched
same day): the gates below implement the pre-registered criteria only.

Per pair (ctrl dir, rsm dir):
  0. hemisphere = pre-registered from CTRL low-band outward Poynting
     (direction() on ctrl; rsm judged at the SAME probe — no post-hoc pick).
  1. ctrl dominant riser (connected-component, frozen audit definitions):
     must cross 0.55 (already established by the source gate; re-printed).
  2. rsm dominant riser at the same 5° probe: ω_stop (median-endpoint,
     nwin 1024/1536 pair per A3).
  3. gate Δω_stop = end_ctrl − end_rsm >= 0.04.
  4. S_A with LOCAL-fce bands at 5° (b5 = B0(5°)/Beq): barrier band
     0.46–0.56 × b5, lower band 0.25–0.45 × b5 (A5.1; 0.5 local = 0.517
     Ωe,eq).  S_A = R_barrier / R_LB, R_X = P_rsm/P_ctrl (Welch) in band X,
     computed in the CTRL element window (primary) and full window
     (reported).  Gate S_A < 0.5.
  5. lower band survives: rsm arm has its own connected riser born in
     0.28–0.40 with sweep >= 0.05, and R_LB >= 0.3 (element not erased).
  6. recorded (not gated): rsm amp in canonical 0.40–0.80×b5 band, R_m
     from m1line in the rsm element window, W10 flood check both arms.
Ensemble: 3/3 pre-committed pairs; dual evidence lines = Δω_stop and S_A
reported separately (a pair can pass one and fail the other — report both,
overall pair PASS = gates 3+4+5 all true).

Usage: python3 lurepro_pair_verdict.py <ctrl> <rsm> [<ctrl> <rsm> ...]
"""
import glob
import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec, w10_occupancy
from lurepro_source_verdict import probe_pair, envelope, direction
from lurepro_ridge_audit import components, dominant_riser, S5

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import read_meta


def dom_riser_5deg(d, dom, wce):
    m, t5, by, bz, *_ = probe_pair(d, dom * S5)
    z5 = by + 1j * bz
    ends = []
    for nwin in (1024, 1536):
        P, fs, tt = spec(z5, t5, wce, nwin)
        dm = dominant_riser(components(P, fs, tt))
        ends.append(dm["end"] if dm else np.nan)
        if nwin == 1024:
            dm1 = dm
    return z5, t5, dm1, ends


def band_power(z, t, wce, w1, w2, w):
    zz = z[w]
    fr, Pa = welch(zz.real, fs=1 / (t[1] - t[0]), nperseg=4096)
    _, Pb = welch(zz.imag, fs=1 / (t[1] - t[0]), nperseg=4096)
    fo = fr * 2 * np.pi / wce
    sel = (fo >= w1) & (fo < w2)
    return float((Pa + Pb)[sel].mean())


def rm_element(d, t0, t1):
    """R_m = 2 rms|B1|/rms|B0w| over |s|<=175, averaged in [t0,t1] Ωe."""
    m = read_meta(d)
    nx, dx, xc = int(m["nx"]), m["dx"], m["b0_xc"]
    x = np.arange(nx) * dx - xc
    w = np.abs(x) <= 175.1
    vals = []
    for f in sorted(glob.glob(os.path.join(d, "m1line_*.bin"))):
        n = int(f[-10:-4])
        tOe = n * m["bline_every"] * m["dt"] * m["wce"]
        if not (t0 <= tOe <= t1):
            continue
        fb = os.path.join(d, f"bline_{n:06d}.bin")
        b = np.fromfile(fb, dtype=np.float32)
        by, bz = b[:nx], b[nx:2 * nx]
        c = np.fromfile(f, dtype=np.complex64)
        b1y, b1z = c[:nx], c[nx:2 * nx]
        r0 = np.sqrt(np.mean(by[w] ** 2 + bz[w] ** 2))
        r1 = np.sqrt(np.mean(np.abs(b1y[w]) ** 2 + np.abs(b1z[w]) ** 2))
        vals.append(2 * r1 / (r0 + 1e-30))
    return float(np.median(vals)) if vals else np.nan


def pair(ctrl, rsm):
    m0 = read_meta(ctrl)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    b5 = 1 + 4.5 * (S5 / lre) ** 2
    print(f"===== PAIR ctrl={ctrl} rsm={rsm}  (b5={b5:.4f}, "
          f"0.5 local = {0.5*b5:.3f} Ωe,eq)")
    dom = direction(ctrl)   # pre-registered from ctrl ONLY

    # flood check both arms (recorded)
    for d in (ctrl, rsm):
        m, t, by, bz, *_ = probe_pair(d, 0.0)
        P, fs, tt = spec(by + 1j * bz, t, wce, 1024)
        w10, _ = w10_occupancy(P, fs, tt, 0, 2000)
        print(f"[flood] {os.path.basename(d.rstrip('/')):24s} W10={w10:.3f} "
              f"({'ok' if w10 < 0.20 else 'FLOOD'})")

    zc, tc, dmc, ec = dom_riser_5deg(ctrl, dom, wce)
    zr, tr, dmr, er = dom_riser_5deg(rsm, dom, wce)
    end_c = min(ec)
    print(f"[ctrl] riser t[{dmc['t0']:.0f},{dmc['t1']:.0f}] birth "
          f"{dmc['birth']:.3f} end(1024/1536) {ec[0]:.4f}/{ec[1]:.4f}")
    if dmr is None:
        print("[rsm ] NO connected riser (sweep>=0.05) found")
        end_r, birth_r = np.nan, np.nan
    else:
        end_r, birth_r = min(er), dmr["birth"]
        print(f"[rsm ] riser t[{dmr['t0']:.0f},{dmr['t1']:.0f}] birth "
              f"{dmr['birth']:.3f} end(1024/1536) {er[0]:.4f}/{er[1]:.4f}")
    dstop = end_c - end_r if np.isfinite(end_r) else np.inf
    g_dstop = dstop >= 0.04
    print(f"[3] Δω_stop = {dstop:+.4f} (ctrl {end_c:.4f} − rsm "
          f"{end_r if np.isfinite(end_r) else float('nan'):.4f}) "
          f"gate >=0.04 -> {'PASS' if g_dstop else 'FAIL'}"
          f"   dual norm: rsm stop /Ωe(5°) = {end_r/b5:.4f}")

    # S_A in ctrl element window (primary) + full (reported), LOCAL bands
    bb = (0.46 * b5, 0.56 * b5)
    lb = (0.25 * b5, 0.45 * b5)
    n = min(len(zc), len(zr))
    for name, w in (("ctrl-element",
                     (tc[:n] * wce > dmc["t0"]) & (tc[:n] * wce < dmc["t1"] + 100)),
                    ("full", np.ones(n, dtype=bool))):
        Rb = (band_power(zr[:n], tr, wce, *bb, w)
              / band_power(zc[:n], tc, wce, *bb, w))
        Rl = (band_power(zr[:n], tr, wce, *lb, w)
              / band_power(zc[:n], tc, wce, *lb, w))
        sa = Rb / Rl
        if name == "ctrl-element":
            g_sa, sa_prim, Rl_prim = sa < 0.5, sa, Rl
        print(f"[4] S_A[{name}] = {sa:.3f} (R_barrier {Rb:.3f} / R_LB "
              f"{Rl:.3f})" + ("  gate <0.5 -> "
              + ("PASS" if sa < 0.5 else "FAIL") if name == "ctrl-element"
              else "  (reported)"))

    g_lb = (dmr is not None and 0.28 <= birth_r <= 0.40
            and dmr["sweep"] >= 0.05 and Rl_prim >= 0.3)
    print(f"[5] lower band survives: rsm riser born "
          f"{birth_r if np.isfinite(birth_r) else float('nan'):.3f} "
          f"(0.28-0.40), sweep {dmr['sweep'] if dmr else 0:.3f}, "
          f"R_LB {Rl_prim:.3f} (>=0.3) -> {'PASS' if g_lb else 'FAIL'}")

    # recorded: rsm canonical amp + R_m in rsm element window
    if dmr is not None:
        w = (tr * wce > dmr["t0"]) & (tr * wce < dmr["t1"] + 100)
        amp = float(envelope(zr[w], tr[w], wce, 0.40 * b5, 0.80 * b5).max()) / wce
        rmv = rm_element(rsm, dmr["t0"], dmr["t1"])
        print(f"[rec] rsm canonical amp {amp:.2e} B_eq | {amp/b5:.2e} B0(5°); "
              f"R_m(element) = {rmv:.3f}")

    ok = g_dstop and g_sa and g_lb
    print(f"[pair] Δω_stop {g_dstop} | S_A {g_sa} | LB-survives {g_lb} "
          f" ==> {'PAIR PASS' if ok else 'PAIR FAIL'}\n")
    return ok, dstop, sa_prim


if __name__ == "__main__":
    a = sys.argv[1:]
    res = [pair(a[i], a[i + 1]) for i in range(0, len(a), 2)]
    np_ = sum(r[0] for r in res)
    print(f"===== ensemble: {np_}/{len(res)} pairs PASS | "
          f"Δω_stop: {['%+.3f' % r[1] for r in res]} | "
          f"S_A: {['%.3f' % r[2] for r in res]} =====")
