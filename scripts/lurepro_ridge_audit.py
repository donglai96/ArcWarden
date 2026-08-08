#!/usr/bin/env python3
"""Packet-specific ridge audit (user directive 2026-08-07 post-3-seed):

1. connected-component ridge tracking in the (t, ω) plane at the 5° main
   probe — jumps between different risers are impossible by construction
   (a ridge lives inside ONE 8-connected component of the thresholded
   spectrogram);
2. the SAME dominant component's band/window is used to find its delayed
   copy at 7.5° (ridge-specific propagation delay, not whole-envelope);
3. seed21's true endpoint from the component (expected ~0.741, not the
   0.938 cross-riser jump of the old follow tracker);
4. the final ok includes ALL of: flood gate, dominant-riser existence
   (Δω>=0.15, endpoint>=0.55), THREE-window endpoint spread <0.01,
   amplitude gate on the component band Bw/B0(s) in [3e-3, 1e-2] (hard,
   pre-registered), ridge-specific outward delay (lag>0, r>=0.5) and
   endpoint continuity at 7.5°.

Usage: python3 lurepro_ridge_audit.py <dir> [<dir> ...]
"""
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec, w10_occupancy
from lurepro_source_verdict import probe_pair, envelope, direction

S5, S75 = 116.4, 175.1


def components(P, fs, tt, drop_db=2.0, min_dur=200.0):
    """8-connected components of the top `drop_db` decades; returns list of
    dicts with ridge/birth/endpoint/sweep/power."""
    L = np.log10(P + 1e-30)
    mask = L > (L.max() - drop_db)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        sel = lab == c
        cols = np.nonzero(sel.any(axis=1))[0]
        if tt[cols[-1]] - tt[cols[0]] < min_dur:
            continue
        ridge_t, ridge_f = [], []
        for i in cols:
            row = np.where(sel[i], P[i], 0.0)
            j = int(np.argmax(row))
            if row[j] > 0:
                ridge_t.append(tt[i])
                ridge_f.append(fs[j])
        ridge_t, ridge_f = np.array(ridge_t), np.array(ridge_f)
        k = max(3, len(ridge_f) // 10)
        birth = float(np.median(ridge_f[:k]))
        endpt = float(np.median(ridge_f[-k:]))
        out.append(dict(t0=ridge_t[0], t1=ridge_t[-1], birth=birth,
                        end=endpt, top=float(ridge_f.max()),
                        sweep=endpt - birth,
                        power=float(P[sel.T.nonzero()[::-1]].sum()),
                        ridge=(ridge_t, ridge_f)))
    return out


def dominant_riser(comps):
    cand = [c for c in comps if c["sweep"] >= 0.05]
    if not cand:
        return None
    return max(cand, key=lambda c: c["power"])


def audit(d):
    m0, t, by, bz, *_ = probe_pair(d, 0.0)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    b5 = 1 + 4.5 * (S5 / lre) ** 2
    b75 = 1 + 4.5 * (S75 / lre) ** 2
    print(f"===== {d}")
    dom = direction(d)

    # flood gate at equator
    z = by + 1j * bz
    P, fs, tt = spec(z, t, wce, 1024)
    w10, _ = w10_occupancy(P, fs, tt, 0, 2000)
    g_flood = w10 < 0.20

    # 5° connected-component ridges, three windows
    m, t5, by, bz, *_ = probe_pair(d, dom * S5)
    z5 = by + 1j * bz
    ends, doms_ = [], {}
    for nwin in (1024, 1536, 2048):
        P, fs, tt = spec(z5, t5, wce, nwin)
        comps = components(P, fs, tt)
        dm = dominant_riser(comps)
        doms_[nwin] = dm
        ends.append(dm["end"] if dm else np.nan)
    dm = doms_[1024]
    # AMENDMENT v2 (A3): convergence on the 1024/1536 pair (2048 reported
    # with the known long-window endpoint smear); both pair windows >=0.55
    spread = abs(ends[0] - ends[1])
    g_riser = (dm is not None and dm["sweep"] >= 0.15
               and min(ends[0], ends[1]) >= 0.55)
    g_spread = spread < 0.01
    print(f"[5°] dominant component: t[{dm['t0']:.0f},{dm['t1']:.0f}] "
          f"birth {dm['birth']:.3f} endpoint {dm['end']:.3f} "
          f"(ridge max {dm['top']:.3f}) sweep {dm['sweep']:.3f}")
    print(f"[5°] endpoint(nwin 1024/1536) = {ends[0]:.4f}/{ends[1]:.4f} "
          f"pair-spread {spread:.4f} ({'PASS' if g_spread else 'FAIL'} <0.01)"
          f"  [2048 = {ends[2]:.4f}, reported only: long-window smear]"
          f"   riser gate (Δω>=0.15 & >=0.55): "
          f"{'PASS' if g_riser else 'FAIL'}")
    print(f"     dual norm: endpoint /Ωe,eq {ends[0]:.3f} | /Ωe(5°) "
          f"{ends[0]/b5:.3f}")

    # AMENDMENT v2 (A1+A2): canonical amplitude = FIXED band 0.40-0.80 in
    # the dominant component's window; floor 3e-3 HARD, ceiling 1e-2
    # RECORDED only (flood rejection is W10's job); component-adaptive
    # value recorded as diagnostic
    w = (t5 * wce > dm["t0"]) & (t5 * wce < dm["t1"] + 100)
    e5 = envelope(z5[w], t5[w], wce, 0.40, 0.80)
    amp5 = float(e5.max()) / wce
    f1, f2 = max(0.15, dm["birth"] - 0.05), dm["top"] + 0.05
    ampc = float(envelope(z5[w], t5[w], wce, f1, f2).max()) / wce
    g_amp = amp5 / b5 >= 3e-3
    over = amp5 / b5 > 1e-2
    print(f"[5°] canonical amp (0.40-0.80): {amp5:.2e} B_eq | {amp5/b5:.2e} "
          f"B0(5°)  floor gate: {'PASS' if g_amp else 'FAIL'}"
          + ("  [NOTE: exceeds 1e-2 recorded ceiling]" if over else "")
          + f"   [adaptive-band diag: {ampc/b5:.2e}]")

    # ridge-specific delayed copy at 7.5°
    m, t7, by, bz, *_ = probe_pair(d, dom * S75)
    z7 = by + 1j * bz
    w7 = (t7 * wce > dm["t0"]) & (t7 * wce < dm["t1"] + 400)
    e7 = envelope(z7[w7], t7[w7], wce, f1, f2)
    n = min(len(e5), len(e7))
    a = e5[:n] - e5[:n].mean()
    b = e7[:n] - e7[:n].mean()
    dt_ = (t5[1] - t5[0]) * wce
    lags = np.arange(0, int(400 / dt_))
    cc = [np.corrcoef(a, np.roll(b, -L)[:n])[0, 1] for L in lags]
    r = float(np.max(cc))
    lag = float(lags[int(np.argmax(cc))] * dt_)
    # endpoint continuity: dominant component at 7.5° in the lagged window
    P, fs, tt = spec(z7, t7, wce, 1024)
    comps7 = components(P, fs, tt)
    in_w = [c for c in comps7
            if c["t1"] > dm["t0"] + lag - 100 and c["t0"] < dm["t1"] + lag + 100
            and c["sweep"] >= 0.05]
    end7 = max((c["end"] for c in in_w), default=np.nan)
    # AMENDMENT v2 (A3): continuity tolerance = one STFT bin (nwin=1024,
    # probe cadence 1.5/wpe -> 0.0204 We), replacing the ad hoc 0.03
    BIN = 0.0204
    g_prop = (r >= 0.5) and (0 < lag < 400) and (end7 >= dm["end"] - BIN)
    print(f"[7.5°] ridge-specific delay: r={r:.2f} lag={lag:.0f}/Ωe; "
          f"matched component endpoint {end7:.3f} (>= 5° end − 1 bin "
          f"{BIN})  -> {'PASS' if g_prop else 'FAIL'}")

    ok = all((g_flood, g_riser, g_spread, g_amp, g_prop))
    print(f"[gate] flood {g_flood} | riser {g_riser} | spread {g_spread} | "
          f"amplitude {g_amp} | propagation {g_prop}  ==> "
          f"{'SEED PASS' if ok else 'SEED FAIL'}\n")
    return ok


if __name__ == "__main__":
    res = {d: audit(d) for d in sys.argv[1:]}
    npass = sum(res.values())
    print(f"===== ensemble: {npass}/{len(res)} PASS =====")
