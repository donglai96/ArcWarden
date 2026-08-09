#!/usr/bin/env python3
"""LRE bifurcation scan — endpoint state map (PLAN_LRE_SCAN §5/§7).

Per arm, pre-registered classification:
  FLOOD  : eq W10 >= 0.20 (third source state — NO endpoint is forced out
           of a flood; recorded with W10 only)
  ELEMENT: dominant connected riser at the MLAT-5° main probe (frozen
           ridge-audit machinery, mlat_probes scaling), endpoint quoted as
           the 1024/1536 pair with spread, canonical amp recorded.
Money plot: omega_end vs lre with flood arms marked, 0.5 / 0.55 lines.

Usage: python3 lrescan_verdict.py lrescan_l1330 lrescan_l1663 ...
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec, w10_occupancy
from lurepro_source_verdict import probe_pair, envelope, direction
from lurepro_ridge_audit import components, dominant_riser, mlat_probes


def arm(d):
    m0, t, by, bz, *_ = probe_pair(d, 0.0)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    s5, _ = mlat_probes(m0)
    b5 = 1 + 4.5 * (s5 / lre) ** 2
    print(f"===== {d} (lre={lre:.1f}, x{13305.0/lre:.1f}, 5deg probe s={s5:.1f})")
    P, fs, tt = spec(by + 1j * bz, t, wce, 1024)
    w10, _ = w10_occupancy(P, fs, tt, 0, 2000)
    if w10 >= 0.20:
        print(f"[state] FLOOD (W10={w10:.3f} >= 0.20) — third source state, "
              f"no endpoint quoted\n")
        return dict(lre=lre, state="flood", w10=w10)
    dom = direction(d)
    m, t5, by, bz, *_ = probe_pair(d, dom * s5)
    z5 = by + 1j * bz
    ends, dm1 = [], None
    for nwin in (1024, 1536):
        P, fs, tt = spec(z5, t5, wce, nwin)
        dm = dominant_riser(components(P, fs, tt))
        ends.append(dm["end"] if dm else np.nan)
        if nwin == 1024:
            dm1 = dm
    if dm1 is None:
        print(f"[state] NO RISER (W10={w10:.3f} clean but no connected "
              f"sweep>=0.05 component)\n")
        return dict(lre=lre, state="noriser", w10=w10)
    w = (t5 * wce > dm1["t0"]) & (t5 * wce < dm1["t1"] + 100)
    amp = float(envelope(z5[w], t5[w], wce, 0.40, 0.80).max()) / wce
    spread = abs(ends[0] - ends[1])
    print(f"[state] ELEMENT  W10={w10:.3f}  birth {dm1['birth']:.3f}  "
          f"end(1024/1536) {ends[0]:.4f}/{ends[1]:.4f} (spread {spread:.4f})")
    print(f"        sweep {dm1['sweep']:.3f}  t[{dm1['t0']:.0f},{dm1['t1']:.0f}]  "
          f"amp {amp:.2e} B_eq | {amp/b5:.2e} B0(5°)\n")
    return dict(lre=lre, state="element", w10=w10, birth=dm1["birth"],
                end=min(ends), spread=spread, amp=amp, sweep=dm1["sweep"])


def main(dirs):
    res = [arm(d) for d in dirs]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for r in res:
        x = 13305.0 / r["lre"]   # compression factor
        if r["state"] == "element":
            ax.errorbar(x, r["end"], yerr=r["spread"], fmt="ko", ms=8)
            ax.plot(x, r["birth"], "b^", ms=7)
        elif r["state"] == "flood":
            ax.plot(x, 0.15, "rx", ms=12, mew=3)
        else:
            ax.plot(x, 0.15, "m*", ms=12)
    ax.axhline(0.5, color="gray", lw=2)
    ax.axhline(0.55, color="gray", lw=1, ls=":")
    ax.set_xlabel("compression factor 13305/lre  (x4 ... x10)")
    ax.set_ylabel(r"$\omega/\Omega_{e,eq}$")
    ax.set_title("LRE scan: endpoint (black) / birth (blue) / flood (red x)")
    ax.set_ylim(0.1, 0.8)
    fig.savefig("lrescan_state_map.png", dpi=130)
    print("fig -> lrescan_state_map.png")
    print(f"{'arm':10s} {'x':>5s} {'state':8s} {'W10':>6s} {'birth':>6s} "
          f"{'end':>7s} {'amp':>9s}")
    for d, r in zip(dirs, res):
        print(f"{d.split(chr(95))[-1]:10s} {13305.0/r['lre']:5.1f} {r['state']:8s} "
              f"{r['w10']:6.3f} {r.get('birth', float('nan')):6.3f} "
              f"{r.get('end', float('nan')):7.4f} "
              f"{r.get('amp', float('nan')):9.2e}")


if __name__ == "__main__":
    main(sys.argv[1:])
