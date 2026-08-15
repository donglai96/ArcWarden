#!/usr/bin/env python3
"""D3c production gate (triggering experiment, fuel-free f0). FROZEN
2026-08-14 before the production run's data exists. User-approved design:
drive 0.40 Ohm_e (amp 1e-4, eq column) -> off at t2000 -> watch to t8000.

Hard gate:
  G0 validity: wd_max(end) < 0.3 (delta-f health). FAIL -> exit 1.
Findings (classified, pre-registered; no optional stopping):
  V1 TRIGGER: at +-234 (geo-mean), drive-band envelope over [3000,4000]
      (>= off + 1 transit) > 1/3 of the steady drive level [1500,1900]
      AND ridge peak frequency in [3000,5000] > 0.42 (risen off the drive
      line) -> "TRIGGERED (self-sustaining)". Envelope alone -> "RINGING";
      neither -> "DIES (convective decay only)".
  V2 CORRIDOR: omega_stop = 95th-pct of the power-thresholded (>10x shot
      floor) ridge frequency at +-234 over [2500,7800].
      Classification: < 0.44 no rise / 0.46-0.55 STALLS AT FUEL EDGE
      (2D corridor-stop, H1-consistent) / > 0.58 BREAKTHROUGH (violates
      fuel-free expectation -> investigate before claiming).
  V3 UB: P_UB(0.55-0.90) at eq and +-116: late [6000,7800] / early
      [800,1400] > 10 AND late amplitude > 3x pre-drive shot floor
      (t<250 all-band) -> "UB EMERGED" -> A1 harmonic bicoherence +
      A2 latitude onset + A3 UB-shell delta-f readouts (d3a_gate reuse).
Usage: python3 d3c_gate.py <rundir>
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec
from s_gate import read_meta, band_env
from d3a_gate import probe_z, band_power_win, fv_shell_series, onset_time
from ub_mine_x4 import bicoherence_diag

OFFS = [-560.0, -480.4, -355.6, -294.6, -234.5, -175.1, -116.4, -58.1, 0.0,
        58.1, 116.4, 175.1, 234.5, 294.6, 355.6, 480.4, 560.0]


def env_win(z, ts, tOe, wce, w1, w2, t1, t2):
    e = band_env(z, ts, w1, w2, wce)
    s = (tOe > t1) & (tOe <= t2)
    return float(np.sqrt((e[s] ** 2).mean()))


def ridge(z, t, ts, tOe, wce, floor, t1, t2):
    P, fs, tt = spec(z, t, wce, 512)
    sel = (tt > t1) & (tt < t2)
    fr, pw = [], []
    for i in np.where(sel)[0]:
        j = int(np.argmax(P[i]))
        fr.append(fs[j]); pw.append(P[i][j])
    fr, pw = np.array(fr), np.array(pw)
    ok = pw > (10 * floor) ** 2 * len(fs)      # rough power threshold
    return fr[ok] if ok.any() else np.array([np.nan])


def main(d):
    m = read_meta(d)
    wce = m["wce"]
    fails = []

    E = np.genfromtxt(f"{d}/energy.csv", delimiter=",", names=True)
    ok0 = float(E["wd_max"][-1]) < 0.3
    print(f"G0: wd_max(end) = {E['wd_max'][-1]:.3f}, wd_rms = "
          f"{E['wd_rms'][-1]:.2e} -> {'PASS' if ok0 else 'FAIL'}")
    if not ok0:
        fails.append("G0 wd broken")

    zs, ss = {}, {}
    for off in (234.5, -234.5):
        z, t, ts = probe_z(d, m, OFFS.index(off))
        tOe = t * wce
        zs[off] = (z, t, ts, tOe)
    floor = np.sqrt(np.mean(np.abs(zs[234.5][0][(zs[234.5][3] > 50) &
                                                (zs[234.5][3] < 250)]) ** 2))
    floor_b0 = floor / wce
    print(f"shot floor (pre-drive, +234): {floor_b0:.2e} B0")

    # V1 ---------------------------------------------------------------
    st, po = [], []
    for off in (234.5, -234.5):
        z, t, ts, tOe = zs[off]
        st.append(env_win(z, ts, tOe, wce, 0.34, 0.46, 1500, 1900))
        po.append(env_win(z, ts, tOe, wce, 0.34, 0.46, 3000, 4000))
    steady = float(np.sqrt(st[0] * st[1])) / wce
    post = float(np.sqrt(po[0] * po[1])) / wce
    z, t, ts, tOe = zs[234.5]
    fr = ridge(z, t, ts, tOe, wce, floor, 3000, 5000)
    fpeak = np.nanmax(fr)
    sustained = post > steady / 3
    risen = fpeak > 0.42
    v1 = ("TRIGGERED (self-sustaining)" if sustained and risen else
          "RINGING (level holds, no rise)" if sustained else
          "DIES (convective decay)")
    print(f"V1: steady {steady:.2e}, post-off [3000,4000] {post:.2e} "
          f"(ratio {post / steady:.2f} vs 1/3), ridge peak {fpeak:.3f} "
          f"-> {v1}")

    # V2 ---------------------------------------------------------------
    fr2 = np.concatenate([ridge(*zs[o][0:3], zs[o][3], wce, floor,
                                2500, 7800) for o in (234.5, -234.5)])
    wstop = float(np.nanpercentile(fr2, 95))
    v2 = ("no rise" if wstop < 0.44 else
          "STALLS AT FUEL EDGE (corridor-stop)" if 0.46 <= wstop <= 0.55 else
          "BREAKTHROUGH >0.58 — investigate" if wstop > 0.58 else
          "between bins — report as measured")
    print(f"V2: omega_stop (95th pct thresholded ridge, [2500,7800]) = "
          f"{wstop:.3f} vs fuel edge 0.50 -> {v2}")

    # V3 ---------------------------------------------------------------
    emerged = False
    for off in (0.0, 116.4, -116.4):
        z3, t3, ts3 = probe_z(d, m, OFFS.index(off))
        t3Oe = t3 * wce
        early = band_power_win(z3, t3Oe, ts3, wce, 0.55, 0.90, 800, 1400)
        late = band_power_win(z3, t3Oe, ts3, wce, 0.55, 0.90, 6000, 7800)
        amp_late = np.sqrt(late) / wce
        hit = (late / early > 10) and (amp_late > 3 * floor_b0)
        emerged |= hit
        print(f"V3 [{off:+6.1f}]: P_UB late/early = {late / early:6.2f}, "
              f"amp {amp_late:.2e} vs 3x floor {3 * floor_b0:.2e} -> "
              f"{'UB' if hit else 'quiet'}")
    if emerged:
        z3, t3, ts3 = probe_z(d, m, OFFS.index(0.0))
        t3Oe = t3 * wce
        f1, b2b, nw = bicoherence_diag(z3[t3Oe > 2500], ts3, wce)
        band = (f1 > 0.30) & (f1 < 0.50)
        j = int(np.argmax(b2b * band))
        print(f"  A1 harmonic: b2 = {b2b[j]:.3f} at f = {f1[j]:.3f} "
              f"(3x floor {3.0 / nw:.3f})")
        for off in (0.0, 116.4, 234.5):
            zz, ttv, _ = probe_z(d, m, OFFS.index(off))
            e = band_env(zz, ts3, 0.55, 0.70, wce)
            print(f"  A2 [{off:+6.1f}]: UB onset t = "
                  f"{onset_time(e, ttv * wce):.0f}")
        tf, fsh, _ = fv_shell_series(d, m)
        j0 = max(1, len(fsh) // 10)
        print(f"  A3 UB-shell |df| growth x"
              f"{fsh[-1] / (abs(fsh[:j0]).mean() + 1e-300):.1f}")

    # figure -----------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, off in zip(axs.flat[:3], (0.0, 234.5, -234.5)):
        zz, ttv, _ = probe_z(d, m, OFFS.index(off))
        P, fsp, ttp = spec(zz, ttv, wce, 512)
        ax.pcolormesh(ttp, fsp, np.log10(P.T + 1e-30), cmap="turbo",
                      vmin=np.log10(P.max()) - 5, vmax=np.log10(P.max()),
                      shading="auto", rasterized=True)
        ax.axhline(0.5, color="w", ls="--", lw=1)
        ax.axhline(0.40, color="m", ls=":", lw=1)
        ax.axvline(2000, color="w", lw=0.5)
        ax.set_ylim(0.02, 0.95); ax.set_title(f"off {off:+.0f}")
    ax = axs.flat[3]
    z, t, ts, tOe = zs[234.5]
    for w1, w2, lab in ((0.34, 0.46, "drive"), (0.46, 0.55, "edge"),
                        (0.55, 0.90, "UB")):
        ax.semilogy(tOe, band_env(z, ts, w1, w2, wce) / wce, label=lab)
    ax.axvline(2000, color="k", lw=0.5); ax.legend(fontsize=8)
    ax.set_title("+234 envelopes /B0")
    fig.savefig("d3c_gate.png", dpi=120)
    print("fig -> d3c_gate.png")

    if fails:
        print("D3c GATE: FAIL —", "; ".join(fails))
        return 1
    print(f"D3c GATE: PASS — V1 {v1} | V2 {v2} | V3 "
          f"{'UB EMERGED' if emerged else 'UB quiet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
