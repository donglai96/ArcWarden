#!/usr/bin/env python3
"""UBSCAN frozen gate (2026-08-16): pancake FUEL ladder n2 on the
giant_x4_atmo40 engine — does more low-E anisotropic fuel turn the
episodic UB episode (expG2_ctrl_r3, Step-3 verdict) into a sustained
upper band, and does a 0.5 valley appear between the two sources?

PRE-REGISTERED (kinetic solver ub_net_threshold.py, engine bare-A 0.52 +
cold, A2=5.0 u_par2=0.06 fixed, only n2 moves — one-parameter fuel axis):
  arm            n2      gamma_UB max  band(+)         t_ign(x2.3 conv)
  expG2_ctrl_r3  0.0035  +6.6e-4 @0.52 [0.45,0.66]     ~11900/We (REF, on disk)
  ubscan_n70     0.0070  +2.5e-3 @0.52 [0.45,0.75]     ~3100/We
  ubscan_n140    0.0140  +6.1e-3 @0.52 [0.45,0.79]     ~1300/We
  LB engine-band gamma: 1.7e-3 / 1.8e-3 / 4.4e-3 (n140 = FLOOD RISK,
  Chen PoP 2026 hiss regime; flood reading pre-registered below).

READOUTS (verdict station = EQUATOR, same convention as fscan gate):
  U1 ignition: UB(0.55-0.80) envelope peak >= 5x early floor AND positive
     log-slope fit (bar CALIBRATED on the ruled reference: 6.3x, +7.6e-4;
     a 10x floor bar misread the ruled-IGNITED arm — recorded).
  U2 fuel law (HEADLINE): TREND measure, adopted from the Step-3 ruling
     itself (box P_UB declined 7e-6->3.8e-6 through the 2nd LB burst):
     R_late = median a_ub^2 [t>12000] / median a_ub^2 [episode-1 window].
     SUSTAINED if R_late >= 0.5; reference reads 0.326 = EPISODIC (matches
     ruling; margin 1.5x below bar). Frame-threshold "alive after t12000" was FALSIFIED on the
     reference (LB-skirt leakage + gap-denominator flickers) — trap class:
     same family as snapshot-JdotE / box-avg / wd-scaled-floor.
     Secondary monotone readouts: alive-duration (Q>=3 prominence AND
     a_ub>=2x floor, unsmoothed) + fluence over alive frames.
  U3 band: occupied UB median + 90th-pct upper edge vs predicted band top.
  U4 gap: frames with both bands alive: R_gap = P(0.47-0.53)/geomean
     (P_LB, P_UB); visible valley bar R_gap < 0.5 (ArmU-class).
  FLOOD guard (fscan calibration, ruled data): W10 >= 0.235 AND
     Bw >= 1.2e-2 -> FLOOD; arm then void for U3/U4 organization claims
     but band powers remain a fuel datum.
  Failure reading (pre-registered): n70+n140 both episodic despite
     3.8x/9.3x gamma -> exhaustion is amplitude-side (QL flattening at
     fixed resonant shell), fuel axis closed -> next axis = u_par2/A2
     (USER decision). Both flood -> ladder overshot, midpoint n2 needs
     USER approval (no roulette).

Usage (from build/): python3 ../scripts/ubscan_gate.py [arm ...]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../scripts")
from plot_chen2026_fig1 import load_probe          # noqa: E402
from lumorph_gate import spec, w10_occupancy       # noqa: E402

ARMS = {  # n2, (gamma_pred, band_hi, t_ign_pred)
    "expG2_ctrl_r3": (0.0035, 6.6e-4, 0.66, 11900),
    "ubscan_n70":    (0.0070, 2.5e-3, 0.75, 3100),
    "ubscan_n140":   (0.0140, 6.1e-3, 0.79, 1300),
}
LB_BAND = (0.20, 0.45)
GAP_BAND = (0.47, 0.53)
UB_BAND = (0.55, 0.80)
W10BAR, FLOOD_BW = 0.235, 1.2e-2   # frozen fscan conjunction bar
NWIN = 1024


def band_amp(P, fs, lo, hi):
    """per-frame amplitude proxy sqrt(sum P) in a band."""
    m = (fs >= lo) & (fs < hi)
    return np.sqrt(P[:, m].sum(axis=1))


def station(d, off=0.0):
    m, t, by, bz = load_probe(d, off)
    z = by + 1j * bz
    P, fs, tt = spec(z, t, 0.2, NWIN)
    a_lb = band_amp(P, fs, *LB_BAND)
    a_gp = band_amp(P, fs, *GAP_BAND)
    a_ub = band_amp(P, fs, *UB_BAND)
    return m, P, fs, tt, a_lb, a_gp, a_ub


def analyze(d):
    n2, gpred, bhi_pred, tpred = ARMS[d]
    m, P, fs, tt, a_lb, a_gp, a_ub = station(d)
    early = tt < 500
    floor = np.median(a_ub[early])
    pk = a_ub.max(); tpk = tt[np.argmax(a_ub)]
    # measured growth rate: log-linear fit from 2x floor to half-peak
    g_meas = np.nan
    try:
        i0 = np.where(a_ub > 2 * floor)[0][0]
        i1 = np.where(a_ub > 0.5 * pk)[0][0]
        if i1 > i0 + 3:
            g_meas = np.polyfit(tt[i0:i1], np.log(a_ub[i0:i1]), 1)[0]
    except IndexError:
        pass
    ignited = (pk >= 5 * floor) and (np.isfinite(g_meas) and g_meas > 0)
    # U2 fuel law: prominence-gated alive frames + episode/late trend
    mb = (fs >= UB_BAND[0]) & (fs < UB_BAND[1])
    mg = np.abs(fs - 0.5) < 0.015
    Q = P[:, mb].max(axis=1) / np.median(P[:, mg], axis=1)
    alive = (Q >= 3.0) & (a_ub >= 2 * floor)
    dtf = tt[1] - tt[0]
    t_alive = alive.sum() * dtf
    fluence = (a_ub[alive] ** 2).sum() * dtf if alive.any() else 0.0
    last_alive = tt[alive][-1] if alive.any() else np.nan
    # trend: episode-1 window = 2000/We from first half-peak crossing
    r_late = np.nan
    try:
        t_ign = tt[np.where(a_ub >= 0.5 * pk)[0][0]]
        ep1 = (tt >= t_ign) & (tt < t_ign + 2000)
        late = tt > 12000
        if ep1.any() and late.any():
            r_late = (np.median(a_ub[late] ** 2)
                      / np.median(a_ub[ep1] ** 2))
    except IndexError:
        pass
    sustained = np.isfinite(r_late) and r_late >= 0.5
    # U3 band morphology over alive frames
    med_f = hi_f = np.nan
    if alive.any():
        mb = (fs >= UB_BAND[0]) & (fs < UB_BAND[1])
        Pu = P[alive][:, mb].sum(axis=0)
        cdf = np.cumsum(Pu) / Pu.sum()
        med_f = fs[mb][np.searchsorted(cdf, 0.5)]
        hi_f = fs[mb][np.searchsorted(cdf, 0.9)]
    # U4 gap over both-bands-alive frames
    both = (a_lb ** 2 > 0.1 * a_lb.max() ** 2) & (a_ub ** 2 > 0.1 * pk ** 2)
    r_gap = np.nan
    if both.any():
        r_gap = np.median((a_gp[both] ** 2)
                          / (a_lb[both] * a_ub[both] + 1e-30))
    # flood guard
    w10, _ = w10_occupancy(P, fs, tt, 500, 2000)
    bw = np.max(np.convolve(band_amp(P, fs, 0.20, 0.75),
                            np.ones(5) / 5, "same"))
    flood = (not np.isnan(w10)) and w10 >= W10BAR and bw >= FLOOD_BW
    # near-equatorial confinement: UB share at +-291 vs eq
    shares = []
    for off in (-291.0, 291.0):
        _, P2, f2, t2, l2, g2, u2 = station(d, off)
        shares.append(u2.max() / pk if pk > 0 else np.nan)
    print(f"\n=== {d} (n2={n2}) ===")
    print(f"U1 ignition: floor {floor:.2e}, peak {pk:.2e} @t{tpk:.0f} "
          f"({pk/floor:.1f}x) -> {'IGNITED' if ignited else 'NO'}; "
          f"gamma_meas {g_meas:+.2e} vs pred {gpred:+.1e} "
          f"(t_ign pred ~{tpred}/We)")
    print(f"U2 fuel: R_late {r_late:.3f} -> "
          f"{'SUSTAINED' if sustained else 'EPISODIC'}; "
          f"alive {t_alive:.0f}/We (last t{last_alive:.0f}), "
          f"fluence {fluence:.3e}")
    print(f"U3 band: median {med_f:.3f}, 90pct edge {hi_f:.3f} "
          f"(pred top {bhi_pred})")
    print(f"U4 gap: R_gap {r_gap:.3f} "
          f"({'VISIBLE VALLEY' if r_gap < 0.5 else 'no valley'})")
    print(f"FLOOD guard: W10 {w10:.3f}, Bw {bw:.2e} -> "
          f"{'FLOOD' if flood else 'ok'}")
    print(f"confinement: UB peak at +-291 = {shares[0]:.2f}/{shares[1]:.2f} "
          f"of eq")
    return dict(n2=n2, floor=floor, pk=pk, g=g_meas, alive=t_alive,
                fluence=fluence, last=last_alive, rlate=r_late,
                sustained=sustained,
                med=med_f, hi=hi_f, rgap=r_gap, w10=w10, bw=bw, flood=flood,
                P=P, fs=fs, tt=tt, aub=a_ub)


def main(arms):
    res = {}
    for d in arms:
        res[d] = analyze(d)
    if len(res) > 1:
        order = sorted(res, key=lambda d: res[d]["n2"])
        al = [res[d]["alive"] for d in order]
        fl = [res[d]["fluence"] for d in order]
        mono_a = all(x < y for x, y in zip(al, al[1:]))
        mono_f = all(x < y for x, y in zip(fl, fl[1:]))
        print("\n=== CROSS-ARM (fuel law) ===")
        print("alive/We :", " ".join(f"{x:.0f}" for x in al),
              "MONOTONE" if mono_a else "NOT monotone")
        print("fluence  :", " ".join(f"{x:.2e}" for x in fl),
              "MONOTONE" if mono_f else "NOT monotone")
        print("HEADLINE:", "fuel law HOLDS" if (mono_a and mono_f)
              else "fuel law FAILS -> exhaustion is amplitude-side")
    # figure
    fig, axs = plt.subplots(len(res), 1, figsize=(11, 3.2 * len(res)),
                            sharex=True, constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, d in zip(axs, res):
        r = res[d]
        ex = [r["tt"][0], r["tt"][-1], r["fs"][0], r["fs"][-1]]
        ax.imshow(10 * np.log10(r["P"].T + 1e-22), origin="lower",
                  aspect="auto", extent=ex, cmap="turbo",
                  vmin=10 * np.log10(r["P"].max()) - 55,
                  vmax=10 * np.log10(r["P"].max()))
        ax.axhline(0.5, color="w", lw=1.2, ls="--")
        for b in UB_BAND:
            ax.axhline(b, color="w", lw=0.6, ls=":")
        ax.set_ylabel(f"{d}\n$\\omega/\\Omega_e$")
    axs[-1].set_xlabel(r"$t\,\Omega_e$")
    fig.savefig("ubscan_gate.png", dpi=120)
    print("\nfig -> ubscan_gate.png")


if __name__ == "__main__":
    main(sys.argv[1:] or list(ARMS))
