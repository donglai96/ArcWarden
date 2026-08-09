#!/usr/bin/env python3
"""LRE scan — nonlinear inhomogeneity ratio S along the element ridge
(PLAN_LRE_SCAN §4/§6).

    S = -(1/(s0 w Ww)) [ s1 dw/dt + c s2 dWe/ds ],   |S|<1 = stable trapping

with Omura-2021-review coefficients (c=1 units, LOCAL We at the 5° probe):
    s0 = delta Uperp0 / xi
    s1 = gamma (1 - VR/Vg)^2
    s2 = (1/(2 xi delta)) { gamma (w/We) Uperp0^2
          - [2 + Lambda delta^2 (We - gamma w)/(We - w)] VR Vp }
    delta = 1/sqrt(1+xi^2), xi^2 = w(We-w)/wpe_c^2, Vp = xi delta,
    VR from the relativistic resonance (omura_threshold.branch form),
    Uperp0 = sqrt(pi/2) uperp  (pure biMax), Lambda = 1.
Measured inputs along the dominant ridge at the MLAT-5° main probe:
    w(t)   : connected-component ridge (frozen tracker), smoothed
    dw/dt  : gradient of smoothed ridge
    Ww(t)  : Bw envelope in the component band (Ww = Bw in code units)
    dWe/ds : analytic parabolic 9 s5 / lre^2 * wce  (a = 4.5/lre^2)
Expectation (pre-registered): stop arms show |S| -> 1 near the endpoint;
cross arms keep |S| < 1 at the same frequencies; systematic in lre.

Usage: python3 lrescan_S_diag.py <dir> [<dir> ...]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec
from lurepro_source_verdict import probe_pair, envelope, direction
from lurepro_ridge_audit import components, dominant_riser, mlat_probes

UPERP = 0.345378
UPERP0 = np.sqrt(np.pi / 2) * UPERP     # pure biMax mean perp speed
LAM = 1.0


def coeffs(w, We, wpe2):
    """Omura coefficients at LOCAL We; all in code (wpe=1, c=1) units."""
    xi2 = w * (We - w) / wpe2
    xi = np.sqrt(xi2)
    dl = 1.0 / np.sqrt(1.0 + xi2)          # Omura's delta
    Vp = xi * dl
    wt = w / We
    A = wt * wt + Vp * Vp
    VR = (wt * wt - np.sqrt(wt ** 4 + A * (1 - wt * wt - UPERP0 ** 2))) / A * Vp
    gam = 1.0 / np.sqrt(1.0 - VR * VR - UPERP0 ** 2)
    k = w / Vp
    dw = 1e-4 * We
    def kk(wq):
        x2 = wq * (We - wq) / wpe2
        return wq * np.sqrt(1 + 1 / x2) if np.ndim(wq) == 0 else None
    Vg = (2 * dw) / (kk(w + dw) - kk(w - dw))
    s0 = dl * UPERP0 / xi
    s1 = gam * (1 - VR / Vg) ** 2
    s2 = (gam * wt * UPERP0 ** 2
          - (2 + LAM * dl * dl * (We - gam * w) / (We - w)) * VR * Vp) \
        / (2 * xi * dl)
    return dict(k=k, Vg=Vg, Vp=Vp, VR=VR, gam=gam, s0=s0, s1=s1, s2=s2)


def ridge_S(d):
    m0, t, by, bz, *_ = probe_pair(d, 0.0)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    s5, _ = mlat_probes(m0)
    b5 = 1 + 4.5 * (s5 / lre) ** 2
    We_loc = wce * b5                       # LOCAL cyclotron freq at probe
    wpe2 = m0.get("cold_nc", 0.99)          # cold wpe^2 (code units)
    dWeds = wce * 9.0 * s5 / lre ** 2       # parabolic dipole gradient
    dom = direction(d)
    m, t5, by, bz, *_ = probe_pair(d, dom * s5)
    z5 = by + 1j * bz
    P, fs, tt = spec(z5, t5, wce, 1024)
    dm = dominant_riser(components(P, fs, tt))
    if dm is None:
        print(f"{d}: no riser — skip")
        return None
    rt, rf = dm["ridge"]
    # smooth ridge and slope (boxcar ~150/Oe)
    nb = max(3, int(150.0 / (rt[1] - rt[0])) | 1)
    ker = np.ones(nb) / nb
    rfs = np.convolve(rf, ker, mode="same")
    edge = nb // 2 + 1
    rt, rfs = rt[edge:-edge], rfs[edge:-edge]
    dwdt = np.gradient(rfs * wce, rt / wce)          # dw/dt in wpe units^2
    # Bw along ridge: component-band envelope sampled at ridge times
    w = (t5 * wce > dm["t0"]) & (t5 * wce < dm["t1"] + 100)
    f1, f2 = max(0.15, dm["birth"] - 0.05), dm["top"] + 0.05
    env = envelope(z5[w], t5[w], wce, f1, f2)
    te = t5[w] * wce
    Bw = np.interp(rt, te, env)                       # code B units
    S, wtr = np.full(len(rt), np.nan), np.full(len(rt), np.nan)
    for i in range(len(rt)):
        wa = rfs[i] * wce                             # absolute w
        if not (0.05 * We_loc < wa < 0.95 * We_loc) or Bw[i] <= 0:
            continue
        c = coeffs(wa, We_loc, wpe2)
        Ww = Bw[i]                                    # Omega_w = Bw (qm=-1)
        wtr[i] = np.sqrt(c["k"] * UPERP0 * Ww / c["gam"])
        S[i] = -(1.0 / (c["s0"] * wa * Ww)) * (c["s1"] * dwdt[i]
                                               + c["s2"] * dWeds)
    return dict(lre=lre, t=rt, w=rfs, wloc=rfs * wce / We_loc, S=S,
                wtr=wtr, Bw=Bw, dwdt=dwdt, end=dm["end"])


def main(dirs):
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                           constrained_layout=True)
    for d in dirs:
        r = ridge_S(d)
        if r is None:
            continue
        lab = f"x{13305.0 / r['lre']:.0f} end {r['end']:.3f}"
        ax[0].plot(r["w"], np.abs(r["S"]), ".-", ms=3, label=lab)
        ax[1].plot(r["w"], r["Bw"] / 0.2, ".-", ms=3, label=lab)
        i = np.nanargmax(np.abs(r["S"]))
        print(f"{d}: |S| max {np.nanmax(np.abs(r['S'])):.2f} at "
              f"w={r['w'][i]:.3f}; |S|(w=0.48-0.52) = "
              f"{np.nanmedian(np.abs(r['S'])[(r['w']>0.48)&(r['w']<0.52)]):.2f}")
    ax[0].axhline(1.0, color="r", lw=2)
    ax[0].axhline(0.4, color="gray", ls=":", lw=1)   # Omura optimum |S|
    ax[0].axvline(0.5, color="gray", lw=1)
    ax[0].set_ylabel("|S| along ridge")
    ax[0].set_ylim(0, 3)
    ax[0].legend(fontsize=8)
    ax[1].axvline(0.5, color="gray", lw=1)
    ax[1].set_xlabel(r"$\omega/\Omega_{e,eq}$")
    ax[1].set_ylabel(r"$B_w/B_{0,eq}$ along ridge")
    ax[1].set_yscale("log")
    fig.savefig("lrescan_S_diag.png", dpi=130)
    print("fig -> lrescan_S_diag.png")


if __name__ == "__main__":
    main(sys.argv[1:])
