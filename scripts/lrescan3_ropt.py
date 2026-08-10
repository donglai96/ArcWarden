#!/usr/bin/env python3
"""LRE scan v3 — over-optimal ratio R_opt = Bw(w)/B_opt(w) along the ridge.

FROZEN BEFORE v3 PILOT DATA (2026-08-09). Pre-registered chain (deck
headers, commit 14fadbe): w_birth falls with lre; R_opt rises through ~1
at the cross->stall boundary; endpoint bifurcates within one f0.

B_opt / B_th from Omura 2021 review (scripts/omura_threshold.py form),
evaluated with the v3 fuel-open f0 (pure-biMax Uperp0 approximation for
the conecut load, documented):
    Utpar = 0.19783565, Utperp = 0.31306 (bare A = 1.5), nh = 0.006,
    cold wpe^2 = 0.994, Oe = 0.2, Q = 0.5, tau = 0.5, Lam = 1.
B_opt is lre-INDEPENDENT -> any R_opt trend across arms comes from the
measured Bw side alone.

Per arm: frozen CC ridge at the MLAT-5deg main probe (same machinery as
lrescan_S_diag), Bw = component-band envelope on the ridge, then
    R_opt(w) = Bw/B_opt(w),  R_th(w) = Bw/B_th(w,lre)
Quotes: w_birth, w_end, R_opt at w=0.5 (interpolated if the ridge spans
it, else at the ridge top), R_opt at the endpoint.

Usage: python3 lrescan3_ropt.py <dir> [<dir> ...]
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

OE = 0.2
NH = 0.006
NC = 0.994
UTPAR = 0.19783565
UTPERP = 0.31306
UPERP0 = np.sqrt(np.pi / 2) * UTPERP
WPE = np.sqrt(NC)
WPH_T = np.sqrt(NH) / OE
Q, TAU, LAM = 0.5, 0.5, 1.0


def disp_k(w):
    return w * np.sqrt(1 + WPE ** 2 / (w * (OE - w)))


def branch(wt):
    w = wt * OE
    dw = 1e-6
    Vg = (2 * dw * OE) / (disp_k(w + dw * OE) - disp_k(w - dw * OE))
    xi = np.sqrt(w * (OE - w)) / WPE
    chi = 1 / np.sqrt(1 + xi * xi)
    Vp = chi * xi
    A = wt * wt + Vp * Vp
    VR = (wt * wt - np.sqrt(wt ** 4 + A * (1 - wt * wt - UPERP0 ** 2))) / A * Vp
    gam = 1 / np.sqrt(1 - VR * VR - UPERP0 ** 2)
    s2 = (gam * wt * UPERP0 ** 2
          - (2 + LAM * chi ** 2 * (1 - gam * wt) / (1 - wt)) * VR * Vp) \
        / (2 * xi * chi)
    return Vg, xi, chi, Vp, VR, gam, s2


def b_opt(wt):
    Vg, xi, chi, Vp, VR, gam, s2 = branch(wt)
    return (0.8 * np.pi ** -2.5 * (Q * Vp * Vg * UPERP0 * WPH_T ** 2)
            / (TAU * wt * UTPAR ** 2) * (1 - VR / Vg)
            * np.exp(-gam ** 2 * VR ** 2 / (2 * UTPAR ** 2)))


def b_th(wt, lre):
    at = 112.5 / lre ** 2
    Vg, xi, chi, Vp, VR, gam, s2 = branch(wt)
    return ((100 * np.pi ** 3 * gam ** 4 * xi)
            / (wt * WPH_T ** 4 * (chi * UPERP0) ** 5)
            * (at * s2 * UTPAR / Q) ** 2
            * np.exp(gam ** 2 * VR ** 2 / UTPAR ** 2))


def ridge_ropt(d):
    m0, t, by, bz, *_ = probe_pair(d, 0.0)
    wce = m0["wce"]
    lre = m0.get("b0_lre", 1330.504)
    s5, _ = mlat_probes(m0)
    dom = direction(d)
    m, t5, by, bz, *_ = probe_pair(d, dom * s5)
    z5 = by + 1j * bz
    P, fs, tt = spec(z5, t5, wce, 1024)
    dm = dominant_riser(components(P, fs, tt))
    if dm is None:
        print(f"{d}: no riser — skip")
        return None
    rt, rf = dm["ridge"]
    nb = max(3, int(150.0 / (rt[1] - rt[0])) | 1)
    ker = np.ones(nb) / nb
    rfs = np.convolve(rf, ker, mode="same")
    edge = nb // 2 + 1
    rt, rfs = rt[edge:-edge], rfs[edge:-edge]
    w = (t5 * wce > dm["t0"]) & (t5 * wce < dm["t1"] + 100)
    f1, f2 = max(0.15, dm["birth"] - 0.05), dm["top"] + 0.05
    env = envelope(z5[w], t5[w], wce, f1, f2)
    Bw = np.interp(rt, t5[w] * wce, env) / wce          # in B0 units
    ok = (rfs > 0.1) & (rfs < 0.9) & (Bw > 0)
    Ropt = np.full(len(rt), np.nan)
    Rth = np.full(len(rt), np.nan)
    for i in np.where(ok)[0]:
        Ropt[i] = Bw[i] / b_opt(rfs[i])
        Rth[i] = Bw[i] / b_th(rfs[i], lre)
    # R_opt at w=0.5: interpolate on the rising part if the ridge spans it
    r05 = np.nan
    if np.nanmax(rfs[ok]) >= 0.5 >= np.nanmin(rfs[ok]):
        j = np.where(ok & (rfs >= 0.5))[0]
        if len(j):
            r05 = Ropt[j[0]]
    return dict(lre=lre, t=rt, w=rfs, Ropt=Ropt, Rth=Rth, Bw=Bw,
                birth=dm["birth"], end=dm["end"], r05=r05)


def main(dirs):
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                           constrained_layout=True)
    for d in dirs:
        r = ridge_ropt(d)
        if r is None:
            continue
        lab = f"x{13305.0 / r['lre']:.0f} birth {r['birth']:.2f} end {r['end']:.3f}"
        ax[0].plot(r["w"], r["Ropt"], ".-", ms=3, label=lab)
        ax[1].plot(r["w"], r["Bw"], ".-", ms=3, label=lab)
        iend = np.where(np.isfinite(r["Ropt"]))[0]
        rend = r["Ropt"][iend[-1]] if len(iend) else np.nan
        print(f"{d}: birth {r['birth']:.3f}  end {r['end']:.3f}  "
              f"R_opt(0.5) = {r['r05']:.2f}  R_opt(end) = {rend:.2f}  "
              f"max Bw/B0 = {np.nanmax(r['Bw']):.2e}")
    ws = np.linspace(0.15, 0.75, 100)
    ax[1].plot(ws, [b_opt(w) for w in ws], "k--", lw=1, label="B_opt (Omura)")
    ax[0].axhline(1.0, color="r", lw=2)
    ax[0].axvline(0.5, color="gray", lw=1)
    ax[0].set_ylabel(r"$R_{opt} = B_w/B_{opt}$ along ridge")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[1].axvline(0.5, color="gray", lw=1)
    ax[1].set_xlabel(r"$\omega/\Omega_{e,eq}$")
    ax[1].set_ylabel(r"$B_w/B_0$ along ridge")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    fig.savefig("lrescan3_ropt.png", dpi=130)
    print("fig -> lrescan3_ropt.png")


if __name__ == "__main__":
    main(sys.argv[1:])
