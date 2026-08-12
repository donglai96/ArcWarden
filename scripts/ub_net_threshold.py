#!/usr/bin/env python3
"""UB Step 1 (PLAN_UB_1D v2 §2): NET upper-band ignition threshold for a
low-energy anisotropic population on top of the giant_x4 element engine.

Parallel R-mode kinetic dispersion, cold + TWO bi-Maxwellian hot species
(whistler_kinetic_dispersion.py form):
    D = 1 - k^2/w^2 + (wpc^2/w^2) w/(We-w)
        + sum_s (wps^2/w^2)[zeta0 Z(zeta) + A_s(1 + zeta Z(zeta))]
Engine (fixed) = giant_x4 hot electrons as bare biMax: n=0.0178,
u_par=0.19783565, u_perp=0.24390817 (A=0.52; the conecut load raises the
effective A to ~0.8 — bare biMax is the CONSERVATIVE engine-damping case,
noted in output). Low-E species: u_par2 = 0.045c (~0.5 keV), A2 and n2
scanned. wpe/We = 5. For each (n2, A2): gamma_UB = max gamma over
w in [0.55, 0.75] We. Threshold = gamma_UB = 0 contour.
Overlays: bare KP marginal A(w)=w/(We-w) at the band bottom (1.22),
the G2 FAILURE point (n2=0.0035, A2=5.0, u_par2=0.06 — recomputed with
its own u_par2 as a cross-check), and the engine-only damping magnitude.

Output: build/ub_net_threshold.png + printed table.
"""
import numpy as np
from scipy.special import wofz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPE = 5.0            # wpe/We
WE = 1.0
ENG = dict(n=0.0178, upar=0.19783565, uper=0.24390817)
UPAR2 = 0.045 * 5.0  # c=... units: solver works in We=1, v in c with c=1?
# NOTE on units: whistler_kinetic_dispersion.py uses We=1, c=1, wpe as given,
# velocities in c. Our deck units are wpe=1, c=1, We=0.2 — SAME physical
# ratios if we set wpe=5, We=1 and keep velocities in c. u in c units:
UPAR2 = 0.045


def Z(z):
    return 1j * np.sqrt(np.pi) * wofz(z)


def chi_hot(w, k, n, upar, A):
    wps2 = n * WPE * WPE
    s2ka = np.sqrt(2.0) * k * upar
    z0 = w / s2ka
    zt = (w - WE) / s2ka
    Zz = Z(zt)
    return (wps2 / w ** 2) * (z0 * Zz + A * (1.0 + zt * Zz))


def D(w, k, species, nc):
    r = 1.0 - k * k / (w * w) + (nc * WPE * WPE / w ** 2) * w / (WE - w)
    for (n, upar, A) in species:
        r += chi_hot(w, k, n, upar, A)
    return r


def cold_root(k):
    lo, hi = 1e-9, WE * (1 - 1e-9)
    f = lambda w: w * w + w * WPE * WPE / (WE - w) - k * k
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_k(k, species, nc, w0=None):
    w = complex(w0 if w0 is not None else cold_root(k), 1e-4)
    for _ in range(120):
        h = 1e-8 * (abs(w) + 1e-8)
        dD = (D(w + h, k, species, nc) - D(w - h, k, species, nc)) / (2 * h)
        step = D(w, k, species, nc) / dD
        w -= step
        if abs(step) < 1e-13:
            break
    return w


def gamma_ub(species, nc, w1=0.55, w2=0.75):
    """max growth (or least damping) with w_r in [w1, w2] We."""
    best = -np.inf
    prev = None
    for k in np.linspace(4.8, 9.2, 110):
        w = solve_k(k, species, nc, w0=prev)
        prev = w.real
        if w1 * WE <= w.real <= w2 * WE:
            best = max(best, w.imag)
    return best


def main():
    eng = (ENG["n"], ENG["upar"], (ENG["uper"] / ENG["upar"]) ** 2 - 1.0)

    g0 = gamma_ub([eng], 1.0 - ENG["n"])
    print(f"engine-only (bare A={eng[2]:.2f}): gamma_UB = {g0:+.3e} We "
          f"(the absorption floor the low-E population must beat)")

    A2s = np.linspace(0.5, 8.0, 16)
    n2s = np.logspace(np.log10(3e-4), np.log10(2e-2), 14)
    G = np.full((len(n2s), len(A2s)), np.nan)
    for i, n2 in enumerate(n2s):
        for j, A2 in enumerate(A2s):
            sp = [eng, (n2, UPAR2, A2)]
            G[i, j] = gamma_ub(sp, 1.0 - ENG["n"] - n2)
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    pc = ax.pcolormesh(A2s, n2s, np.sign(G) * np.sqrt(np.abs(G)),
                       cmap="RdBu_r", shading="auto",
                       vmin=-np.sqrt(np.nanmax(np.abs(G))),
                       vmax=np.sqrt(np.nanmax(np.abs(G))))
    fig.colorbar(pc, ax=ax, label=r"sign($\gamma$)·$\sqrt{|\gamma_{UB}|}$")
    cs = ax.contour(A2s, n2s, G, levels=[0.0], colors="k", linewidths=2.5)
    ax.clabel(cs, fmt="net threshold")
    for lev, ls in ((1e-4, "--"), (1e-3, ":")):
        c2 = ax.contour(A2s, n2s, G, levels=[lev], colors="k",
                        linewidths=1.0, linestyles=ls)
        ax.clabel(c2, fmt=f"{lev:g}")
    ax.axvline(0.55 / 0.45, color="g", lw=2,
               label="bare KP marginal @0.55 (A=1.22)")
    # G2 failure point (its own u_par2=0.06): recompute gamma for honesty
    g2 = gamma_ub([eng, (0.0035, 0.06, 5.0)], 1.0 - ENG["n"] - 0.0035)
    ax.plot(5.0, 0.0035, "kx", ms=14, mew=3,
            label=f"G2 point (u2=0.06): gamma={g2:+.1e}")
    ax.set_yscale("log")
    ax.set_xlabel(r"low-E anisotropy $A_2$")
    ax.set_ylabel(r"low-E density $n_2/n_0$")
    ax.set_title(f"NET UB threshold (engine bare-A={eng[2]:.2f} + cold, "
                 f"wpe/We=5, u_par2={UPAR2}c ~0.5 keV)")
    ax.legend(fontsize=9, loc="lower left")
    fig.savefig("ub_net_threshold.png", dpi=130)
    print("fig -> ub_net_threshold.png")

    # threshold table
    print("\nn2 -> minimum A2 for net gamma_UB > 0:")
    for i, n2 in enumerate(n2s):
        row = G[i]
        j = np.where(row > 0)[0]
        a_min = A2s[j[0]] if len(j) else np.inf
        print(f"  n2 = {n2:.2e}: A2_min = {a_min:.2f}"
              + ("" if np.isfinite(a_min) else "  (never ignites in scan)"))
    print(f"\nG2 retro-check: gamma_UB(G2 params) = {g2:+.3e} We "
          f"({'consistent with its measured failure' if g2 < 1e-4 else 'UNEXPECTED'})")


if __name__ == "__main__":
    main()
