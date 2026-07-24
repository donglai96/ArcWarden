#!/usr/bin/env python3
"""G0.2 — Kinetic parallel-whistler growth spectrum for multi-bi-Maxwellian
electron plasmas (kappa emulated by bi-Max sums). Purpose: design the
flagship f0 — confirm upper-band fuel exists (0.1-2.6 keV anisotropy)
without disturbing the lower-band chirping recipe.

Dispersion (R-mode, parallel, nonrelativistic, immobile ions):
  D(w,k) = 1 - c^2k^2/w^2
           + sum_s (wps^2/w^2) [ z0 Z(zeta) + A_s (1 + zeta Z(zeta)) ]
  zeta = (w - We)/(sqrt2 k vpar_s),  z0 = w/(sqrt2 k vpar_s),
  A_s = Tperp/Tpar - 1,  Z via Faddeeva (scipy wofz).
Cold species use the analytic cold susceptibility.

Built-in gates (run = they print PASS/FAIL):
  G1 cold limit reproduces the cold whistler branch to <1e-6;
  G2 Kennel-Petschek marginal frequency w_m = We*A/(A+1) for a single
     bi-Max (Case II: A=0.520 -> w_m = 0.342 We);
  G3 gamma_max for Case II params matches the measured linear growth of
     our own giant run (2.6e-3 We, rel; accept 1.8e-3..3.5e-3 nonrel).

Units: c = 1, We0 = 1, wpe = 5 (Case II family). 1 keV <-> v/c = 0.0626.
"""
import numpy as np
from scipy.special import wofz

WPE = 5.0
WE = 1.0
SQRT2 = np.sqrt(2.0)


def Z(zeta):
    return 1j * np.sqrt(np.pi) * wofz(zeta)


class BiMax:
    def __init__(self, n, vpar, vperp, name=""):
        self.n, self.vpar, self.vperp, self.name = n, vpar, vperp, name
        self.A = (vperp / vpar) ** 2 - 1.0

    def chi(self, w, k):
        wps2 = self.n * WPE ** 2
        zeta = (w - WE) / (SQRT2 * k * self.vpar)
        z0 = w / (SQRT2 * k * self.vpar)
        return (wps2 / w ** 2) * (z0 * Z(zeta) + self.A * (1 + zeta * Z(zeta)))


class Cold:
    def __init__(self, n):
        self.n = n

    def chi(self, w, k):
        return self.n * WPE ** 2 / (w * (WE - w))


def D(w, k, species):
    return 1 - k ** 2 / w ** 2 + sum(s.chi(w, k) for s in species)


def solve_w(k, species, w0):
    w = complex(w0)
    for _ in range(60):
        d = D(w, k, species)
        h = 1e-7 * (abs(w) + 1e-4)
        dp = (D(w + h, k, species) - d) / h
        step = d / dp
        w -= step
        if abs(step) < 1e-12:
            break
    return w


def cold_whistler_w(k):
    # solve k^2 = w^2 + wpe^2 w/(We - w) on 0<w<We by bisection
    lo, hi = 1e-6, WE - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid ** 2 + WPE ** 2 * mid / (WE - mid) < k ** 2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def branch(species, kmin=0.8, kmax=12.0, nk=700):
    ks = np.linspace(kmin, kmax, nk)
    out = []
    for k in ks:
        w = solve_w(k, species, cold_whistler_w(k) + 0j)
        out.append((k, w.real, w.imag))
    return np.array(out)


def gates():
    ok = True
    # G1: cold limit
    sp = [Cold(1.0)]
    k = 1.0
    w = solve_w(k, sp, cold_whistler_w(k) + 0j)
    err = abs(w.real - cold_whistler_w(k)) + abs(w.imag)
    print(f"GATE1 cold-limit: err={err:.2e}  ", "PASS" if err < 1e-6 else "FAIL")
    ok &= err < 1e-6

    # G2 + G3: Case II single hot bi-Max
    sp = [Cold(0.9822), BiMax(0.0178, 0.19783565, 0.24390817, "caseII")]
    br = branch(sp)
    gmax = br[:, 2].max()
    wmax = br[br[:, 2].argmax(), 1]
    pos = br[br[:, 2] > 0]
    wm = pos[:, 1].max() if len(pos) else np.nan
    A = (0.24390817 / 0.19783565) ** 2 - 1
    wm_kp = A / (1 + A)
    print(f"GATE2 KP marginal: solver w_m={wm:.3f}, theory {wm_kp:.3f}  ",
          "PASS" if abs(wm - wm_kp) < 0.03 else "FAIL")
    ok &= abs(wm - wm_kp) < 0.03
    # Measured giant-run gamma (2.6e-3) is for the LOSS-CONE distribution in
    # the dipole: subtracting small pitch angles raises the effective
    # anisotropy, so the uniform bi-Max value here should sit BELOW it.
    print(f"GATE3 gamma_max={gmax:.2e} at w={wmax:.3f} "
          f"(loss-cone giant run measured 2.6e-3; bi-Max must be below)  ",
          "PASS" if 1.2e-3 < gmax < 2.9e-3 else "FAIL")
    ok &= 1.2e-3 < gmax < 2.9e-3
    return ok


def design():
    """Flagship f0 candidates: Case II core + warm anisotropic component."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core = [Cold(0.9822), BiMax(0.0178, 0.19783565, 0.24390817, "hot core")]
    fig, ax = plt.subplots(figsize=(8, 5))
    br0 = branch(core)
    ax.plot(br0[:, 1], br0[:, 2] * 1e3, "k", lw=2, label="Case II core only")

    # warm component ~0.4 keV parallel; KP: growth at w needs
    # Tperp/Tpar > 1/(1-w), so UB out to 0.67/0.75 needs R >= 3/4.
    # Li 2019's observation-fitted warm component is R = 4 — now we see why.
    for nw, R, cc in ((0.005, 2.0, "tab:blue"), (0.005, 3.0, "tab:green"),
                      (0.005, 4.0, "tab:red"), (0.002, 4.0, "tab:purple")):
        sp = [Cold(0.9822 - nw),
              BiMax(0.0178, 0.19783565, 0.24390817),
              BiMax(nw, 0.04, 0.04 * np.sqrt(R), "warm")]
        br = branch(sp)
        ax.plot(br[:, 1], br[:, 2] * 1e3, color=cc, lw=1.6,
                label=f"+ warm n={nw}, Tperp/Tpar={R:.0f} (0.4 keV)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0.5, color="r", ls=":", lw=1)
    ax.set_xlim(0.1, 0.9)
    ax.set_xlabel(r"$\omega/\Omega_{e0}$")
    ax.set_ylabel(r"$\gamma\ (10^{-3}\,\Omega_{e0})$")
    ax.set_title("flagship f0 design: warm anisotropic component fuels the "
                 "upper band\nwithout disturbing the lower-band drive")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = "docs/figs/gap_growth_design.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # console: UB/LB peak summary over the (n_warm, R) fuel grid
    print("\nfuel grid (warm 0.4 keV): UB = w in [0.55, 0.85]")
    for nw in (0.002, 0.005, 0.010):
        for R in (2.0, 3.0, 4.0):
            sp = [Cold(0.9822 - nw), BiMax(0.0178, 0.19783565, 0.24390817),
                  BiMax(nw, 0.04, 0.04 * np.sqrt(R))]
            br = branch(sp)
            lb = br[br[:, 1] < 0.5]
            ub = br[(br[:, 1] >= 0.55) & (br[:, 1] <= 0.85)]
            wub = ub[ub[:, 2].argmax(), 1]
            print(f"  n_warm={nw:.3f} R={R:.0f}: LB peak {lb[:,2].max():.2e}"
                  f" | UB peak {ub[:,2].max():+.2e} at w={wub:.2f}")

    # single-component alternative: raise the CORE anisotropy, lower nh to
    # hold the LB drive at the discrete-regime level (~1.5-2e-3)
    print("\nsingle-component sweet-spot scan (vpar=0.198c core):")
    for R, nh in ((1.52, 0.0178), (2.0, 0.010), (2.5, 0.007), (3.0, 0.005),
                  (3.0, 0.0035)):
        sp = [Cold(1.0 - nh),
              BiMax(nh, 0.19783565, 0.19783565 * np.sqrt(R))]
        br = branch(sp)
        lb = br[br[:, 1] < 0.5]
        ub = br[(br[:, 1] >= 0.55) & (br[:, 1] <= 0.85)]
        wlb = lb[lb[:, 2].argmax(), 1]
        wub = ub[ub[:, 2].argmax(), 1]
        print(f"  R={R:.2f} nh={nh:.4f}: LB peak {lb[:,2].max():.2e} "
              f"at w={wlb:.2f} | UB peak {ub[:,2].max():+.2e} at w={wub:.2f}")


if __name__ == "__main__":
    if gates():
        design()
    else:
        print("GATES FAILED — do not trust design output")
