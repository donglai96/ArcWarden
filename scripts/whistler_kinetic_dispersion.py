#!/usr/bin/env python3
"""Parallel-propagating R-mode (whistler) kinetic dispersion: cold electrons +
bi-Maxwellian hot electrons. Units: Omega_e0 = c = 1 (matches chirp1d).

    D(w,k) = 1 - k^2/w^2 + (wpc^2/w^2) * w/(1-w)
             + (wph^2/w^2) * [ zeta0 Z(zeta) + A (1 + zeta Z(zeta)) ]
    zeta0 = w/(sqrt(2) k a_par),  zeta = (w-1)/(sqrt(2) k a_par),
    A = Tperp/Tpar - 1.

Newton root-finding in complex w, warm-started from the cold whistler root.

Usage: whistler_kinetic_dispersion.py [wpe] [nh] [uth_para] [uth_perp]
"""
import sys
import numpy as np
from scipy.special import wofz


def Z(zeta):
    """Plasma dispersion function Z(zeta) = i sqrt(pi) w(zeta)."""
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def cold_root(k, wpe):
    """Cold whistler branch root of w^2 + w*wpe^2/(1-w) = k^2 on (0,1)."""
    lo, hi = 1e-9, 1 - 1e-9
    f = lambda w: w * w + w * wpe * wpe / (1 - w) - k * k
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def D(w, k, wpc2, wph2, apar, A):
    s2ka = np.sqrt(2.0) * k * apar
    zeta0 = w / s2ka
    zeta = (w - 1.0) / s2ka
    Zz = Z(zeta)
    hot = zeta0 * Zz + A * (1.0 + zeta * Zz)
    return 1.0 - k * k / w**2 + (wpc2 / w**2) * w / (1.0 - w) + (wph2 / w**2) * hot


def solve_k(k, wpe, nh, apar, A, w0=None):
    wpc2 = wpe * wpe
    wph2 = nh * wpc2
    w = complex(w0 if w0 is not None else cold_root(k, wpe), 1e-4)
    for _ in range(100):
        h = 1e-8 * (abs(w) + 1e-8)
        dD = (D(w + h, k, wpc2, wph2, apar, A) - D(w - h, k, wpc2, wph2, apar, A)) / (2 * h)
        step = D(w, k, wpc2, wph2, apar, A) / dD
        w = w - step
        if abs(step) < 1e-13:
            break
    return w


def main():
    wpe = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    nh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    upar = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    uper = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
    A = (uper / upar) ** 2 - 1.0

    ks = np.linspace(0.3, 8.0, 400)
    best = (0.0, 0.0, 0.0)
    print(f"# wpe={wpe} nh={nh} uth_para={upar} uth_perp={uper} A={A:.3f}")
    print("# k    w_r      gamma")
    prev = None
    rows = []
    for k in ks:
        w = solve_k(k, wpe, nh, upar, A, w0=prev)
        prev = w.real
        rows.append((k, w.real, w.imag))
        if w.imag > best[2]:
            best = (k, w.real, w.imag)
    for k, wr, gi in rows[:: max(1, len(rows) // 60)]:
        print(f"{k:7.4f} {wr:9.6f} {gi:+.6e}")
    print(f"# max growth: k={best[0]:.4f}  w_r={best[1]:.6f}  gamma={best[2]:.6e}")
    print(f"# marginal frequency A/(A+1) = {A/(A+1):.4f}")


if __name__ == "__main__":
    main()
