#!/usr/bin/env python3
"""G2.2 reference — cold-plasma whistler ray tracing in the mirror2d slab.

Geometry (matches b0_prof=3): Bx = B0(1 + a x~^2), By = -2 a B0 x~ y~,
uniform density (cold-dominated), c = 1, wpe = 1.

Quasi-longitudinal whistler dispersion as ray Hamiltonian:
    w(r, k) = We(r) * (k.bhat(r)/|k|) * |k|^2 / (|k|^2 + 1)
Rays:  dr/dt = dw/dk,  dk/dt = -dw/dr  (2D: x, y, kx, ky).

Launch quasi-parallel at the equator (on-axis and off-axis) for a set of
w/We0; report the LOCAL wave-normal angle theta(x) = angle(k, bhat) along
each ray. The PIC gate (G2.2): the WNA(x-region) map from mirror2d f2d
snapshots must reproduce these curves in the linear stage.

Usage: ray_mirror2d.py [--a=9.54e-5] [--xc=102.4] [--yc=3.2] [--wce=0.25]
                       [--Ly=6.4] [--out=docs/figs/ray_mirror2d.png]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

P = {"a": 9.5367431640625e-05, "xc": 102.4, "yc": 3.2, "wce": 0.25,
     "Ly": 6.4, "out": "docs/figs/ray_mirror2d.png"}
for arg in sys.argv[1:]:
    if arg.startswith("--") and "=" in arg:
        k, v = arg[2:].split("=", 1)
        P[k] = v if k == "out" else float(v)


def bfield(x, y):
    xt, yt = x - P["xc"], y - P["yc"]
    return P["wce"] * (1.0 + P["a"] * xt * xt), -2.0 * P["a"] * P["wce"] * xt * yt


def omega(x, y, kx, ky):
    Bx, By = bfield(x, y)
    B = np.hypot(Bx, By)
    k = np.hypot(kx, ky)
    kpar = (kx * Bx + ky * By) / B
    return B * kpar * k / (k * k + 1.0)


def rhs(t, s):
    x, y, kx, ky = s
    eps_r, eps_k = 1e-4, 1e-6
    dwdx = (omega(x + eps_r, y, kx, ky) - omega(x - eps_r, y, kx, ky)) / (2 * eps_r)
    dwdy = (omega(x, y + eps_r, kx, ky) - omega(x, y - eps_r, kx, ky)) / (2 * eps_r)
    dwdkx = (omega(x, y, kx + eps_k, ky) - omega(x, y, kx - eps_k, ky)) / (2 * eps_k)
    dwdky = (omega(x, y, kx, ky + eps_k) - omega(x, y, kx, ky - eps_k)) / (2 * eps_k)
    return [dwdkx, dwdky, -dwdx, -dwdy]


def k_of_omega(w, We):
    # parallel whistler: w = We k^2/(k^2+1)  ->  k = sqrt(w/(We-w))
    return np.sqrt(w / (We - w))


def trace(wfrac, y0_off, tmax=40000.0):
    We0 = P["wce"]
    w = wfrac * We0
    k0 = k_of_omega(w, We0)
    s0 = [P["xc"], P["yc"] + y0_off, k0, 0.0]
    hit_wall = lambda t, s: (s[0] - 2.0) * (2.0 * P["xc"] - 2.0 - s[0])
    hit_wall.terminal = True
    sol = solve_ivp(rhs, [0, tmax], s0, max_step=50.0, rtol=1e-8,
                    events=hit_wall, dense_output=False)
    x, y, kx, ky = sol.y
    Bx, By = bfield(x, y)
    B = np.hypot(Bx, By)
    cosw = (kx * Bx + ky * By) / (np.hypot(kx, ky) * B)
    th = np.degrees(np.arccos(np.clip(np.abs(cosw), 0, 1)))
    werr = np.abs(omega(x, y, kx, ky) / w - 1.0).max()
    return x, y, th, werr


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    offs = [0.0, 0.25 * P["Ly"] / 2, -0.25 * P["Ly"]]
    offs = [0.0, P["Ly"] / 8, P["Ly"] / 4]
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
    for wfrac, ls in ((0.3, "-"), (0.5, "--")):
        for y0, cc in zip(offs, cols):
            x, y, th, werr = trace(wfrac, y0)
            lam = x - P["xc"]
            axes[0].plot(lam, th, ls=ls, color=cc, lw=1.4,
                         label=f"$\\omega$={wfrac}$\\Omega_e$, $\\tilde y_0$={y0:.1f}")
            axes[1].plot(lam, y - P["yc"], ls=ls, color=cc, lw=1.2)
            print(f"w={wfrac:.2f} y0={y0:+.2f}: theta at |x~|=50: "
                  f"{np.interp(50.0, np.abs(lam), th):5.2f} deg  "
                  f"(w conservation err {werr:.1e})")
    axes[0].set_xlabel(r"$\tilde x$ (field-aligned, $c/\omega_{pe}$)")
    axes[0].set_ylabel(r"local WNA $\theta$ (deg)")
    axes[0].set_title("whistler WNA vs latitude — mirror2d ray tracing")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(r"$\tilde x$"); axes[1].set_ylabel(r"$\tilde y$ (ray path)")
    axes[1].set_title("ray paths (field-line following)")
    fig.tight_layout()
    fig.savefig(P["out"], dpi=140)
    print("wrote", P["out"])


if __name__ == "__main__":
    main()
