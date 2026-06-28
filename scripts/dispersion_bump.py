#!/usr/bin/env python3
"""omega-k spectrum of the electrostatic field Ex(x,t); verify v_phase ~ v_beam.

Reads `field_xt.csv` written by run_deck (needs [time] field_history_every > 0):
a dense E(x,t) history (one row per time sample = Ex averaged over y), with a
`# dx=.. dt=..` header. Takes the 2D FFT to get power in the (k, omega) plane,
finds the dominant mode, and reports the phase velocity v_phi = omega/k.

For a bump-on-tail instability the unstable Langmuir wave is resonant with the
beam, so v_phi should be close to the beam drift v_b (default 1.0).

Usage:
    python3 scripts/dispersion_bump.py [field_xt.csv | folder] [--vb 1.0] [--vth 0.2]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_field_xt(path):
    dx = dt = None
    with open(path) as f:
        header = f.readline()
    for tok in header.replace("#", " ").split():
        if tok.startswith("dx="):
            dx = float(tok[3:])
        elif tok.startswith("dt="):
            dt = float(tok[3:])
    if dx is None or dt is None:
        raise SystemExit(f"{path}: missing '# dx=.. dt=..' header")
    E = np.loadtxt(path, delimiter=",", comments="#")
    return E, dx, dt


def compute_spectrum(E, dx, dt):
    """E(x,t) -> (k>0, omega>0) power quadrant. Returns (kk, om, Z) with
    Z[i,j] ~ |E(k=kk[j], omega=om[i])|², physical omega (>0 for +x propagation)."""
    nt, nx = E.shape
    E = E - E.mean(axis=0, keepdims=True)          # drop the static part per cell
    F = np.fft.fftshift(np.fft.fft2(E * np.hanning(nt)[:, None]))
    P = np.abs(F) ** 2
    k = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=dx))
    w = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nt, d=dt))
    # a wave ~exp(i(kx-wt)) lands at (k>0, w<0); map to physical omega = -w
    ik0 = int(np.searchsorted(k, 0.0))
    iw0 = int(np.searchsorted(w, 0.0))
    kk = k[ik0:]
    om = -w[:iw0][::-1]
    Z = P[:iw0, ik0:][::-1, :]
    return kk, om, Z


def find_peak(kk, om, Z, kband=(0.2, 4.0), wband=(0.3, 3.0)):
    """Dominant (k, omega) in a physical band -> (k_pk, omega_pk)."""
    kb = (kk >= kband[0]) & (kk <= kband[1])
    wb = (om >= wband[0]) & (om <= wband[1])
    Zb = Z[np.ix_(wb, kb)]
    iw, ik = np.unravel_index(int(np.argmax(Zb)), Zb.shape)
    return kk[kb][ik], om[wb][iw]


def load_spectrum(path):
    if os.path.isdir(path):
        path = os.path.join(path, "field_xt.csv")
    E, dx, dt = read_field_xt(path)
    return compute_spectrum(E, dx, dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="build/bump_field/field_xt.csv",
                    help="field_xt.csv or the folder containing it")
    ap.add_argument("--vb", type=float, default=1.0, help="beam velocity to compare")
    ap.add_argument("--vth", type=float, default=0.2, help="bulk thermal (Bohm-Gross curve)")
    ap.add_argument("--wpe", type=float, default=1.0, help="plasma frequency")
    args = ap.parse_args()

    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "field_xt.csv")
    if not os.path.exists(path):
        sys.stderr.write(f"error: {path} not found.\n"
                         "Run a deck with [time] field_history_every > 0 first.\n")
        return 1

    E, dx, dt = read_field_xt(path)
    nt, nx = E.shape
    print(f"loaded E(x,t): {nt} time samples x {nx} cells  (dx={dx:.4g}, dt={dt:.4g})")

    kk, om, Z = compute_spectrum(E, dx, dt)
    k_pk, w_pk = find_peak(kk, om, Z)
    vph = w_pk / k_pk
    bohm_gross = np.sqrt(args.wpe ** 2 + 3.0 * (k_pk * args.vth) ** 2)

    print("\n--- dominant electrostatic mode ---")
    print(f"  k        = {k_pk:.4f}   (lambda = {2*np.pi/k_pk:.3f})")
    print(f"  omega    = {w_pk:.4f}   (~ omega_pe = {args.wpe:.3f})")
    print(f"  v_phase  = omega/k = {vph:.4f}")
    print(f"  v_beam   = {args.vb:.4f}")
    rel = abs(vph - args.vb) / args.vb
    print(f"  |v_phase - v_beam| / v_beam = {100*rel:.1f}%   "
          f"-> {'PASS' if rel < 0.20 else 'CHECK'} (resonant with the beam)")
    print(f"  Bohm-Gross omega at this k  = {bohm_gross:.4f} "
          f"(cold/warm Langmuir reference)")

    # ---- plot the (k, omega) power spectrum with reference lines ----
    fig, ax = plt.subplots(figsize=(8, 6))
    pm = ax.pcolormesh(kk, om, np.log10(Z + Z.max() * 1e-6), shading="auto",
                       cmap="magma")
    fig.colorbar(pm, ax=ax, label=r"$\log_{10}|E(k,\omega)|^2$")
    kline = np.linspace(0.0, kk.max(), 200)
    ax.plot(kline, args.vb * kline, "c--", lw=1.4, label=rf"beam $\omega=v_b k$ ($v_b$={args.vb})")
    ax.plot(kline, np.sqrt(args.wpe**2 + 3.0 * (kline * args.vth)**2), "w:", lw=1.2,
            label=r"Bohm-Gross $\omega=\sqrt{\omega_{pe}^2+3k^2v_{th}^2}$")
    ax.plot([k_pk], [w_pk], "o", mfc="none", mec="lime", mew=1.8, ms=12,
            label=rf"peak: $v_\phi$={vph:.2f}")
    ax.set_xlim(0, min(4.0, kk.max()))
    ax.set_ylim(0, min(3.0, om.max()))
    ax.set_xlabel("k"); ax.set_ylabel(r"$\omega$")
    ax.set_title(r"Bump-on-tail $\omega$–k spectrum of $E_x$ "
                 rf"($v_\phi$={vph:.2f} vs $v_b$={args.vb})")
    ax.legend(loc="upper left", fontsize=9)
    out = os.path.join(os.path.dirname(path) or ".", "dispersion.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
