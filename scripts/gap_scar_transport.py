#!/usr/bin/env python3
"""G0.1 — Scar transport: where does an off-equator Landau plateau wall the
equatorial spectrum? (The calculation missing from the 0.5 fce literature.)

Physics: a lower-band wave at (equatorial-normalized) frequency w~ becomes
oblique by latitude lam_w and Landau-carves a scar at the LOCAL parallel
phase velocity, v_par = Vp_local(w~, lam_w), across a v_perp column.
Bounce transport to the equator conserves (E, mu):
    v_perp_eq^2 = v_perp^2 / b,   v_par_eq^2 = v_par^2 + v_perp^2 (1 - 1/b),
with b(lam) = B/Beq = sqrt(1+3 sin^2 lam)/cos^6 lam (dipole).
The equatorial wall sits at frequencies w~' whose cyclotron resonance speed
|V_R(w~')| = Vp_eq(w~')(1/w~' - 1) falls on the transported scar.

Everything nonrelativistic (v <= 0.35c, gamma <= 1.06), cold dispersion
chi ~ 1 kept exactly, wpe constant along the line (Lambda = 1).

Outputs: docs/figs/gap_scar_transport.png + a console table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPE = 5.0          # wpe / We0 (Case II family)


def bdip(lam):
    s, c = np.sin(lam), np.cos(lam)
    return np.sqrt(1 + 3 * s * s) / c ** 6


def vp_local(wt, b):
    """Parallel phase velocity /c at local field strength b (We0 units)."""
    xi2 = wt * (b - wt) / WPE ** 2
    xi = np.sqrt(xi2)
    chi = 1.0 / np.sqrt(1.0 + xi2)
    return chi * xi


def vr_eq(wt):
    """|cyclotron resonance speed| /c at the equator for frequency wt."""
    return vp_local(wt, 1.0) * (1.0 / wt - 1.0)


def wall_freq(v):
    """Invert |V_R(w')| = v on the equatorial branch (monotone decreasing)."""
    lo, hi = 1e-3, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if vr_eq(mid) > v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scar_image(wt_w, lam_w_deg, vperp):
    """Equatorial (v_par_eq, v_perp_eq) image of the scar written by wave
    wt_w at latitude lam_w across a v_perp column."""
    b = bdip(np.radians(lam_w_deg))
    vpar_w = vp_local(wt_w, b)
    vpar_eq = np.sqrt(vpar_w ** 2 + vperp ** 2 * (1 - 1 / b))
    return vpar_w, vpar_eq, vperp / np.sqrt(b)


def main():
    vperp = np.linspace(0, 0.24, 49)          # up to one Case II vth_perp
    lams = np.linspace(0.01, 27, 120)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) walled frequency vs writing latitude, ridge (vperp=0) and band
    ax = axes[0]
    for wt_w, cc in ((0.50, "tab:red"), (0.40, "tab:orange"),
                     (0.30, "tab:green")):
        ridge, lo_edge = [], []
        for ld in lams:
            vpar_w, vpar_eq, _ = scar_image(wt_w, ld, vperp)
            ridge.append(wall_freq(vpar_eq[0]))
            lo_edge.append(wall_freq(vpar_eq[-1]))
        ridge, lo_edge = np.array(ridge), np.array(lo_edge)
        ax.plot(lams, ridge, color=cc, lw=2,
                label=rf"$\tilde\omega_w={wt_w}$ ridge ($v_\perp=0$)")
        ax.fill_between(lams, lo_edge, ridge, color=cc, alpha=0.18)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel(r"writing latitude $\lambda_w$ (deg)")
    ax.set_ylabel(r"walled equatorial frequency $\tilde\omega'$")
    ax.set_title("(a) where the transported scar walls the spectrum\n"
                 "(shading: $v_\\perp$ up to one $v_{th\\perp}$)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0.3, 0.6)

    # (b) scar dispersion (wall dilution) vs latitude
    ax = axes[1]
    for wt_w, cc in ((0.50, "tab:red"), (0.40, "tab:orange"),
                     (0.30, "tab:green")):
        spread = []
        for ld in lams:
            _, vpar_eq, _ = scar_image(wt_w, ld, vperp)
            spread.append(vpar_eq[-1] - vpar_eq[0])
        ax.plot(lams, np.array(spread) / vp_local(0.5, 1.0), color=cc, lw=2,
                label=rf"$\tilde\omega_w={wt_w}$")
    ax.set_xlabel(r"writing latitude $\lambda_w$ (deg)")
    ax.set_ylabel(r"scar spread $\Delta v_{\parallel,eq}\,/\,V_p^{eq}(0.5)$")
    ax.set_title("(b) wall dilution: high-latitude scars smear\n"
                 "(sharp wall needs low-latitude writing)")
    ax.legend(fontsize=8)

    # (c) the assembled wall: near-equator anchor + graded shoulder
    ax = axes[2]
    # weight scar strength ~ 1/(spread + narrow ridge width); writing
    # efficiency vs latitude left uniform on purpose (geometry only)
    wgrid = np.linspace(0.30, 0.62, 300)
    erosion = np.zeros_like(wgrid)
    for wt_w in (0.30, 0.35, 0.40, 0.45, 0.50):
        for ld in np.linspace(2, 25, 40):
            _, vpar_eq, _ = scar_image(wt_w, ld, vperp)
            w_hi = wall_freq(vpar_eq[0])
            w_lo = wall_freq(vpar_eq[-1])
            width = max(w_hi - w_lo, 1e-3)
            sel = (wgrid >= w_lo) & (wgrid <= w_hi)
            erosion[sel] += 1.0 / width / 40 / 5
    erosion /= erosion.max()
    ax.plot(wgrid, erosion, "k", lw=2)
    ax.axvline(0.5, color="r", ls=":", lw=1)
    ax.annotate("hard anchor:\n$V_p$ max + $|V_R|=V_p$\ndegeneracy at 0.5",
                xy=(0.505, 0.95), fontsize=8, color="r")
    ax.set_xlabel(r"equatorial frequency $\tilde\omega'$")
    ax.set_ylabel("relative erosion strength (geometry only)")
    ax.set_title("(c) assembled wall profile\n"
                 "(uniform writing in $\\lambda_w\\in[2,25]$, LB 0.3-0.5)")

    fig.tight_layout()
    out = "docs/figs/gap_scar_transport.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # console table
    print(f"\nWPE/We0 = {WPE};  Vp_eq(0.5) = {vp_local(0.5,1.0):.4f} c ; "
          f"|VR_eq(0.5)| = {vr_eq(0.5):.4f} c  (degeneracy check)")
    print("\nlam_w   b      wave0.5: Vp_loc  ridge wall  vperp-edge wall  spread/Vp(0.5)")
    for ld in (2, 5, 10, 15, 20, 25):
        b = bdip(np.radians(ld))
        vpar_w, vpar_eq, _ = scar_image(0.5, ld, vperp)
        print(f"{ld:5.0f}  {b:5.3f}   {vpar_w:.4f}c     "
              f"{wall_freq(vpar_eq[0]):.3f}       {wall_freq(vpar_eq[-1]):.3f}"
              f"            {(vpar_eq[-1]-vpar_eq[0])/vp_local(0.5,1.0):.3f}")


if __name__ == "__main__":
    main()
