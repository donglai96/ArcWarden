#!/usr/bin/env python3
"""V4R domain layout — dipole2d at L0 = 4400 (l_re = 2540, x5.2).

Left: meridional plane with the TRUE-dipole-shaped field lines
(L = r^3/x^2 contours), |B| map, marker shell, box, absorber frame,
particle walls (high-latitude ends only), probe stations, latitude rays.
Right: mirror profile along the central line vs the real 3D dipole and
the retired circle geometry (the exact cos^3 identity annotated).
Numbers from the warden2d pre-flight of decks/warden2d_v4r.ini.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- V4R pre-flight numbers ---
L0, B0eq = 4400.0, 0.2
X0, X1, Z0, Z1 = 1087.0, 4540.0, -2300.0, 2300.0
ND = 240 * 0.25                      # absorber frame, 60 c/wpe
dL, edge = 40.0, 8.0
lam_w, lam_run, lam_ub = np.radians(50), np.radians(40), np.radians(20)
C = 0.5 * B0eq * L0**3

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.8),
                              gridspec_kw={"width_ratios": [1.5, 1]})

x = np.linspace(700, 4700, 900)
z = np.linspace(-2500, 2500, 1000)
X, Z = np.meshgrid(x, z)
R2 = X**2 + Z**2
Bx = -6 * C * X**4 * Z / R2**4
Bz = C * X**3 * (2 * X**2 - 4 * Z**2) / R2**4
Babs = np.sqrt(Bx**2 + Bz**2)
Lsh = R2 * np.sqrt(R2) / X**2

pc = ax.pcolormesh(X, Z, np.log10(Babs / B0eq), cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, shading="auto", rasterized=True)
plt.colorbar(pc, ax=ax, label=r"$\log_{10}|B_0|/B_{0,\rm eq}(L_0)$",
             fraction=0.045, pad=0.01)

for L in [2400, 3000, 3600, 4400, 5200, 6200]:
    lw, c = (2.4, "k") if L == L0 else (0.7, "0.3")
    ax.contour(X, Z, Lsh, levels=[L], colors=c, linewidths=lw)

# particle shell (flat 40 + Gaussian edge 8 in L)
a = np.abs(Lsh - L0) - dL / 2
prof = np.where(a <= 0, 1.0, np.exp(-np.clip(a, 0, None)**2 / (2 * edge**2)))
ax.contourf(X, Z, prof, levels=[0.05, 1.01], colors=["#ffd54f"], alpha=0.6)
ax.annotate("PARTICLES: 2.31e8 full-f markers\nshell $L_0\\pm20$, Gauss edge 8\n"
            "(cold fluid + waves fill the whole box)",
            xy=(4100, 1400), xytext=(2450, 2050), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.15"))

# box + absorber frame
ax.add_patch(Rectangle((X0, Z0), X1 - X0, Z1 - Z0, fc="none", ec="k", lw=2))
for r in [Rectangle((X0, Z0), X1 - X0, ND, hatch="////"),
          Rectangle((X0, Z1 - ND), X1 - X0, ND, hatch="////"),
          Rectangle((X0, Z0), ND, Z1 - Z0, hatch="////"),
          Rectangle((X1 - ND, Z0), ND, Z1 - Z0, hatch="////")]:
    r.set(fc="none", ec="0.45", lw=0.5)
    ax.add_patch(r)
ax.text(X0 + 80, Z0 + 90, "wave absorber frame (60 c/$\\omega_{pe}$)",
        fontsize=8, color="0.35")

# particle walls: high-|lambda| ends only
ax.plot([X0 + ND, X0 + ND], [Z0 + ND, Z1 - ND], color="crimson", lw=3)
ax.plot([X0 + ND, X1 - ND], [Z1 - ND, Z1 - ND], color="crimson", lw=3)
ax.plot([X0 + ND, X1 - ND], [Z0 + ND, Z0 + ND], color="crimson", lw=3)
ax.annotate("particle walls ($u_\\parallel$-flip)\nhigh-$\\lambda$ ends only",
            xy=(X0 + ND, 1200), xytext=(1250, 1850), color="crimson",
            fontsize=9, arrowprops=dict(arrowstyle="->", color="crimson"))
ax.annotate("NO outer wall\n(gc cannot cross $L$)", xy=(X1 - ND, 300),
            xytext=(3550, 900), fontsize=9, color="0.2",
            arrowprops=dict(arrowstyle="->", color="0.2"))

# latitude rays + probe stations on the central line
for lam, c, lab in [(lam_ub, "seagreen", "$\\pm20°$ UB domain"),
                    (lam_run, "royalblue", "$\\pm40°$ runway end (s=3335)"),
                    (lam_w, "purple", "$\\pm50°$ wall, $B/B_{eq}$=6.26")]:
    for sgn in (1, -1):
        ax.plot([0, 5000 * np.cos(lam)], [0, sgn * 5000 * np.sin(lam)],
                ls=":", color=c, lw=1.1)
    ax.text(5000 * np.cos(lam) * 0.42, 5000 * np.sin(lam) * 0.46, lab,
            color=c, fontsize=8.5, rotation=np.degrees(lam), ha="center")
for lam in np.radians([0, 5, -5, 10, -10, 20, -20]):
    c2 = np.cos(lam)**2
    ax.plot(L0 * c2 * np.cos(lam), L0 * c2 * np.sin(lam), "wo", mec="k", ms=6)
ax.plot([], [], "wo", mec="k", label="probe stations (0, ±5°, ±10°, ±20°)")
ax.legend(loc="lower right", fontsize=8)

ax.set_xlim(700, 4750); ax.set_ylim(-2500, 2500); ax.set_aspect("equal")
ax.set_xlabel(r"$x$ [$c/\omega_{pe}$] (radial)")
ax.set_ylabel(r"$z$ [$c/\omega_{pe}$] (dipole axis)")
ax.set_title("V4R domain — dipole2d, $L_0$=4400, box 13811×18400 cells, 23.6 GB\n"
             r"true-dipole line shapes $r = L\cos^2\lambda$, $B_{eq}\propto L^{-3}$, "
             r"$l_{re}$ = 2540 ($\times$5.2, least-compressed 2D chorus run)")

# ---- right: mirror profiles ----
lam = np.linspace(0, np.radians(55), 400)
Bd2 = np.sqrt(1 + 4 * np.tan(lam)**2) / np.cos(lam)**2
B3d = np.sqrt(1 + 3 * np.sin(lam)**2) / np.cos(lam)**6
Bcirc = 1 / np.cos(lam)**2
ax2.semilogy(np.degrees(lam), B3d, "r-.", lw=1.8, label="real 3D dipole")
ax2.semilogy(np.degrees(lam), Bd2, "g-", lw=2.2,
             label=r"V4R dipole2d $= {\rm real}\times\cos^3\lambda$")
ax2.semilogy(np.degrees(lam), Bcirc, "0.6", lw=1.4, ls="--",
             label="retired circle (V4a)")
for lamv, cc in [(20, "seagreen"), (40, "royalblue"), (50, "purple")]:
    ax2.axvline(lamv, ls=":", color=cc, lw=1.1)
ax2.axhline(6.26, ls=":", color="purple", lw=1)
ax2.text(50.7, 6.6, "6.26", color="purple", fontsize=9)
ax2.set_xlim(0, 55); ax2.set_ylim(1, 40)
ax2.set_xlabel(r"latitude $\lambda$ [deg]")
ax2.set_ylabel(r"$B/B_{\rm eq}$")
ax2.set_title("mirror profile: the planar-2D gap is exactly $\\cos^3\\lambda$\n"
              "(azimuthal flux convergence — irreducible in meridional 2D)")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(alpha=0.3, which="both")

fig.tight_layout()
out = "docs/figs/v4r_domain.png"
fig.savefig(out, dpi=150)
print("wrote", out)
