#!/usr/bin/env python3
"""Line-dipole background field of pic2d (PLAN_2D_REBORN D1) + W10 layout.

Left: field lines (L = const circles through the origin), |B| = M/r^2
colormap, the W10 box, absorber frame, particle walls (high-latitude ends
only), and the delta-f marker shell. Right: mirror profile along the
central line B(s)/Beq = sec^2(s/L) vs the 1D parabolic family (l_re = L)
and the real 3D dipole at equal latitude.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- W10 deck numbers (decks/warden2d_w10.ini, measured pre-flight) ---
L0, B0eq = 1330.5, 0.2
M = B0eq * L0**2
X0, X1, Z0, Z1 = 457.0, 1470.0, -765.0, 765.0     # box
ND = 240 * 0.25                                    # absorber frame, 60 c/wpe
dL, edge = 40.0, 8.0                               # shell flat width + Gauss edge
lam_w, lam_run, lam_ub = np.radians(50), np.radians(40), np.radians(20)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.4),
                              gridspec_kw={"width_ratios": [1.35, 1]})

# ---------------- left: meridional plane ----------------
x = np.linspace(20, 1650, 900)
z = np.linspace(-900, 900, 1000)
X, Z = np.meshgrid(x, z)
R2 = X**2 + Z**2
Babs = M / R2
Lsh = R2 / X

pc = ax.pcolormesh(X, Z, np.log10(Babs / B0eq), cmap="RdBu_r",
                   vmin=-1.2, vmax=1.2, shading="auto", rasterized=True)
plt.colorbar(pc, ax=ax, label=r"$\log_{10}\,|B_0|/B_{0,\rm eq}(L_0)$",
             fraction=0.04, pad=0.01)

for L in [700, 900, 1100, 1250, 1330.5, 1450, 1650]:
    lw, c = (2.2, "k") if L == L0 else (0.7, "0.25")
    ax.contour(X, Z, Lsh, levels=[L], colors=c, linewidths=lw)

# delta-f marker shell: flat top + Gaussian edges in L
a = np.abs(Lsh - L0) - dL / 2
prof = np.where(a <= 0, 1.0, np.exp(-np.clip(a, 0, None)**2 / (2 * edge**2)))
ax.contourf(X, Z, prof, levels=[0.05, 1.01], colors=["#ffd54f"], alpha=0.55)

# box + absorber frame
ax.add_patch(Rectangle((X0, Z0), X1 - X0, Z1 - Z0, fc="none", ec="k", lw=1.8))
for r in [Rectangle((X0, Z0), X1 - X0, ND, hatch="////"),
          Rectangle((X0, Z1 - ND), X1 - X0, ND, hatch="////"),
          Rectangle((X0, Z0), ND, Z1 - Z0, hatch="////"),
          Rectangle((X1 - ND, Z0), ND, Z1 - Z0, hatch="////")]:
    r.set(fc="none", ec="0.4", lw=0.5)
    ax.add_patch(r)

# particle walls: high-latitude ends only (inner-x + top/bottom, mask edge)
ax.plot([X0 + ND, X0 + ND], [Z0 + ND, Z1 - ND], color="crimson", lw=3)
ax.plot([X0 + ND, X1 - ND], [Z1 - ND, Z1 - ND], color="crimson", lw=3)
ax.plot([X0 + ND, X1 - ND], [Z0 + ND, Z0 + ND], color="crimson", lw=3)
ax.annotate("particle walls (u$_\\parallel$-flip)\nhigh-$\\lambda$ ends only",
            xy=(X0 + ND, 400), xytext=(120, 620), color="crimson", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="crimson"))
ax.annotate("NO wall on outer edge\n(gc cannot cross $L$)",
            xy=(X1 - ND, 100), xytext=(1120, 430), fontsize=9, color="0.2",
            arrowprops=dict(arrowstyle="->", color="0.2"))

# latitude rays
for lam, c, lab in [(lam_ub, "seagreen", r"$\pm20°$ (UB domain)"),
                    (lam_run, "royalblue", r"$\pm40°$ (runway end)"),
                    (lam_w, "purple", r"$\pm50°$ (wall, $B/B_{eq}=2.42$)")]:
    for s in (1, -1):
        ax.plot([0, 1700 * np.cos(lam)], [0, s * 1700 * np.sin(lam)],
                ls=":", color=c, lw=1.2)
    ax.text(1700 * np.cos(lam) * 0.55, 1700 * np.sin(lam) * 0.58, lab,
            color=c, fontsize=9, rotation=np.degrees(lam), ha="center")

ax.plot(0, 0, "ko", ms=8)
ax.text(30, -60, "dipole\norigin", fontsize=9)
ax.plot(L0, 0, "k*", ms=12)
ax.text(L0 - 60, 40, "equator of $L_0$", fontsize=9)

ax.set_xlim(0, 1700); ax.set_ylim(-900, 900); ax.set_aspect("equal")
ax.set_xlabel(r"$x$  [$c/\omega_{pe}$] (radial)")
ax.set_ylabel(r"$z$  [$c/\omega_{pe}$] (along dipole axis)")
ax.set_title("pic2d line-dipole background — W10 layout\n"
             r"$\psi = Mx/r^2$: field lines are circles, $|B|=M/r^2$, "
             r"$\nabla\!\cdot\!B_0=0$ exact; shell = $\delta f$ markers")

# ---------------- right: mirror profile along the line ----------------
lam = np.linspace(0, np.radians(60), 400)
s_over_L = lam                                    # s = L*lambda exactly
B2d = 1 / np.cos(lam)**2                          # sec^2
Bpar = 1 + s_over_L**2                            # 1D parabolic, l_re = L
B3d = np.sqrt(1 + 3 * np.sin(lam)**2) / np.cos(lam)**6

ax2.plot(np.degrees(lam), B2d, "k-", lw=2.2,
         label=r"pic2d line dipole: $\sec^2(s/L)$")
ax2.plot(np.degrees(lam), Bpar, "b--", lw=1.6,
         label=r"1D code parabolic $1+(s/l_{re})^2$,  $l_{re}=L$")
ax2.plot(np.degrees(lam), B3d, "r-.", lw=1.6,
         label=r"real 3D dipole $\sqrt{1+3\sin^2\lambda}/\cos^6\lambda$")
for lamv, c in [(20, "seagreen"), (40, "royalblue"), (50, "purple")]:
    ax2.axvline(lamv, ls=":", color=c, lw=1.2)
ax2.axhline(2.42, ls=":", color="purple", lw=1)
ax2.text(50.6, 2.47, "2.42", color="purple", fontsize=9)
ax2.set_xlim(0, 60); ax2.set_ylim(1, 8)
ax2.set_xlabel(r"latitude $\lambda$ [deg]   ($s = L\lambda$ exactly)")
ax2.set_ylabel(r"$B(s)/B_{\rm eq}$")
ax2.set_title("mirror profile along the central field line")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(alpha=0.3)

fig.tight_layout()
out = "docs/figs/pic2d_background_linedipole.png"
fig.savefig(out, dpi=160)
print("wrote", out)
