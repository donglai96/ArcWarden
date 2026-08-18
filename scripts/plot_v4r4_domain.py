#!/usr/bin/env python3
"""V4R4 compact-box layout: inner cut x=2500 (wall lambda ~30.5 deg),
active L band [3300,4700], vs the old V4R3 rectangle (dashed)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

L0, B0eq = 4000.0, 0.2
X0, X1, Z0, Z1 = 2500.0, 4140.0, -1657.0, 1657.0     # V4R4 pre-flight
OX0, OX1, OZ0, OZ1 = 1087.0, 4540.0, -2300.0, 2300.0  # V4R3 (old)
ND = 240 * 0.25
dL, edge = 40.0, 8.0
Lbmin, Lbmax = 3300.0, 4700.0
C = 0.5 * B0eq * L0**3

fig, ax = plt.subplots(figsize=(10.5, 8))
x = np.linspace(900, 4700, 900)
z = np.linspace(-2450, 2450, 1100)
X, Z = np.meshgrid(x, z)
R2 = X**2 + Z**2
Bx = -6 * C * X**4 * Z / R2**4
Bz = C * X**3 * (2 * X**2 - 4 * Z**2) / R2**4
Babs = np.sqrt(Bx**2 + Bz**2)
Lsh = R2 * np.sqrt(R2) / X**2

pc = ax.pcolormesh(X, Z, np.log10(Babs / B0eq), cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, shading="auto", rasterized=True)
plt.colorbar(pc, ax=ax, label=r"$\log_{10}|B_0|/B_{0,\rm eq}$", fraction=0.04)

for L in [2600, 3300, 4000, 4700, 5600]:
    lw, cc = (2.4, "k") if L == L0 else (0.7, "0.3")
    ax.contour(X, Z, Lsh, levels=[L], colors=cc, linewidths=lw)

# active band shading + particle shell
band = (Lsh >= Lbmin) & (Lsh <= Lbmax)
ax.contourf(X, Z, band.astype(float), levels=[0.5, 1.5],
            colors=["#a8e6a3"], alpha=0.35)
a = np.abs(Lsh - L0) - dL / 2
prof = np.where(a <= 0, 1.0, np.exp(-np.clip(a, 0, None)**2 / (2 * edge**2)))
ax.contourf(X, Z, prof, levels=[0.05, 1.01], colors=["#ffd54f"], alpha=0.75)

# new box + absorbers + walls
ax.add_patch(Rectangle((X0, Z0), X1 - X0, Z1 - Z0, fc="none", ec="k", lw=2.2))
for r in [Rectangle((X0, Z0), X1 - X0, ND, hatch="////"),
          Rectangle((X0, Z1 - ND), X1 - X0, ND, hatch="////"),
          Rectangle((X0, Z0), ND, Z1 - Z0, hatch="////"),
          Rectangle((X1 - ND, Z0), ND, Z1 - Z0, hatch="////")]:
    r.set(fc="none", ec="0.45", lw=0.5); ax.add_patch(r)
ax.plot([X0 + ND, X0 + ND], [Z0 + ND, Z1 - ND], color="crimson", lw=3)
ax.plot([X0 + ND, X1 - ND], [Z1 - ND, Z1 - ND], color="crimson", lw=3)
ax.plot([X0 + ND, X1 - ND], [Z0 + ND, Z0 + ND], color="crimson", lw=3)
# old box for comparison
ax.add_patch(Rectangle((OX0, OZ0), OX1 - OX0, OZ1 - OZ0, fc="none",
                       ec="0.4", lw=1.4, ls="--"))
ax.text(OX0 + 40, OZ1 - 150, "old V4R3 box (2.12e8 cells)", color="0.35",
        fontsize=9)
ax.text(X0 + 60, Z1 - 260, "V4R4 box (8.7e7 cells)\nfields 5.2 GB (was 15.3)",
        fontsize=9, weight="bold")
ax.annotate("particle wall: lines exit at $\\lambda \\approx 30.5°$\n"
            "(mirror ratio 2.08)", xy=(X0 + ND, 1100), xytext=(1150, 1900),
            color="crimson", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="crimson"))
ax.text(3550, -1350, "green = active field band\nL ∈ [3300, 4700]\n"
        "(waves absorbed at band edge)", fontsize=9, color="darkgreen")
ax.text(3350, 550, "shell: 2.3e8 markers", fontsize=9)
for lam in np.radians([0, 5, -5, 10, -10, 20, -20]):
    c2 = np.cos(lam)**2
    ax.plot(L0 * c2 * np.cos(lam), L0 * c2 * np.sin(lam), "wo", mec="k", ms=5)

ax.set_xlim(900, 4750); ax.set_ylim(-2450, 2450); ax.set_aspect("equal")
ax.set_xlabel(r"$x$ [$c/\omega_{pe}$]"); ax.set_ylabel(r"$z$ [$c/\omega_{pe}$]")
ax.set_title("V4R4 compact layout — inner cut $x_{min}$=2500, active band, "
             "true-dipole lines ($L_0$=4000, $l_{re}$=2309)")
fig.tight_layout()
fig.savefig("docs/figs/v4r4_domain.png", dpi=150)
print("wrote docs/figs/v4r4_domain.png")
