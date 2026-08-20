#!/usr/bin/env python3
"""PRE-RUN CARD (user ritual, 2026-08-19): before EVERY warden2d run, produce
one figure that answers — what is the initial distribution, where is the cold
fluid, where is the hot shell, what are grid/Debye/gyro scales, where are the
probes, what are the boundary conditions. Requires a --nsteps=1 dump first
(real loaded density, not the design).

Usage: prerun_card.py <outdir> <deck.ini> [out.png]
"""
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir, deck_path = sys.argv[1], sys.argv[2]
png = sys.argv[3] if len(sys.argv) > 3 else f"{outdir}/prerun_card.png"

# ---- parse meta + deck ----------------------------------------------------
meta = open(f"{outdir}/meta.txt").read()
m = re.search(r"box ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+) "
              r"(\d+) (\d+) ([-\d.e+]+) ([-\d.e+]+)", meta)
x0, x1, z0, z1 = (float(m.group(i)) for i in range(1, 5))
nx, nz = int(m.group(5)), int(m.group(6))
dx, dz = float(m.group(7)), float(m.group(8))
dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
L0 = float(re.search(r"L0 ([\d.]+)", meta).group(1))
B0eq = float(re.search(r"B0eq ([\d.]+)", meta).group(1))
sh = re.search(r"shell (\S+) L0=([\d.]+) dL=([\d.]+) edge=([\d.]+)", meta)
sp_name, sL0, sdL, sedge = sh.group(1), *(float(sh.group(i)) for i in (2, 3, 4))
band = re.search(r"active_band ([\d.]+) ([\d.]+)", meta)
Lb0, Lb1 = float(band.group(1)), float(band.group(2))
p1 = [(float(a), float(b)) for a, b in
      re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
p2 = [(float(a), float(b)) for a, b in
      re.findall(r"^\s+L2\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
pL2 = re.search(r"probe2_L ([\d.]+)", meta)
L2 = float(pL2.group(1)) if pL2 else 0.0

def dget(sec, key, dflt):
    txt = open(deck_path).read()
    msec = re.search(rf"\[{sec}[^\]]*\]([^\[]*)", txt)
    if not msec: return dflt
    mkey = re.search(rf"^\s*{key}\s*=\s*([-\d.e+]+)", msec.group(1), re.M)
    return float(mkey.group(1)) if mkey else dflt

prebal = int(dget("background", "prebalance", 0))
nc = dget("cold", "nc", 1.0)
absorber = int(dget("boundary", "absorber_cells", 240))
runway = dget("boundary", "runway_lam_deg", 25.0)
x_min = dget("domain", "x_min", 0.0)
n0 = dget("species engine", "n0", 0.01)
uthpar = dget("species engine", "uthpar", 0.2)
uthperp = dget("species engine", "uthperp", 0.3)
ppc = int(dget("species engine", "ppc", 100))
deltaf = int(dget("species engine", "deltaf", 1))
nsteps = int(dget("time", "nsteps", 0))
snap_ev = int(dget("diag", "snap_every", 0))
probe_ev = int(dget("diag", "probe_every", 4))

# ---- density map ----------------------------------------------------------
dens = np.fromfile(f"{outdir}/dens_{sp_name}.bin", dtype=np.float32)
dens = dens.reshape(nz, nx) / (dx * dz) / n0          # n/n0
xs = x0 + (np.arange(nx) + 0.5) * dx
zs = z0 + (np.arange(nz) + 0.5) * dz

# load verification numbers
Xg, Zg = np.meshgrid(xs[::2], zs[::2])
Lg = (Xg**2 + Zg**2) ** 1.5 / Xg**2                    # dipole2d L = r^3/x^2
d2 = dens[::2, ::2]
wtot = d2.sum()
Lmean = (Lg * d2).sum() / wtot
env = np.abs(Lg - sL0) <= sdL / 2 + 3 * sedge
frac3 = d2[env].sum() / wtot

def line_xz(L, lam):
    c = np.cos(lam)
    return L * c**3, L * c**2 * np.sin(lam)

# ---- derived scales -------------------------------------------------------
wph = np.sqrt(n0)
lamD_hot = uthpar / wph
rho_eq = uthperp / B0eq
def mr(lam):
    t = np.tan(lam)
    return np.sqrt(1 + 4 * t * t) / np.cos(lam) ** 2
rho_rw = uthperp / (B0eq * mr(np.radians(runway)))
w03 = 0.3 * B0eq
k03 = w03 * np.sqrt(1 + nc / (w03 * (B0eq - w03)))
lam_w03 = 2 * np.pi / k03
A = uthperp**2 / uthpar**2 - 1
lam_wall = np.degrees(np.arccos(np.cbrt((x_min + absorber * dx) / L0))) \
    if x_min > 0 else float("nan")
cfl = 0.999 * min(dx, dz) / np.sqrt(2)

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1.1])
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1]); axC.axis("off")

im = axA.imshow(dens, origin="lower", extent=[x0, x1, z0, z1], cmap="inferno",
                aspect="equal", vmin=0, vmax=np.percentile(dens, 99.9))
fig.colorbar(im, ax=axA, label="hot n/n0 (loaded, t=0)", shrink=0.75)
lam = np.linspace(-np.radians(60), np.radians(60), 400)
for Lc, st, cl in [(sL0 - sdL/2, "-", "w"), (sL0 + sdL/2, "-", "w"),
                   (sL0 - sdL/2 - 3*sedge, ":", "w"),
                   (sL0 + sdL/2 + 3*sedge, ":", "w"),
                   (Lb0, "--", "lime"), (Lb1, "--", "lime")]:
    px, pz = line_xz(Lc, lam)
    inb = (px > x0) & (px < x1) & (pz > z0) & (pz < z1)
    axA.plot(px[inb], pz[inb], st, color=cl, lw=0.8, alpha=0.8)
# absorber frame + particle wall
axA.add_patch(plt.Rectangle((x0 + absorber*dx, z0 + absorber*dz),
                            (x1-x0) - 2*absorber*dx, (z1-z0) - 2*absorber*dz,
                            fill=False, ec="cyan", ls="-", lw=1.2))
axA.axvline(x_min, color="red", lw=1.2, ls="-")
# runway end marks on L0
for s in (+1, -1):
    rx, rz = line_xz(L0, s * np.radians(runway))
    axA.plot(rx, rz, "x", color="orange", ms=10, mew=2)
# probe stations
for (tab, L, cl, mk) in [(p1, L0, "deepskyblue", "o"), (p2, L2, "magenta", "s")]:
    for lamd, _ in tab:
        px, pz = line_xz(L, np.radians(lamd))
        axA.plot(px, pz, mk, color=cl, ms=5, mec="k", mew=0.4)
axA.set_xlabel("x [c/ωpe]"); axA.set_ylabel("z [c/ωpe]")
axA.set_title(f"{outdir} — initial state map\n"
              "white: shell flat-top(—)/±3σ(:) · green: active band · "
              "cyan: wave absorber frame · red: particle wall x_min ·\n"
              "orange ×: runway end · dots: probes L0(blue)/L2(magenta) · "
              f"cold fluid nc={nc} UNIFORM over the whole field region")

kc = np.argmin(np.abs(zs))
axB.plot(xs, dens[kc-8:kc+8].mean(axis=0), "C1", label="hot n/n0 (eq, |z|<2)")
axB.axhline(nc / n0, color="C0", ls="--",
            label=f"cold fluid nc/n0 = {nc/n0:.0f} (uniform)")
for Lc in (sL0 - sdL/2, sL0 + sdL/2):
    axB.axvline(Lc, color="k", lw=0.7)
for Lc in (sL0 - sdL/2 - 3*sedge, sL0 + sdL/2 + 3*sedge):
    axB.axvline(Lc, color="k", lw=0.7, ls=":")
axB.axvline(L0, color="deepskyblue", ls="-.", lw=1, label="probe L0")
if L2: axB.axvline(L2, color="magenta", ls="-.", lw=1, label="probe L2")
axB.set_yscale("log"); axB.set_ylim(1e-3, 2 * nc / n0)
axB.set_xlim(sL0 - 6*sdL, sL0 + 6*sdL)
axB.set_xlabel("x = L at equator"); axB.set_ylabel("n/n0")
axB.legend(fontsize=8, loc="center left"); axB.grid(alpha=0.3)
axB.set_title("equatorial density cut (log)")

probe_txt1 = " ".join(f"{a:+.0f}°" for a, _ in p1)
card = f"""RUN CARD  (deck: {deck_path.split('/')[-1]},  nsteps = {nsteps} → t = {nsteps*dt:.0f}/ωpe = {nsteps*dt*B0eq:.0f}/Ωe_eq)

GRID/BOX   x∈[{x0:.0f},{x1:.0f}]  z∈[±{z1:.0f}]   {nx}×{nz} cells, dx=dz={dx}
           dt={dt} (CFL limit {cfl:.3f}, ratio {dt/cfl:.2f})
SCALES     ωpe/Ωe_eq = {1/B0eq:.0f}   L0={L0:.0f} → lre_eff=√1.5·L0={np.sqrt(1.5)*L0:.0f} (1D-equivalent)
           hot Debye λ_D=uth_par/ωph = {lamD_hot:.2f} = {lamD_hot/dx:.1f} dx    (cold: T=0 fluid, no Debye scale)
           hot ρ⊥: eq {rho_eq:.2f} = {rho_eq/dz:.1f} dz | runway edge {rho_rw:.2f} = {rho_rw/dz:.1f} dz (gate ≥2)
           whistler λ_par(0.3Ωe,eq) = {lam_w03:.1f} = {lam_w03/dx:.0f} cells;  shell dL = {sdL:.0f} = {sdL/lam_w03:.1f} λ_par
PLASMA     cold fluid nc={nc} (linearized, T=0) UNIFORM everywhere fields live
           hot {'δf' if deltaf else 'FULL-F'} n0={n0}  uth_par={uthpar}  uth⊥={uthperp}  (A={A:.1f})
           shell L0={sL0:.0f} ΔL={sdL:.0f} edge σ={sedge:.0f} (Gaussian)  ppc={ppc}
           diamagnetic prebalance: {'ON — dent δB∥=−p⊥/B0 (eq ' + format(-n0*uthperp**2/B0eq, '.1e') + ') seeded in wave-B at t=0' if prebal else 'OFF'}
LOAD CHECK ⟨L⟩ = {Lmean:.1f} (design {sL0:.0f});  {100*frac3:.2f}% of weight inside ±3σ envelope
BOUNDARY   waves: Umeda masked damping frame, {absorber} cells ({absorber*dx:.0f} c/ωpe) on all box edges
                  + band-edge damping ramps at active_band L∈[{Lb0:.0f},{Lb1:.0f}] (green)
           particles: field-aligned reflection at gc x<x_min={x_min:.0f} (red) and |z| walls
           runway ±{runway:.0f}° (orange ×): absorber verified strictly outside
           wall latitude {lam_wall:.1f}°, mirror ratio {mr(np.radians(lam_wall)):.2f}  (1D twin had 2.47)
PROBES     L0={L0:.0f} line (blue ●) and L2={L2:.0f} line (magenta ■), λ = {probe_txt1}
           sampled every {probe_ev} steps; local fce recorded per station (dual normalization)
DIAG/NYQ   probes: Δt={probe_ev*dt:.1f}/ωpe → Nyquist {np.pi/(probe_ev*dt):.1f} ωpe = {np.pi/(probe_ev*dt)/B0eq:.0f} Ωe_eq (spectra: probes ONLY)
           f2d: every {snap_ev} steps = {snap_ev*dt*B0eq:.0f}/Ωe ({nsteps//max(snap_ev,1)} frames, {6*nx*nz*4*(nsteps//max(snap_ev,1))/1e9:.0f} GB) — position/envelope only
           sline: every {max(1,snap_ev//4)} steps = {max(1,snap_ev//4)*dt*B0eq:.1f}/Ωe along L0 line — position/envelope only (Nyquist below whistler band)
"""
axC.text(0.0, 1.0, card, family="monospace", fontsize=8.3, va="top")
fig.tight_layout()
fig.savefig(png, dpi=115)
print(card)
print("wrote", png)
