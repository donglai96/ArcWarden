#!/usr/bin/env python3
"""G1.3 quicklook — mirror2d diagnostics pack (fv / wl / f2d formats).

Panels:
  (a) f(vpar, vperp) at the equator region, latest fv snapshot (log)
  (b) delta f vs the FIRST fv snapshot (plateau / scar carving)
  (c) resonance-tagged J.E ledger, equator: Landau (E_par) vs cyclotron
      (E_perp) channels vs vpar — the division-of-labor figure
  (d) WNA map from the latest f2d snapshot: k-space magnetic power vs
      theta = atan2(ky, kx) and |k| (axis-aligned B0 to lowest order)

Usage: plot_mirror2d_diags.py <rundir> <out.png>
"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def meta(d):
    m = {}
    for ln in open(f"{d}/meta.txt"):
        p = ln.split()
        if len(p) >= 2 and p[0] not in ("species", "probe_ix"):
            m[p[0]] = float(p[1])
    return m


def main():
    d, out = sys.argv[1], sys.argv[2]
    m = meta(d)
    nreg, npar, nperp = int(m["nreg"]), int(m["npar"]), int(m["nperp"])
    nwb, vmax = int(m["nwb"]), m["vmax"]
    nx, ny = int(m["nx"]), int(m["ny"])
    eq = nreg // 2                      # equator region (b0_xc = Lx/2 decks)

    fvf = sorted(glob.glob(f"{d}/fv_*.bin"))
    wlf = sorted(glob.glob(f"{d}/wl_*.bin"))
    f2f = sorted(glob.glob(f"{d}/f2d_*.bin"))
    fv0 = np.fromfile(fvf[0]).reshape(nreg, npar, nperp)
    fv1 = np.fromfile(fvf[-1]).reshape(nreg, npar, nperp)
    vpar = (np.arange(npar) + 0.5) / npar * 2 * vmax - vmax
    vperp = (np.arange(nperp) + 0.5) / nperp * vmax

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    F = fv1[eq].T
    pc = ax.pcolormesh(vpar, vperp, np.maximum(F / F.max(), 1e-8),
                       norm=LogNorm(vmin=1e-6, vmax=1), cmap="viridis",
                       shading="auto", rasterized=True)
    ax.set_xlabel(r"$v_\parallel/c$"); ax.set_ylabel(r"$v_\perp/c$")
    ax.set_title(f"(a) f(vpar,vperp) equator region, {fvf[-1].split('/')[-1]}")
    fig.colorbar(pc, ax=ax)

    ax = axes[0, 1]
    D = (fv1[eq] - fv0[eq]).T / fv0[eq].max()
    vm = np.abs(D).max() or 1e-12
    pc = ax.pcolormesh(vpar, vperp, D, cmap="RdBu_r", vmin=-vm, vmax=vm,
                       shading="auto", rasterized=True)
    ax.set_xlabel(r"$v_\parallel/c$"); ax.set_ylabel(r"$v_\perp/c$")
    ax.set_title("(b) $\\Delta f$ since first snapshot (scar monitor)")
    fig.colorbar(pc, ax=ax)

    ax = axes[1, 0]
    wl = sum(np.fromfile(f).reshape(nreg, nwb, 2) for f in wlf)  # whole run
    vw = (np.arange(nwb) + 0.5) / nwb * 2 * vmax - vmax
    ax.plot(vw, wl[eq, :, 0], "tab:blue", lw=1.4, label="Landau ($E_\\parallel$)")
    ax.plot(vw, wl[eq, :, 1], "tab:red", lw=1.4, label="cyclotron ($E_\\perp$)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$v_\parallel/c$"); ax.set_ylabel("work on particles (code units)")
    ax.set_title("(c) resonance-tagged J·E ledger, equator (run total)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    if f2f:
        f2 = np.fromfile(f2f[-1], dtype=np.float32).reshape(6, ny, nx)
        Bk = np.fft.fftshift(np.abs(np.fft.fft2(f2[4] + 1j * f2[5])) ** 2)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=m["dx"])) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=m["dy"])) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        kk = np.hypot(KX, KY); th = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))
        kb = np.linspace(0, kk.max() * 0.5, 40)
        tb = np.linspace(0, 90, 30)
        H, _, _ = np.histogram2d(kk.ravel(), th.ravel(), bins=[kb, tb],
                                 weights=Bk.ravel())
        pc = ax.pcolormesh(tb, kb, np.maximum(H / (H.max() or 1), 1e-8),
                           norm=LogNorm(vmin=1e-5, vmax=1), cmap="inferno",
                           shading="auto", rasterized=True)
        ax.set_xlabel(r"WNA $\theta$ (deg)"); ax.set_ylabel(r"$|k| c/\omega_{pe}$")
        ax.set_title(f"(d) magnetic k-power vs WNA, {f2f[-1].split('/')[-1]}")
        fig.colorbar(pc, ax=ax)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
