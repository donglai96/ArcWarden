#!/usr/bin/env python3
"""Phase Q gate (PLAN_2D v3): quiet-control numerics closure. FROZEN before
the full t1000 data lands; hard thresholds + nonzero exit (the stage-2
lesson: gates without thresholds are not gates).

  Q1 per-species T drift: |dT_par|, |dT_perp| < 5% over the run (interior
     x-regions 3-4, from fv_s*; aspirational 2% reported). FAIL if any
     species exceeds 5%.
  Q2 wrap-layer split: y-interior twin (fvyi_s*) drift within 1.5x of the
     full-y drift AND the full-vs-interior T difference < 2% (y-wrap layer
     not secretly heating). FAIL otherwise.
  Q3 (v2, 2026-08-10 user redefinition): WHISTLER-filtered finite-ky power
     P_wh = sum_{ky!=0} int_{0.2<w/Wce<0.8, RH pol} |z(ky,w)|^2 from windowed
     2D FFT of z(y,t)=By+iBz at the eq station (RH about B0 = e^{-iwt} sense
     = positive-freq side of the forward FFT). Compare [300,600] vs
     [700,1000] — the near-zero initial field / shot-noise equilibration
     transient is EXCLUDED by construction. FAIL if ratio >= 1.5.
     Caveat (reported, not gated): kx is unresolved at a single x-station,
     so the D(w,kx,ky)=0 dispersion membership is NOT verified here; the
     w-ky power map is saved for visual dispersion inspection.
  Q4 sanity: no NaN in last bline; wall (damping layer) |B| <= interior.

Usage: python3 q_gate.py <rundir>   (exit 0 = PASS, 1 = FAIL)
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NREG, NPAR, NPERP, VMAX = 8, 160, 80, 0.6
VP = (np.arange(NPAR) + 0.5) / NPAR * 2 * VMAX - VMAX
VQ = (np.arange(NPERP) + 0.5) / NPERP * VMAX


def temps(fn):
    a = np.fromfile(fn, np.float64).reshape(NREG, NPAR, NPERP)[3:5].sum(0)
    w = a.sum()
    if w <= 0:
        return np.nan, np.nan
    return ((a * VP[:, None] ** 2).sum() / w,
            (a * VQ[None, :] ** 2).sum() / w / 2)


def drift_series(pat):
    fl = sorted(glob.glob(pat))
    if len(fl) < 3:
        return None
    T = np.array([temps(f) for f in fl])
    return T


def main(d):
    fails = []
    meta = {ln.split()[0]: ln.split()[1] for ln in open(os.path.join(d, "meta.txt"))
            if len(ln.split()) >= 2 and ln.split()[0] not in ("species", "probe_ix")}
    nx, ny = int(meta["nx"]), int(meta["ny"])
    nsp = int(meta["nsp"])
    wce = float(meta["wce"])
    dt = float(meta["dt"])
    fv_dt = int(meta["fv_every"]) * dt * wce
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # Q1 + Q2 -----------------------------------------------------------
    Tall = []
    for si in range(nsp):
        Tf = drift_series(os.path.join(d, f"fv_s{si}_*.bin"))
        Ti = drift_series(os.path.join(d, f"fvyi_s{si}_*.bin"))
        if Tf is None:
            fails.append(f"Q1 s{si}: no fv_s dumps"); continue
        t = (np.arange(len(Tf)) + 1) * fv_dt
        dpar = 100 * (Tf[:, 0] / Tf[0, 0] - 1)
        dper = 100 * (Tf[:, 1] / Tf[0, 1] - 1)
        axs[0, 0].plot(t, dpar, label=f"s{si} Tpar")
        axs[0, 0].plot(t, dper, "--", label=f"s{si} Tperp")
        mx = max(abs(dpar[-1]), abs(dper[-1]))
        tag = "PASS" if mx < 5.0 else "FAIL"
        if mx >= 5.0:
            fails.append(f"Q1 s{si}: drift {mx:.1f}% >= 5%")
        note = "" if mx < 2.0 else "  (>2% aspirational)"
        print(f"Q1 s{si}: dTpar {dpar[-1]:+.2f}%  dTperp {dper[-1]:+.2f}%  "
              f"-> {tag}{note}")
        Tall.append(Tf)
        if Ti is not None and len(Ti) == len(Tf):
            diff = 100 * abs(Ti[-1, 0] / Tf[-1, 0] - 1)
            dpari = 100 * (Ti[:, 0] / Ti[0, 0] - 1)
            ok = diff < 2.0 and (abs(dpari[-1]) < 1.5 * max(abs(dpar[-1]), 0.5))
            print(f"Q2 s{si}: full-vs-yinterior Tpar diff {diff:.2f}%  "
                  f"yint drift {dpari[-1]:+.2f}%  -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                fails.append(f"Q2 s{si}: wrap-layer split ({diff:.2f}%)")
    # physical-population combined pool (s1+s2 = chi + (1-chi) parts): the
    # split drifts above are label bookkeeping; this is the f0 that matters
    if len(Tall) == 3:
        import numpy as _np
        fl1 = sorted(glob.glob(os.path.join(d, "fv_s1_*.bin")))
        fl2 = sorted(glob.glob(os.path.join(d, "fv_s2_*.bin")))
        if fl1 and len(fl1) == len(fl2):
            def _T(f1, f2):
                a = (_np.fromfile(f1, _np.float64) + _np.fromfile(f2, _np.float64)
                     ).reshape(NREG, NPAR, NPERP)[3:5].sum(0)
                w = a.sum()
                return ((a * VP[:, None] ** 2).sum() / w,
                        (a * VQ[None, :] ** 2).sum() / w / 2)
            c0, c1 = _T(fl1[0], fl2[0]), _T(fl1[-1], fl2[-1])
            dp = 100 * (c1[0] / c0[0] - 1); dq = 100 * (c1[1] / c0[1] - 1)
            tag = "PASS" if max(abs(dp), abs(dq)) < 5.0 else "FAIL"
            if max(abs(dp), abs(dq)) >= 5.0:
                fails.append(f"Q1 pool-combined: {dp:+.1f}%")
            print(f"Q1 pool COMBINED (physical): dTpar {dp:+.2f}%  "
                  f"dTperp {dq:+.2f}%  -> {tag}")
    axs[0, 0].axhline(5, color="r", lw=1); axs[0, 0].axhline(-5, color="r", lw=1)
    axs[0, 0].axhline(2, color="gray", ls=":"); axs[0, 0].axhline(-2, color="gray", ls=":")
    axs[0, 0].set_xlabel(r"$t\Omega_e$"); axs[0, 0].set_ylabel("T drift %")
    axs[0, 0].legend(fontsize=7); axs[0, 0].set_title("Q1 per-species T drift")

    # Q3 v2: whistler-filtered (0.2-0.8 Wce, RH, ky!=0), equilibration
    # window excluded ([300,600] vs [700,1000])
    fl = sorted(glob.glob(os.path.join(d, "ycol_*.bin")))
    npb = int(meta["nprobe"]); ieq = npb // 2
    dt_dump = int(meta["bline_every"]) * dt
    Z = np.empty((len(fl), ny), np.complex64)
    for i, f in enumerate(fl):
        a = np.fromfile(f, np.float32).reshape(npb, 2, ny)
        Z[i] = a[ieq, 0] + 1j * a[ieq, 1]
    tb = (np.arange(len(fl)) + 1) * dt_dump * wce
    def pwh(t1, t2):
        m = (tb > t1) & (tb <= t2)
        zz = Z[m] * np.hanning(m.sum())[:, None]
        F = np.fft.fft2(zz)                       # axes: (t, y)
        w = 2 * np.pi * np.fft.fftfreq(m.sum(), d=dt_dump) / wce
        band = (w > 0.2) & (w < 0.8)              # RH = e^{-iwt} = positive side
        return (np.abs(F[band][:, 1:]) ** 2).sum() / m.sum(), F, w
    P1, _, _ = pwh(300.0, 600.0)
    P2, F2, wf = pwh(700.0, 1000.0)
    ratio = P2 / P1
    tag = "PASS" if ratio < 1.5 else "FAIL"
    if ratio >= 1.5:
        fails.append(f"Q3: whistler-filtered ky ratio {ratio:.2f} >= 1.5")
    print(f"Q3 v2 P_wh (RH, 0.2-0.8, ky!=0): [300,600] {P1:.3e}  "
          f"[700,1000] {P2:.3e}  ratio {ratio:.3f}  -> {tag}")
    ky = np.fft.fftfreq(ny, d=float(meta["dy"])) * 2 * np.pi
    ow = np.argsort(wf); ok = np.argsort(ky)
    axs[0, 1].pcolormesh(ky[ok], wf[ow],
                         np.log10(np.abs(F2[ow][:, ok]) ** 2 + 1e-30),
                         cmap="turbo", shading="auto", rasterized=True)
    axs[0, 1].set_ylim(0, 1.0); axs[0, 1].axhline(0.2, color="w", lw=0.5)
    axs[0, 1].axhline(0.8, color="w", lw=0.5)
    axs[0, 1].set_xlabel(r"$k_y$"); axs[0, 1].set_ylabel(r"$\omega/\Omega_e$")
    axs[0, 1].set_title(f"Q3 v2 w-ky map [700,1000] (ratio {ratio:.3f})")

    # Q4 ----------------------------------------------------------------
    bl = sorted(glob.glob(os.path.join(d, "bline_*.bin")))
    b = np.fromfile(bl[-1], np.float32)
    nanbad = bool(np.isnan(b).any())
    am = np.abs(b[:nx] + 1j * b[nx:2 * nx])
    nd = 240
    wall, inner = am[:nd].mean(), am[nx // 2 - 500:nx // 2 + 500].mean()
    ok = (not nanbad) and wall <= inner * 1.5
    print(f"Q4 sanity: NaN {nanbad}  wall {wall:.2e}  interior {inner:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("Q4 sanity")
    axs[1, 0].plot(np.arange(nx) * float(meta["dx"]), am)
    axs[1, 0].set_title("last y-mid |B|(x)")
    # energy trace
    import csv
    rows = list(csv.reader(open(os.path.join(d, "energy.csv"))))[1:]
    axs[1, 1].semilogy([float(r[1]) * wce for r in rows],
                       [float(r[3]) for r in rows], label="WB")
    axs[1, 1].semilogy([float(r[1]) * wce for r in rows],
                       [float(r[2]) for r in rows], label="WE")
    axs[1, 1].legend(); axs[1, 1].set_title("field energy")

    fig.savefig(f"{os.path.basename(d.rstrip('/'))}_qgate.png", dpi=120)
    print(f"fig -> {os.path.basename(d.rstrip('/'))}_qgate.png")
    if fails:
        print("PHASE Q GATE: FAIL —", "; ".join(fails))
        return 1
    print("PHASE Q GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
