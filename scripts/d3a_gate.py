#!/usr/bin/env python3
"""D3a gate (PLAN_2D v4 D3, UB-fuel-free arm). FROZEN 2026-08-14 before the
resumed run's data (t4500 -> t10000) is judged. Hard gates + exit code.

Design recap: f0 = biMax T_perp/T_par = 2.0 -> KP fuel edge w_m = 0.50
EXACTLY. Any power at w > 0.55 (eq units) is NOT linear fuel: it must be
harmonic / two-stage (wave-processed f) / crossing "dots" / latitude
mapping. D2 (T-ratio 4.5, w_m = 0.78) = fuel-loaded control that grew a
spectrally-continuous UB extension.

Gates (pre-registered):
  G0 validity:  wd_max(end) < 0.3  (delta-f linearization health; D2 ended
                at 0.12 = watch zone). FAIL -> all physics readouts invalid.
  G1 ignition:  max(gamma_LB[1000,5000], gamma_LB[5000,9500]) > 1e-4 AND
                WB(end)/min WB(t>500) > 10. FAIL -> verdict = INCONCLUSIVE
                (UB question UNTESTED — no LB waves to process f; the one
                pre-authorized retry is nh 0.015 -> 0.025, nothing else).
  G2 UB (judged only if G1 PASS):
                P_UB[8500,9800] / P_UB[500,1500] > 10 AND
                max(gamma_UB windows) > 1e-4  -> UB EMERGED
                else UB ABSENT -> pre-registered conclusion: ambient
                sub-threshold UB cannot arise in the 2D slab without
                fuel/injection or crossing elements (D2 contrast).
Attribution readouts (printed, no gates; meaningful only if UB EMERGED):
  A1 harmonic:   bicoherence b2(f,f)->2f, late window, vs 3x floor; UB peak
                 freq vs 2x LB peak freq.
  A2 latitude:   UB onset time eq vs off-eq stations (local-unit bands).
  A3 two-stage:  delta-f in the UB cyclotron shell |v_par| in [0.05,0.08]c
                 (eq region, from fv time series = sum w*wd) vs UB onset.
Always reported: eq + off-eq spectrograms, band envelopes, W10, wl
Landau-vs-cyclotron ledger, and the same measures on the D2 control.

Usage: python3 d3a_gate.py <d3a_dir> <d2_dir>
Exit 0 = G0+G1 PASS (UB verdict then printed either way), 1 otherwise.
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lumorph_gate import spec, w10_occupancy
from s_gate import read_meta, band_env
from ub_mine_x4 import bicoherence_diag

DEC = 20                     # probe decimation (matches s_gate)


def probe_z(d, m, ip):
    npb = int(m["nprobe"])
    raw = np.fromfile(os.path.join(d, "probe.bin"), np.float32)
    nt = len(raw) // (2 * npb)
    pb = raw[:nt * 2 * npb].reshape(nt, npb, 2)
    z = (pb[:, ip, 0] + 1j * pb[:, ip, 1])[::DEC]
    ts = m["probe_every"] * m["dt"] * DEC
    t = (np.arange(len(z)) + 1) * ts
    return z, t, ts


def station_b(m, off):
    return 1.0 + m["b0_a"] * off * off


def band_power_win(z, tOe, ts, wce, w1, w2, t1, t2):
    e = band_env(z, ts, w1, w2, wce)
    s = (tOe > t1) & (tOe <= t2)
    return float((e[s] ** 2).mean()) if s.any() else np.nan


def gamma_windows(env, tOe):
    """two fit windows scaled to run length; == ((1000,5000),(5000,9500))
    for the original t10000 run (d3b generalization, frozen pre-data)."""
    te = tOe[-1]
    out = []
    for t1, t2 in ((1000.0, 0.5 * te), (0.5 * te, te - 500.0)):
        s = (tOe > t1) & (tOe < t2)
        out.append(np.polyfit(tOe[s], np.log(env[s] + 1e-30), 1)[0]
                   if s.sum() > 10 else np.nan)
    return out


def onset_time(env, tOe, floor_win=(500.0, 1500.0), k=10.0):
    s = (tOe > floor_win[0]) & (tOe < floor_win[1])
    thr = k * np.sqrt((env[s] ** 2).mean())
    lit = np.where((env > thr) & (tOe > floor_win[1]))[0]
    return tOe[lit[0]] if len(lit) else np.inf


def fv_shell_series(d, m):
    """delta-f integral over the UB cyclotron shell, eq region, per snapshot."""
    nreg, npar, nperp = int(m["nreg"]), int(m["npar"]), int(m["nperp"])
    vmax = m["vmax"]
    vpar = (np.arange(npar) + 0.5) / npar * 2 * vmax - vmax
    sh = (np.abs(vpar) > 0.05) & (np.abs(vpar) < 0.08)
    fl = sorted(glob.glob(os.path.join(d, "fv_*.bin")))
    t = (np.arange(len(fl)) + 1) * m["fv_every"] * m["dt"] * m["wce"]
    v = [np.abs(np.fromfile(f).reshape(nreg, npar, nperp)[nreg // 2][sh]).sum()
         for f in fl]
    return t, np.array(v), fl


def main(d3, d2):
    m3, m2 = read_meta(d3), read_meta(d2)
    wce = m3["wce"]
    fails = []

    # ---- G0 wd validity ------------------------------------------------
    E = np.genfromtxt(os.path.join(d3, "energy.csv"), delimiter=",",
                      names=True)
    wd_max_end = float(E["wd_max"][-1])
    ok0 = wd_max_end < 0.3
    print(f"G0 delta-f validity: wd_max(end) = {wd_max_end:.4f}, "
          f"wd_rms(end) = {E['wd_rms'][-1]:.2e} -> {'PASS' if ok0 else 'FAIL'}")
    if not ok0:
        fails.append("G0 wd_max >= 0.3 (linearization broken)")

    # ---- G1 ignition ---------------------------------------------------
    tE = E["time"] * wce
    wb = E["WB"]
    late_ok = tE > 500
    wb_ratio = float(wb[-1] / wb[late_ok].min())
    ieq = int(m3["nprobe"]) // 2
    z, t, ts = probe_z(d3, m3, ieq)
    tOe = t * wce
    envLB = band_env(z, ts, 0.20, 0.45, wce)
    gLB = gamma_windows(envLB, tOe)
    ok1 = (np.nanmax(gLB) > 1e-4) and (wb_ratio > 10)
    print(f"G1 LB ignition: gamma_LB windows = {gLB[0]:+.2e} / {gLB[1]:+.2e}, "
          f"WB(end)/min = {wb_ratio:.1f} -> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        fails.append("G1 no ignition -> UB question UNTESTED "
                     "(pre-authorized retry: nh 0.015 -> 0.025 only)")

    # ---- G2 UB emergence ----------------------------------------------
    envUB = band_env(z, ts, 0.55, 0.70, wce)
    gUB = gamma_windows(envUB, tOe)
    pub_early = band_power_win(z, tOe, ts, wce, 0.55, 0.70, 500, 1500)
    te = tOe[-1]
    pub_late = band_power_win(z, tOe, ts, wce, 0.55, 0.70, te - 1500, te - 200)
    ub_ratio = pub_late / pub_early
    emerged = (ub_ratio > 10) and (np.nanmax(gUB) > 1e-4)
    print(f"G2 UB emergence: P_UB late/early = {ub_ratio:.2f}, gamma_UB = "
          f"{gUB[0]:+.2e} / {gUB[1]:+.2e} -> "
          f"{'EMERGED' if emerged else 'ABSENT'}"
          + ("" if ok1 else "  [VOID — G1 failed, do not interpret]"))

    # ---- always-on report ---------------------------------------------
    P, fs, tt = spec(z, t, wce, 512)
    w10, _ = w10_occupancy(P, fs, tt, 0, tOe[-1])
    envBAR = band_env(z, ts, 0.45, 0.55, wce)
    print(f"report: W10[full] = {w10:.3f}; amp_max |B|_LB = "
          f"{envLB.max():.2e} (B_th ~ 1e-3 at lre 1330); BAR gamma = "
          f"{gamma_windows(envBAR, tOe)[1]:+.2e}")

    # D2 control, same measures
    z2, t2v, ts2 = probe_z(d2, m2, int(m2["nprobe"]) // 2)
    t2Oe = t2v * m2["wce"]
    te2 = t2Oe[-1]
    pub2 = band_power_win(z2, t2Oe, ts2, m2["wce"], 0.55, 0.70, te2 - 1500, te2 - 200)
    plb2 = band_power_win(z2, t2Oe, ts2, m2["wce"], 0.20, 0.45, te2 - 1500, te2 - 200)
    plb3 = band_power_win(z, tOe, ts, wce, 0.20, 0.45, te - 1500, te - 200)
    print(f"D2 control [te-1500,te-200]: UB/LB = {pub2 / plb2:.3e}; "
          f"D3a UB/LB = {pub_late / plb3:.3e}"
          f"  (fuel-loaded vs fuel-free contrast)")

    # ---- attribution readouts ------------------------------------------
    f1, b2b, nw = bicoherence_diag(z[tOe > 0.5 * te], ts, wce)
    band = (f1 > 0.20) & (f1 < 0.40)
    j = int(np.argmax(b2b * band))
    print(f"A1 harmonic: max b2(f,f) in LB = {b2b[j]:.4f} at f = {f1[j]:.3f} "
          f"(3x floor {3.0 / nw:.4f}) -> "
          f"{'LOCKED' if b2b[j] > 3.0 / nw else 'no lock'}")

    # probe_ix lines are not in read_meta; use deck order instead:
    offs = [-560.0, -480.4, -355.6, -294.6, -234.5, -175.1, -116.4, -58.1,
            0.0, 58.1, 116.4, 175.1, 234.5, 294.6, 355.6, 480.4, 560.0]
    print("A2 latitude: UB onset time by station (local-unit band):")
    for off in (0.0, 116.4, -116.4, 234.5, -234.5):
        ip = offs.index(off)
        zz, ttv, _ = probe_z(d3, m3, ip)
        bloc = station_b(m3, off)
        e = band_env(zz, ts, 0.55 * bloc, 0.70 * bloc, wce)
        print(f"    off {off:+7.1f} (b = {bloc:.3f}): t_on = "
              f"{onset_time(e, ttv * wce):.0f}/Oe")

    tf, fsh, fvf = fv_shell_series(d3, m3)
    tub = onset_time(envUB, tOe)
    j0 = max(1, int(0.1 * len(fsh)))
    grew = fsh[-1] / (np.abs(fsh[:j0]).mean() + 1e-300)
    print(f"A3 two-stage: UB-shell |delta f| grew x{grew:.1f} over run; "
          f"UB onset t = {tub:.0f}/Oe (shell-before-onset = two-stage sign)")

    # ---- figure ---------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, (ip, lab) in zip(axs[0], [(ieq, "eq"), (offs.index(116.4), "+116"),
                                      (offs.index(234.5), "+234")]):
        zz, ttv, _ = probe_z(d3, m3, ip)
        Pp, fsp, ttp = spec(zz, ttv, wce, 512)
        ax.pcolormesh(ttp, fsp, np.log10(Pp.T + 1e-30), cmap="turbo",
                      vmin=np.log10(Pp.max()) - 5, vmax=np.log10(Pp.max()),
                      shading="auto", rasterized=True)
        ax.axhline(0.5, color="w", ls="--", lw=1)
        boff = station_b(m3, offs[ip] if ip != ieq else 0.0)
        ax.axhline(0.5 * boff, color="c", ls=":", lw=1)
        ax.set_ylim(0.05, 0.9); ax.set_title(f"D3a {lab} (cyan = local 0.5)")
    ax = axs[1, 0]
    for nm, e in (("LB", envLB), ("BAR", envBAR), ("UB", envUB)):
        ax.semilogy(tOe, e / wce, label=nm)
    ax.legend(); ax.set_title("D3a eq band envelopes /B0")
    ax.set_xlabel(r"$t\Omega_e$")
    ax = axs[1, 1]
    ax.semilogy(tf, fsh / (np.abs(fsh[:j0]).mean() + 1e-300))
    ax.axvline(tub, color="r", ls="--", lw=1, label="UB onset")
    ax.legend(); ax.set_title(r"UB shell $|\delta f|$ (eq, norm to early)")
    ax.set_xlabel(r"$t\Omega_e$")
    ax = axs[1, 2]
    P2, fs2, tt2 = spec(z2, t2v, m2["wce"], 512)
    ax.pcolormesh(tt2, fs2, np.log10(P2.T + 1e-30), cmap="turbo",
                  vmin=np.log10(P2.max()) - 5, vmax=np.log10(P2.max()),
                  shading="auto", rasterized=True)
    ax.axhline(0.5, color="w", ls="--", lw=1)
    ax.set_ylim(0.05, 0.9); ax.set_title("D2 fuel-loaded control (eq)")
    fig.savefig("d3a_gate.png", dpi=120)
    print("fig -> d3a_gate.png")

    if fails:
        print("D3a GATE: FAIL —", "; ".join(fails))
        return 1
    print(f"D3a GATE: PASS — UB verdict: {'EMERGED' if emerged else 'ABSENT'}"
          f" (fuel-free box, w_m = 0.50)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
