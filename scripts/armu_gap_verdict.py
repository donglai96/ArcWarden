#!/usr/bin/env python3
"""Phase-1 verdict (RSM_TEST_PLAN.md): does the Arm U selective gap survive
the f575a97 apron-OOB fix?

Probe-based, quantitative only (no visual claims):
  per probe (equator, +-145, +-291), PSD over the analysis window ->
    P_lo  = band power [0.35, 0.45] We0   (lower shoulder)
    P_gap = band power [0.45, 0.55] We0   (gap band)
    P_up  = band power [0.55, 0.65] We0   (upper shoulder)
  in-run valley   V = P_gap / sqrt(P_lo * P_up)          (<1 = valley)
  cross-run ratio R_x = P_x^rsm / P_x^ctrl
  selectivity     S = R_gap / sqrt(R_lo * R_up)          (<1 = gap-selective)
Accept (plan sec 2): ctrl has no deep valley; rsm suppression is
band-selective (S well below 1), visible at equator + one near-source pair.

Usage: armu_gap_verdict.py CTRL_DIR RSM_DIR [--tmin 1000] [--tmax 1e9]
       [--wce 0.2] [--dtrec 1.5] [--out FIG.png] [--label TAG]
dtrec = probe record spacing in 1/wpe (probe_every*dt). Times in We0^-1.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

NPR = 21
PROBE_LAT = {8: "-291", 9: "-145", 10: "eq", 11: "+145", 12: "+291"}
BANDS = {"lo": (0.35, 0.45), "gap": (0.45, 0.55), "up": (0.55, 0.65)}


def load_probe(d, dtrec_wpe, wce):
    raw = np.fromfile(Path(d) / "probe.bin", dtype=np.float32)
    nrec = raw.size // (NPR * 2)
    a = raw[: nrec * NPR * 2].reshape(nrec, NPR, 2)
    t = np.arange(nrec) * dtrec_wpe * wce          # in 1/We0
    return t, a


def band_powers(t, a, ip, tmin, tmax, dtrec_we):
    m = (t >= tmin) & (t <= tmax)
    x = a[m, ip, 0].astype(np.float64)
    y = a[m, ip, 1].astype(np.float64)
    fs = 1.0 / dtrec_we                            # samples per We0 time
    nseg = min(4096, x.size)
    f, pxx = signal.welch(x, fs=fs, nperseg=nseg)
    _, pyy = signal.welch(y, fs=fs, nperseg=nseg)
    p = pxx + pyy                                  # both transverse components
    w = f * 2 * np.pi                              # rad: omega/We0
    out = {}
    for k, (w0, w1) in BANDS.items():
        sel = (w >= w0) & (w < w1)
        out[k] = float(np.trapz(p[sel], w[sel])) if sel.any() else np.nan
    return out, w, p


def spectrogram(t, a, ip, dtrec_we):
    x = a[:, ip, 0].astype(np.float64) + 1j * a[:, ip, 1].astype(np.float64)
    fs = 1.0 / dtrec_we
    f, tt, S = signal.spectrogram(x, fs=fs, nperseg=1024, noverlap=896,
                                  return_onesided=False)
    idx = np.argsort(f)
    return f[idx] * 2 * np.pi, tt + t[0], np.abs(S[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl"); ap.add_argument("rsm")
    ap.add_argument("--tmin", type=float, default=1000.0)
    ap.add_argument("--tmax", type=float, default=1e9)
    ap.add_argument("--wce", type=float, default=0.2)
    ap.add_argument("--dtrec", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="armu_fixed")
    args = ap.parse_args()
    dtrec_we = args.dtrec * args.wce

    tc, ac = load_probe(args.ctrl, args.dtrec, args.wce)
    tr, ar = load_probe(args.rsm, args.dtrec, args.wce)
    tmax = min(args.tmax, tc[-1], tr[-1])
    print(f"# window t = [{args.tmin:g}, {tmax:g}] /We0  "
          f"(ctrl nrec={tc.size}, rsm nrec={tr.size})")
    hdr = (f"{'probe':>6} {'V_ctrl':>8} {'V_rsm':>8} {'R_lo':>8} "
           f"{'R_gap':>8} {'R_up':>8} {'S_sel':>8}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for ip in sorted(PROBE_LAT):
        pc, w, psd_c = band_powers(tc, ac, ip, args.tmin, tmax, dtrec_we)
        pr, _, psd_r = band_powers(tr, ar, ip, args.tmin, tmax, dtrec_we)
        Vc = pc["gap"] / np.sqrt(pc["lo"] * pc["up"])
        Vr = pr["gap"] / np.sqrt(pr["lo"] * pr["up"])
        R = {k: pr[k] / pc[k] for k in BANDS}
        S = R["gap"] / np.sqrt(R["lo"] * R["up"])
        rows[ip] = (w, psd_c, psd_r)
        print(f"{PROBE_LAT[ip]:>6} {Vc:8.3f} {Vr:8.3f} {R['lo']:8.3f} "
              f"{R['gap']:8.3f} {R['up']:8.3f} {S:8.3f}")
    print("# V<1 valley in-run | R_x = rsm/ctrl band power | "
          "S<1 gap-selective suppression")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, (lab, t, a) in enumerate(
            [("ctrl", tc, ac), ("rsm", tr, ar)]):
        wsp, tt, S = spectrogram(t, a, 10, dtrec_we)
        m = (wsp >= 0) & (wsp <= 1.0)
        ax = axs[0, col]
        pc = ax.pcolormesh(tt, wsp[m], np.log10(S[m] + 1e-12), shading="auto",
                           cmap="jet")
        fig.colorbar(pc, ax=ax, label=r"log$_{10}$ spec")
        for yv in (0.45, 0.55):
            ax.axhline(yv, color="w", lw=0.6, ls="--")
        ax.set(title=f"{args.label} {lab} @equator", xlabel=r"t $\Omega_e$",
               ylabel=r"$\omega/\Omega_e$")
    ax = axs[1, 0]
    w, psd_c, psd_r = rows[10]
    ax.semilogy(w, psd_c, label="ctrl"); ax.semilogy(w, psd_r, label="rsm")
    ax.axvspan(0.45, 0.55, color="0.85")
    ax.set(xlim=(0, 1), xlabel=r"$\omega/\Omega_e$", ylabel="PSD",
           title="equator PSD (window)")
    ax.legend()
    ax = axs[1, 1]
    for ip in (9, 11):
        _, pc_, pr_ = rows[ip]
        ax.semilogy(rows[ip][0], pc_, lw=0.8, label=f"ctrl {PROBE_LAT[ip]}")
        ax.semilogy(rows[ip][0], pr_, lw=0.8, ls="--",
                    label=f"rsm {PROBE_LAT[ip]}")
    ax.axvspan(0.45, 0.55, color="0.85")
    ax.set(xlim=(0, 1), xlabel=r"$\omega/\Omega_e$", ylabel="PSD",
           title=r"$\pm145$ PSD")
    ax.legend(fontsize=7)
    out = args.out or f"{args.label}_verdict.png"
    fig.savefig(out, dpi=140)
    print(f"fig -> {out}")

    # multi-probe spectrogram grid: rows = ctrl/rsm, cols = probe locations
    probes = sorted(PROBE_LAT)
    fig2, axs2 = plt.subplots(2, len(probes), figsize=(4 * len(probes), 7),
                              sharex=True, sharey=True, constrained_layout=True)
    for row, (lab, t, a) in enumerate([("ctrl", tc, ac), ("rsm", tr, ar)]):
        for col, ip in enumerate(probes):
            wsp, tt, S = spectrogram(t, a, ip, dtrec_we)
            msel = (wsp >= 0) & (wsp <= 1.0)
            ax = axs2[row, col]
            pc = ax.pcolormesh(tt, wsp[msel], np.log10(S[msel] + 1e-12),
                               shading="auto", cmap="jet")
            for yv in (0.45, 0.55):
                ax.axhline(yv, color="w", lw=0.6, ls="--")
            ax.set_title(f"{lab} @{PROBE_LAT[ip]}", fontsize=10)
            if col == len(probes) - 1:
                fig2.colorbar(pc, ax=ax, label=r"log$_{10}$ spec")
            if row == 1:
                ax.set_xlabel(r"t $\Omega_e$")
            if col == 0:
                ax.set_ylabel(r"$\omega/\Omega_e$")
    out2 = out.replace("_verdict", "_specs")
    fig2.savefig(out2, dpi=140)
    print(f"fig -> {out2}")


if __name__ == "__main__":
    sys.exit(main())
