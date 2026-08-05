#!/usr/bin/env python3
"""Uniform-B Li big-box RSM verdict (the m=0+m1 sufficiency watershed).

Usage: rsm_big_verdict.py RSM_DIR [CTRL_DIR] [--pref PREFIX]

V6 discipline: P0 (m=0 probe/bline) and P1 (m=1 modal) NEVER mixed; a gap
claim needs (a) 0.5 feature DEEPENING across time quarters (carving, not a
mode-comb valley), (b) spanning multiple k_par modes, (c) f(vpar) RSM-ctrl
deficit at the Landau band. Also: 2W1 + J1.E1 ledger (is E_par work
sustained?) and ridge-filtered E_par/E_total from the 6-field m1line.
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB, VMAX = 240, 0.6


def read_meta(d):
    m = {}
    for line in open(os.path.join(d, "meta.txt")):
        p = line.split()
        if len(p) >= 2 and p[0] not in ("species", "probe_ix"):
            try: m[p[0]] = float(p[1])
            except ValueError: pass
    return m


def load_blines(d, nx):
    fl = sorted(glob.glob(os.path.join(d, "bline_*.bin")))
    B = np.zeros((len(fl), nx), np.complex64)
    for s, f in enumerate(fl):
        a = np.fromfile(f, dtype=np.float32)
        B[s] = a[:nx] + 1j * a[nx:2*nx]
    return B


def load_m1(d, nx):
    fl = sorted(glob.glob(os.path.join(d, "m1line_*.bin")))
    out = {k: np.zeros((len(fl), nx), np.complex64)
           for k in ("b1y", "b1z", "e1x", "e1y", "e1z", "b1x")}
    order = ["b1y", "b1z", "e1x", "e1y", "e1z", "b1x"]
    for s, f in enumerate(fl):
        a = np.fromfile(f, dtype=np.float32)
        nf = a.size // (nx * 2)
        a = a.reshape(nf, nx, 2)
        for j in range(min(nf, 6)):
            out[order[j]][s] = a[j, :, 0] + 1j * a[j, :, 1]
    return out


def wk_spec(A, dt_line, dx, wce):
    """2D FFT (t,x) -> power on (omega/We0 [signed], k_par); A complex."""
    nt, nx = A.shape
    W = np.hanning(nt)[:, None]
    F = np.fft.fftshift(np.fft.fft2(A * W))
    w = np.fft.fftshift(np.fft.fftfreq(nt, dt_line)) * 2 * np.pi / wce
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2 * np.pi
    return w, k, np.abs(F) ** 2


def quarter_wspec(A, dt_line, dx, wce):
    nt = A.shape[0]
    out = []
    for q in range(4):
        w, k, P = wk_spec(A[q*nt//4:(q+1)*nt//4], dt_line, dx, wce)
        out.append((w, k, P))
    return out


def band_metric(w, P1d):
    """min in [0.44,0.56] over median of [0.25,0.44]+[0.56,0.7] (|omega|)."""
    aw = np.abs(w)
    gap = P1d[(aw >= 0.44) & (aw <= 0.56)]
    lb = P1d[(aw >= 0.25) & (aw < 0.44)]
    ub = P1d[(aw > 0.56) & (aw <= 0.70)]
    ref = np.median(np.concatenate([lb, ub]))
    return gap.min() / ref, gap.min() / np.median(lb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rsm"); ap.add_argument("ctrl", nargs="?")
    ap.add_argument("--pref", default="rsm_big")
    a = ap.parse_args()
    m = read_meta(a.rsm)
    nx, dx, dt = int(m["nx"]), m["dx"], m["dt"]
    wce = m["wce"]; ble = int(m["bline_every"])
    dt_line = ble * dt
    tq_lab = [f"Q{q+1}" for q in range(4)]

    B0 = load_blines(a.rsm, nx)
    M = load_m1(a.rsm, nx)
    nt = B0.shape[0]
    T = nt * dt_line * wce
    print(f"# rsm {a.rsm}: nt={nt} T={T:.0f}/We0")

    # ---------- P0 / P1 quarter spectra ----------
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True,
                            sharey="row")
    Q0 = quarter_wspec(B0, dt_line, dx, wce)
    Q1 = quarter_wspec(M["b1z"], dt_line, dx, wce)
    print("# P0 (m=0) quarter gap metric: min[0.44,0.56]/median shoulders | /lower only")
    for q, (w, k, P) in enumerate(Q0):
        pw = P.sum(axis=1)
        g, gl = band_metric(w, pw)
        print(f"  P0 {tq_lab[q]}: {g:.3f} | {gl:.3f}")
        ax = axs[0, q]
        sel = (np.abs(w) <= 1.2)
        pc = ax.pcolormesh(k, w[sel], np.log10(P[sel] + 1e-9), shading="auto",
                           cmap="jet")
        ax.set(xlim=(-2, 2), title=f"P0 {tq_lab[q]}")
        if q == 3: fig.colorbar(pc, ax=ax)
        for yv in (0.45, 0.55, -0.45, -0.55):
            ax.axhline(yv, color="w", lw=0.5, ls="--")
    print("# P1 (m=1 B1z) quarter gap metric:")
    for q, (w, k, P) in enumerate(Q1):
        pw = P.sum(axis=1)
        g, gl = band_metric(w, pw)
        print(f"  P1 {tq_lab[q]}: {g:.3f} | {gl:.3f}")
        ax = axs[1, q]
        sel = (np.abs(w) <= 1.2)
        pc = ax.pcolormesh(k, w[sel], np.log10(P[sel] + 1e-9), shading="auto",
                           cmap="jet")
        ax.set(xlim=(-2, 2), title=f"P1 {tq_lab[q]}", xlabel=r"$k_\parallel$")
        if q == 3: fig.colorbar(pc, ax=ax)
        for yv in (0.45, 0.55, -0.45, -0.55):
            ax.axhline(yv, color="w", lw=0.5, ls="--")
    axs[0, 0].set_ylabel(r"P0  $\omega/\Omega_e$")
    axs[1, 0].set_ylabel(r"P1  $\omega/\Omega_e$")
    fig.savefig(f"{a.pref}_quarters.png", dpi=130)
    print(f"fig -> {a.pref}_quarters.png")

    # ---------- ridge-filtered E_par/E_total (m1, full run, quarters) ----------
    print("# ridge-filtered m1 E_par/E_total (top-1% |B1| (w,k) bins):")
    QEx = quarter_wspec(M["e1x"], dt_line, dx, wce)
    QEy = quarter_wspec(M["e1y"], dt_line, dx, wce)
    QEz = quarter_wspec(M["e1z"], dt_line, dx, wce)
    for q in range(4):
        _, _, PB = Q1[q]
        thr = np.quantile(PB, 0.99)
        msk = PB >= thr
        ex = QEx[q][2][msk].sum(); ey = QEy[q][2][msk].sum(); ez = QEz[q][2][msk].sum()
        print(f"  {tq_lab[q]}: E1x^2/(E1x^2+E1y^2+E1z^2) = {ex/(ex+ey+ez):.4f}")

    # ---------- energy + J.E ledger ----------
    e = np.genfromtxt(os.path.join(a.rsm, "energy.csv"), delimiter=",",
                      names=True)
    te = e["time"] * wce
    fig2, axs2 = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    ax = axs2[0]
    ax.semilogy(te, e["WB"], label="WB (m0)")
    ax.semilogy(te, e["W1"], label="2W1 (m1)")
    ax.set(xlabel=r"t $\Omega_e$", title="field energy"); ax.legend()
    ax = axs2[1]
    nsm = 25
    sm = lambda x: np.convolve(x, np.ones(nsm)/nsm, "same")
    ax.plot(te, sm(e["PJ1E1"]), label=r"$\langle 2Re J_1\!\cdot\!E_1^*\rangle$")
    ax.plot(te, sm(e["PJ1xE1x"]), label=r"$\langle 2Re J_{1x}E_{1x}^*\rangle$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel=r"t $\Omega_e$", title=f"m1 power ledger (smoothed x{nsm})")
    ax.legend(fontsize=8)
    for q in range(4):
        s = slice(q*len(te)//4, (q+1)*len(te)//4)
        print(f"  ledger {tq_lab[q]}: <PJ1E1>={e['PJ1E1'][s].mean():+.3e}  "
              f"<PJ1xE1x>={e['PJ1xE1x'][s].mean():+.3e}")

    # ---------- f(vpar): rsm vs ctrl ----------
    v = np.linspace(-VMAX, VMAX, NB, endpoint=False) + VMAX/NB
    fl = sorted(glob.glob(os.path.join(a.rsm, "fhist_*.bin")))
    fr0 = np.fromfile(fl[0], dtype=np.float64)
    fr1 = np.fromfile(fl[-1], dtype=np.float64)
    ax = axs2[2]
    if a.ctrl:
        cl = sorted(glob.glob(os.path.join(a.ctrl, "fhist_*.bin")))
        fc1 = np.fromfile(cl[-1], dtype=np.float64)
        n = min(len(fl), len(cl))
        d10 = np.fromfile(fl[n-1], dtype=np.float64) / \
              np.fromfile(cl[n-1], dtype=np.float64)
        ax.plot(v, d10, label=f"rsm/ctrl (dump {n-1})")
        ax.axhline(1, color="k", lw=0.6)
        for s in (1, -1):
            ax.axvspan(s*0.04, s*0.2, color="0.85", zorder=0)
        ax.set(xlim=(-0.35, 0.35), title="f(vpar) rsm/ctrl",
               xlabel=r"$v_\parallel/c$")
        ax.legend(fontsize=8)
        # Landau band deficit
        for s in (1, -1):
            msk = (v*s >= 0.04) & (v*s <= 0.2)
            print(f"  f ratio rsm/ctrl mean [{'+' if s>0 else '-'}0.04,0.2]: "
                  f"{d10[msk].mean():.4f}  min {d10[msk].min():.4f} "
                  f"max {d10[msk].max():.4f}")
    else:
        ax.semilogy(v, fr0, label="t=0")
        ax.semilogy(v, fr1, label="end")
        ax.set(xlim=(-0.35, 0.35), title="f(vpar) rsm (no ctrl yet)",
               xlabel=r"$v_\parallel/c$")
        ax.legend(fontsize=8)
    fig2.savefig(f"{a.pref}_energetics.png", dpi=130)
    print(f"fig -> {a.pref}_energetics.png")


if __name__ == "__main__":
    sys.exit(main())
