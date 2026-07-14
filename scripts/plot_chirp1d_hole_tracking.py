#!/usr/bin/env python3
"""Phase-space-hole tracking for a chirp1d rising-tone run (Tao GRL17 Fig 3+):
in a NARROW field-aligned window (single local Omega), histogram (zeta, u_par)
with the zeta-average removed at each u_par to isolate the trapped-island
modulation, and track the hole center against the resonant momentum
u_R(omega(t)) computed from the measured ridge frequency at the same location.

Usage: plot_chirp1d_hole_tracking.py <prefix> [--times 1000,1200,1400,1600]
       [--hcen 40] [--hwid 8] [--probe-k 3] [--uperp 0.5,0.8] [--out f.png]

Requires phase dumps with the 9-float record (x,ux,uy,uz,wd,By,Bz,Ey,Ez).
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def read_dumps(path, rec=9):
    raw = np.fromfile(path, dtype=np.float32)
    recs, i = [], 0
    while i + 2 <= len(raw):
        t0, npk = raw[i], int(raw[i + 1])
        i += 2
        if npk <= 0 or i + rec * npk > len(raw):
            break
        recs.append((float(t0), raw[i:i + rec * npk].reshape(npk, rec)))
        i += rec * npk
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--times", default="1000,1200,1400,1600")
    ap.add_argument("--hcen", type=float, default=40.0)
    ap.add_argument("--hwid", type=float, default=8.0)
    ap.add_argument("--probe-k", type=int, default=3,
                    help="probe index used for the ridge omega(t)")
    ap.add_argument("--uperp", default="0.5,0.8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = read_meta(args.prefix + "_meta.txt")
    wpe = float(meta["wpe"])
    a = float(meta["a"])
    dt = float(meta["dt"])
    stride = int(meta["probe_stride"])
    nprobe = int(meta["nprobe"])
    up0, up1 = (float(v) for v in args.uperp.split(","))
    times = [float(v) for v in args.times.split(",")]

    # ---- ridge omega(t) at the chosen probe ----
    d = np.fromfile(args.prefix + "_probes.bin", dtype=np.float32)
    ns = len(d) // (nprobe * 4)
    d = d[: ns * nprobe * 4].reshape(ns, nprobe, 4)
    dtp = stride * dt
    s = d[:, args.probe_k, 0].astype(float) + 1j * d[:, args.probe_k, 1].astype(float)
    npg, hop = 1024, 64
    win = np.hanning(npg)
    nw = (len(s) - npg) // hop
    S = np.empty((npg, nw), complex)
    for j in range(nw):
        S[:, j] = np.fft.fft(s[j * hop: j * hop + npg] * win)
    f = np.fft.fftfreq(npg, d=dtp) * 2 * np.pi
    t = (np.arange(nw) * hop + npg / 2) * dtp
    sel = (f >= 0.15) & (f <= 0.95)
    fs, P = f[sel], np.abs(S[sel]) ** 2

    def ridge_at(tq):
        return fs[np.argmax(P[:, np.argmin(abs(t - tq))])]

    Wloc = 1.0 + a * args.hcen**2

    def kloc(w):
        return np.sqrt(w * w + w * wpe * wpe / (Wloc - w))

    def uR(w, uperp=0.65):
        v = -0.2
        for _ in range(60):
            g = 1.0 / np.sqrt(1 - min(v * v + (uperp / np.sqrt(1 + uperp**2))**2, 0.98))
            v = (w - Wloc / g) / kloc(w)
        g = 1.0 / np.sqrt(1 - min(v * v + (uperp / np.sqrt(1 + uperp**2))**2, 0.98))
        return v * g

    recs = read_dumps(args.prefix + "_phase.bin")
    fig, axes = plt.subplots(2, len(times) + 1, figsize=(4.0 * (len(times) + 1), 7),
                             squeeze=False)
    hole_t, hole_u = [], []
    h0, h1 = args.hcen - args.hwid / 2, args.hcen + args.hwid / 2
    for c, tw in enumerate(times):
        t0, dd = recs[int(np.argmin([abs(tt - tw) for tt, _ in recs]))]
        h, ux, uy, uz, wd = dd[:, 0], dd[:, 1], dd[:, 2], dd[:, 3], dd[:, 4]
        Byp, Bzp = dd[:, 5], dd[:, 6]
        up = np.hypot(uy, uz)
        slc = (h > h0) & (h < h1) & (up > up0) & (up < up1)
        zeta = np.mod(np.arctan2(uz[slc], uy[slc]) - np.arctan2(Bzp[slc], Byp[slc]),
                      2 * np.pi)
        w_om = ridge_at(t0)
        u_res = uR(w_om)

        for r, wgt in [(0, None), (1, wd[slc])]:
            H, xe, ye = np.histogram2d(zeta / np.pi, ux[slc], bins=[40, 56],
                                       range=[[0, 2], [-0.4, 0.1]], weights=wgt)
            Hm = H - H.mean(axis=0, keepdims=True)
            ax = axes[r, c]
            lim = np.percentile(np.abs(Hm), 99) or 1
            ax.pcolormesh(xe, ye, Hm.T, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.axhline(u_res, color="g", lw=1.2, ls="--")
            ax.set_xlabel(r"$\zeta/\pi$")
            if c == 0:
                ax.set_ylabel(r"$u_\parallel/c$")
            if r == 0:
                ax.set_title(f"t={t0:.0f}  $\\omega$={w_om:.2f}", fontsize=10)
            else:
                # hole center from the zeta ~ pi band of the delta-f modulation
                band = (xe[:-1] + np.diff(xe) / 2 > 0.6) & (xe[:-1] + np.diff(xe) / 2 < 1.4)
                prof = Hm[band].mean(axis=0)
                uc = ye[:-1] + np.diff(ye) / 2
                m = (uc > -0.3) & (uc < 0.05)
                hole_t.append(t0)
                hole_u.append(uc[m][np.argmin(prof[m])])

    axes[0, 0].text(-0.34, 0.5, r"counts $-\langle\rangle_\zeta$",
                    transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.34, 0.5, r"$\delta f-\langle\delta f\rangle_\zeta$",
                    transform=axes[1, 0].transAxes, rotation=90, va="center")

    ax = axes[0, len(times)]
    tt = np.linspace(min(times) - 100, max(times) + 150, 60)
    ax.plot(tt, [uR(ridge_at(x)) for x in tt], "g-", label=r"$u_R(\omega(t))$")
    ax.plot(hole_t, hole_u, "ko--", label="hole center")
    ax.set_xlabel(r"$t\,\Omega_{e0}$"); ax.set_ylabel(r"$u_\parallel/c$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("hole tracks the sweeping resonance", fontsize=10)
    axes[1, len(times)].axis("off")

    fig.suptitle(f"(ζ,u∥) at h∈[{h0:.0f},{h1:.0f}] (Ω_loc={Wloc:.2f}), "
                 f"u⊥∈[{up0},{up1}], ζ-average removed")
    fig.tight_layout()
    out = args.out or args.prefix + "_hole_tracking.png"
    fig.savefig(out, dpi=135)
    print("hole centers:", [f"{u:.3f}" for u in hole_u])
    print("wrote", out)


if __name__ == "__main__":
    main()
