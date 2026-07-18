#!/usr/bin/env python3
"""M4 chirping analysis for chirp2d runs (2D Yee code path).

Four panels:
  (a) omega-t spectrogram (STFT of By+iBz at the equator probe, R-mode =
      positive frequencies) -- the rising-tone money plot
  (b) omega-k dispersion from the early (antenna-driven) window vs the cold
      whistler branch at the equatorial wce
  (c) x-t map of |Bperp| (packet propagation + boundary check)
  (d) field energy WB(t) + wd rms

Usage: plot_chirp2d.py <rundir> [--tmax T] [--out fig.png]
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_meta(rd):
    m = {}
    for line in open(os.path.join(rd, "meta.txt")):
        k, v = line.split(None, 1)
        if k == "probe_ix":
            m.setdefault("probe_ix", []).append(int(v))
        else:
            m[k] = float(v) if ("." in v or "e" in v or "E" in v) else int(v)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = a.rundir
    m = read_meta(rd)
    nx, dx, dt = int(m["nx"]), m["dx"], m["dt"]
    wce, nprobe = m["wce"], int(m["nprobe"])
    pev, bev = int(m["probe_every"]), int(m["bline_every"])

    # ---- probes -> STFT spectrogram (By + i Bz, R-mode at +omega)
    pr = np.fromfile(os.path.join(rd, "probe.bin"), dtype=np.float32)
    pr = pr.reshape(-1, 2 * nprobe)
    ic = nprobe // 2                       # equator probe
    sig = pr[:, 2 * ic] + 1j * pr[:, 2 * ic + 1]
    dts = pev * dt                         # probe cadence
    if a.tmax:
        sig = sig[: int(a.tmax / dts)]
    nwin = min(4096, 2 ** int(np.log2(max(len(sig) // 8, 64))))
    hop = nwin // 4
    nseg = max(1, (len(sig) - nwin) // hop)
    spec = np.empty((nwin, nseg))
    win = np.hanning(nwin)
    for s in range(nseg):
        seg = sig[s * hop : s * hop + nwin] * win
        spec[:, s] = np.abs(np.fft.fft(seg)) ** 2
    freqs = np.fft.fftfreq(nwin, dts) * 2 * np.pi     # rad/time
    tseg = (np.arange(nseg) * hop + nwin // 2) * dts
    pos = freqs > 0
    fpos = freqs[pos] / wce                            # in units of wce
    sp = spec[pos][np.argsort(fpos)]
    fpos = np.sort(fpos)

    # ---- blines -> omega-k dispersion (early window) + x-t |Bperp|
    files = sorted(glob.glob(os.path.join(rd, "bline_*.bin")))
    nlines = len(files)
    dtb = bev * dt
    stride = max(1, nlines // 1200)                    # cap x-t map size
    bp_xt, tl = [], []
    for i in range(0, nlines, stride):
        d = np.fromfile(files[i], dtype=np.float32)
        bp_xt.append(np.abs(d[:nx] + 1j * d[nx:]))
        tl.append((i + 1) * dtb * stride / stride * 1)
    bp_xt = np.array(bp_xt)
    tl = (np.arange(0, nlines, stride) + 1) * dtb

    # dispersion window: t in [500, 3000] (antenna-driven), central +-400 c/wpe
    i0, i1 = int(500 / dtb), min(int(3000 / dtb), nlines)
    xw = int(400 / dx)
    xc = nx // 2
    blk = []
    for i in range(i0, i1):
        d = np.fromfile(files[i], dtype=np.float32)
        blk.append((d[:nx] + 1j * d[nx:])[xc - xw : xc + xw])
    blk = np.array(blk)                                # (nt, nxw)
    blk *= np.hanning(blk.shape[0])[:, None]
    blk *= np.hanning(blk.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(blk))
    kk = np.fft.fftshift(np.fft.fftfreq(blk.shape[1], dx)) * 2 * np.pi
    ww = np.fft.fftshift(np.fft.fftfreq(blk.shape[0], dtb)) * 2 * np.pi

    # cold whistler branch at equatorial wce: w^2 - k^2 - w nc/(w - wce) = 0
    nc = m.get("cold_nc", 0.994)
    kth = np.linspace(0.05, 2.5, 300)
    wth = []
    for k in kth:
        lo, hi = 1e-6, wce - 1e-6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            Dlo = lo * lo - k * k - lo * nc / (lo - wce)
            Dmid = mid * mid - k * k - mid * nc / (mid - wce)
            if Dlo * Dmid <= 0:
                hi = mid
            else:
                lo = mid
        wth.append(0.5 * (lo + hi))
    wth = np.array(wth)

    # ---- energies
    en = np.genfromtxt(os.path.join(rd, "energy.csv"), delimiter=",", names=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a0, a1, a2, a3 = ax.flat

    p = sp / sp.max()
    a0.pcolormesh(tseg, fpos, np.log10(p + 1e-12), cmap="turbo",
                  vmin=-6.5, vmax=-1, shading="auto")
    a0.set_ylim(0, 1.0)
    a0.set_xlabel(r"$t\,\omega_{pe}$"); a0.set_ylabel(r"$\omega/\omega_{ce}$")
    a0.set_title("(a) equator probe STFT of $B_y+iB_z$ (R-mode)")
    a0.axhline(0.25, color="w", lw=0.5, ls="--")
    a0.axhline(0.5, color="w", lw=0.5, ls=":")

    Pk = np.abs(F) ** 2
    a1.pcolormesh(kk, ww / wce, np.log10(Pk / Pk.max() + 1e-12), cmap="turbo",
                  vmin=-7, vmax=0, shading="auto")
    a1.plot(kth, wth / wce, "w--", lw=1, label="cold whistler (eq.)")
    a1.plot(-kth, -wth / wce, "w--", lw=1)
    a1.set_xlim(-2.5, 2.5); a1.set_ylim(-1.05, 1.05)
    a1.set_xlabel(r"$k\,c/\omega_{pe}$"); a1.set_ylabel(r"$\omega/\omega_{ce}$")
    a1.set_title("(b) $\\omega$-$k$, $t\\in[500,3000]$, $|x-x_c|<400$")
    a1.legend(fontsize=8, loc="lower right")

    ext = [0, nx * dx, tl[0], tl[-1]]
    a2.imshow(bp_xt, origin="lower", aspect="auto", cmap="magma",
              extent=ext, norm=LogNorm(vmin=max(bp_xt.max() * 1e-4, 1e-9),
                                       vmax=bp_xt.max()))
    a2.set_xlabel(r"$x\,c/\omega_{pe}$"); a2.set_ylabel(r"$t\,\omega_{pe}$")
    a2.set_title(r"(c) $|B_\perp|(x,t)$")

    a3.semilogy(en["time"], en["WB"], label=r"$W_B$")
    a3.semilogy(en["time"], en["WE"], label=r"$W_E$", alpha=0.7)
    a3r = a3.twinx()
    a3r.semilogy(en["time"], en["wd_rms"], "g:", label="wd rms")
    a3.set_xlabel(r"$t\,\omega_{pe}$"); a3.set_ylabel("field energy")
    a3.set_title("(d) energies + wd rms")
    a3.legend(loc="upper left", fontsize=8)

    wtitle = f"chirp2d {os.path.basename(rd)}: wce={wce}, nh={m.get('nh', 0)}, ppc={int(m.get('ppc', 0))}, deltaf={int(m.get('deltaf', 0))}"
    fig.suptitle(wtitle, fontsize=11)
    fig.tight_layout()
    out = a.out or os.path.join(rd, "chirp2d_summary.png")
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
