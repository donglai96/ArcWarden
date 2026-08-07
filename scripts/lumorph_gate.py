#!/usr/bin/env python3
"""Lu-morph phase-2 gate (docs/PLAN_LUMORPH.md): 8 simultaneous criteria +
stop rules, all quantitative.

Metrics:
  1 early broadband: median 10-dB spectral occupancy width W10 (Ωe units)
    over strong frames (P > 10% of window max) in t<2000. Calibrated on
    d120_A20 (known broadband burst) vs quiet cases — bar set by --w10bar.
  2/8 connected element: chirp-following tracker (birth-seeded, ±[0.05,
    +0.06] band follow, parabolic peak interpolation) alive mask must be
    one contiguous stretch >= 300/Ωe with no gap > 5 frames.
  3 birth in [0.28, 0.40]: ridge f at first alive frame.
  4 sweep width >= 0.15: f_end - f_birth.
  5 crosses 0.55 with nwin={1024,1536,2048} spread < 0.01 (ω_stop per
    window, interpolated).
  6 eq element amplitude: band-limited (0.2-0.75) envelope peak in the
    element interval, in [3e-3, 1e-2] B0.
  7 no natural 0.46-0.56 valley in the element window (V with up/hi noise
    guard — V only meaningful if up/hi >= 3, else shape is a monotone
    slope, reported as such).
Usage:
  python3 lumorph_gate.py <dir>                 # full gate
  python3 lumorph_gate.py <dir> --early         # t2000 stop-rule check only
  python3 lumorph_gate.py <dir> --fstart 0.30   # tracker seed frequency
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import load_probe, stft


def spec(z, t, wce, nwin, hop=64):
    S = stft(z, nwin, hop)
    f = np.fft.fftfreq(nwin, d=t[1] - t[0]) * 2 * np.pi / wce
    tt = (t[nwin // 2] + np.arange(S.shape[0]) * hop * (t[1] - t[0])) * wce
    sel = (f > 0.10) & (f < 0.95)
    return np.abs(S[:, sel]) ** 2, f[sel], tt


def w10_occupancy(P, fs, tt, t0, t1):
    """median 10-dB occupancy width over strong frames in [t0,t1]."""
    m = (tt > t0) & (tt < t1)
    if not m.any():
        return np.nan, 0
    Pm = P[m]
    strong = Pm.max(axis=1) > 0.10 * Pm.max()
    if not strong.any():
        return 0.0, 0
    df = fs[1] - fs[0]
    widths = [(row > row.max() / 10).sum() * df for row in Pm[strong]]
    return float(np.median(widths)), int(strong.sum())


def follow(P, fs, tt, fstart, t0):
    fc, traj = fstart, []
    i0 = np.searchsorted(tt, t0)
    for i in range(i0, P.shape[0]):
        win = (fs > fc - 0.05) & (fs < fc + 0.06)
        if not win.any():
            break
        j = np.argmax(P[i, win])
        idx = np.nonzero(win)[0][j]
        fc = fs[idx]
        if 0 < idx < len(fs) - 1:
            a, b, c = (np.log(P[i, idx - 1] + 1e-30),
                       np.log(P[i, idx] + 1e-30),
                       np.log(P[i, idx + 1] + 1e-30))
            d = (a - c) / (2 * (a - 2 * b + c)) if (a - 2 * b + c) != 0 else 0
            fi = fs[idx] + d * (fs[1] - fs[0])
        else:
            fi = fc
        traj.append((tt[i], fi, P[i, idx]))
    return np.array(traj)


def element_from_traj(traj):
    """contiguous alive stretch containing the power max; gap tolerance 5."""
    alive = traj[:, 2] > 0.03 * traj[:, 2].max()
    imax = int(np.argmax(traj[:, 2]))
    lo = hi = imax
    gap = 0
    for i in range(imax + 1, len(traj)):
        gap = 0 if alive[i] else gap + 1
        if gap > 5:
            break
        if alive[i]:
            hi = i
    gap = 0
    for i in range(imax - 1, -1, -1):
        gap = 0 if alive[i] else gap + 1
        if gap > 5:
            break
        if alive[i]:
            lo = i
    return lo, hi


def main(d, early=False, fstart=0.30, w10bar=0.20):
    m, t, by, bz = load_probe(d, 0.0)
    wce = m["wce"]
    z = by + 1j * bz
    tOe = t * wce
    P1, fs1, tt1 = spec(z, t, wce, 1024)

    w10, nstrong = w10_occupancy(P1, fs1, tt1, 0, 2000)
    c1 = w10 < w10bar
    print(f"[1] early (t<2000) 10dB occupancy W10 = {w10:.3f} Ωe over "
          f"{nstrong} strong frames  (bar < {w10bar})  -> "
          f"{'PASS' if c1 else 'FAIL — broadband burst'}")
    if early:
        return

    traj = follow(P1, fs1, tt1, fstart, 200)
    lo, hi = element_from_traj(traj)
    el = traj[lo:hi + 1]
    dur = el[-1, 0] - el[0, 0]
    birth, fend = el[0, 1], el[:, 1].max()
    c2 = dur >= 300
    c3 = 0.28 <= birth <= 0.40
    c4 = (fend - birth) >= 0.15
    print(f"[2] element: t [{el[0,0]:.0f},{el[-1,0]:.0f}] dur {dur:.0f}/Ωe "
          f"(bar >=300, contiguous) -> {'PASS' if c2 else 'FAIL'}")
    print(f"[3] birth f = {birth:.3f} (bar 0.28-0.40) -> "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"[4] sweep = {fend - birth:.3f} (bar >=0.15) -> "
          f"{'PASS' if c4 else 'FAIL'}")

    stops = []
    for nwin in (1024, 1536, 2048):
        Pw, fsw, ttw = spec(z, t, wce, nwin)
        tr = follow(Pw, fsw, ttw, fstart, 200)
        l2, h2 = element_from_traj(tr)
        stops.append(tr[l2:h2 + 1][:, 1].max())
    spread = max(stops) - min(stops)
    c5 = (min(stops) >= 0.55) and (spread < 0.01)
    print(f"[5] f_top(nwin 1024/1536/2048) = "
          + "/".join(f"{x:.4f}" for x in stops)
          + f"  spread {spread:.4f} (bar: all >=0.55, spread <0.01) -> "
          f"{'PASS' if c5 else 'FAIL'}")

    w = (tOe > el[0, 0]) & (tOe < el[-1, 0])
    n = int(w.sum())
    F = np.fft.fft(z[w])
    fr = np.fft.fftfreq(n, d=t[1] - t[0]) * 2 * np.pi / wce
    F[(fr < 0.2) | (fr > 0.75)] = 0
    env = np.abs(np.fft.ifft(F))
    nbox = max(1, int(60 / wce / (t[1] - t[0])))
    env = np.convolve(env, np.ones(nbox) / nbox, mode="same")
    bw = env.max() / wce
    c6 = 3e-3 <= bw <= 1e-2
    print(f"[6] element-window eq Bw/B0 = {bw:.2e} (bar 3e-3..1e-2) -> "
          f"{'PASS' if c6 else 'FAIL'}")

    fr2, Pa = welch(z[w].real, fs=1 / (t[1] - t[0]), nperseg=4096)
    _, Pb = welch(z[w].imag, fs=1 / (t[1] - t[0]), nperseg=4096)
    Pw2 = Pa + Pb
    fo = fr2 * 2 * np.pi / wce
    band = lambda a, b: Pw2[(fo >= a) & (fo < b)].mean()
    loP, gapP, upP, hiP = (band(.35, .45), band(.46, .56),
                           band(.56, .66), band(.70, .80))
    if upP / hiP >= 3:
        V = gapP / np.sqrt(loP * upP)
        c7 = V >= 1.0
        print(f"[7] valley V = {V:.2f} (up/hi {upP/hiP:.1f} — two-sided "
              f"valid; bar V>=1 = no natural valley) -> "
              f"{'PASS' if c7 else 'FAIL'}")
    else:
        c7 = True
        print(f"[7] up/hi = {upP/hiP:.1f} < 3: no upper shoulder yet, shape "
              f"= monotone slope, no valley claim possible -> PASS (n/a)")

    c8 = c2 and c4 and nstrong >= 0   # dot/fake-ridge excluded by 2+4 rules
    print(f"[8] not-a-dot: covered by [2] contiguity + [4] sweep -> "
          f"{'PASS' if c8 else 'FAIL'}")

    allpass = all((c1, c2, c3, c4, c5, c6, c7, c8))
    print(f"\n=== {d}: {'GATE PASS' if allpass else 'GATE FAIL'} ===")

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    Pl = np.log10(P1.T + 1e-30)
    im = ax.pcolormesh(tt1, fs1, Pl, cmap="turbo", vmin=Pl.max() - 5,
                       vmax=Pl.max(), shading="auto")
    ax.plot(el[:, 0], el[:, 1], "m.", ms=2)
    ax.axhline(0.5, color="w", lw=2)
    ax.axhline(0.55, color="w", lw=1, ls=":")
    ax.set_xlabel(r"$t\,\Omega_e$")
    ax.set_ylabel(r"$\omega/\Omega_e$")
    ax.set_title(f"{d}  gate={'PASS' if allpass else 'FAIL'}  "
                 f"birth {birth:.2f} top {max(stops):.3f} Bw {bw:.1e}")
    plt.colorbar(im, ax=ax, label="log10 P")
    fig.savefig(f"{d}_gate.png", dpi=130)
    print(f"fig -> {d}_gate.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    early = "--early" in sys.argv
    fstart = 0.30
    for a in sys.argv:
        if a.startswith("--fstart"):
            fstart = float(a.split("=")[1])
    main(args[0], early=early, fstart=fstart)
