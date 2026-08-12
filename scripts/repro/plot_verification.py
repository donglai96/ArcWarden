#!/usr/bin/env python3
"""Paper verification figure (4 panels) from deck runs alone.

Runs decks/verification/two_stream.ini through arcsim (short run for the
growth/phase panels, a 10x longer run for the conservation panels), then plots:
  (a) electric-field energy growth vs the analytic two-stream rate,
  (b) (x, vx) phase space at saturation (trapping vortices),
  (c) relative total-energy drift over the long run,
  (d) relative net-charge drift over the long run.

Usage: plot_verification.py <build_dir> [--out=paper/figs/verification_panels.png]
Existing run dirs are reused; delete <build>/verif_short|verif_long to rerun.
"""
import csv, math, pathlib, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
DECK = ROOT / "decks/verification/two_stream.ini"


def run(build, outdir, extra=()):
    if not (outdir / "energy.csv").exists():
        subprocess.run([str(build / "arcsim"), str(DECK), str(outdir), *extra],
                       check=True)
    with open(outdir / "energy.csv") as f:
        rows = list(csv.DictReader(f))
    get = lambda k: [float(r[k]) for r in rows]
    return {k: get(k) for k in ("time", "ke", "ee", "total", "charge")}


def main():
    build = pathlib.Path(sys.argv[1]).resolve()
    out = ROOT / "paper/figs/verification_panels.png"
    for a in sys.argv[2:]:
        if a.startswith("--out="):
            out = pathlib.Path(a[6:])

    short = run(build, build / "verif_short")
    long_ = run(build, build / "verif_long", ("--nsteps=6000",))

    # (a) growth: fit log EE where EE spans its linear-growth decades
    t, ee = short["time"], short["ee"]
    ee_min, ee_max = ee[0], max(ee)
    lo, hi = 30 * ee_min, ee_max / 30
    win = [(ti, ei) for ti, ei in zip(t, ee) if lo < ei < hi and ti < t[ee.index(ee_max)]]
    (t0, e0), (t1, e1) = win[0], win[-1]
    gamma_fit = 0.5 * math.log(e1 / e0) / (t1 - t0)      # EE ~ exp(2*gamma*t)
    gamma_th = 1.0 / math.sqrt(8.0)                      # cold symmetric beams

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.5))
    a = ax[0][0]
    a.semilogy(t, ee, "k-", lw=1.2, label="ArcWarden $W_E$")
    tt = [t0, t1]
    a.semilogy(tt, [e0, e0 * math.exp(2 * gamma_th * (t1 - t0))], "r--",
               label=rf"analytic $\gamma=\omega_{{pe}}/\sqrt{{8}}$")
    a.set_xlabel(r"$t\,\omega_{pe}$"); a.set_ylabel(r"$W_E$")
    a.set_title(rf"(a) two-stream growth: $\gamma_{{fit}}={gamma_fit:.3f}$ vs ${gamma_th:.3f}$")
    a.legend(fontsize=8)

    # (b) phase space at saturation (last dumped frame)
    frames = sorted((build / "verif_short").glob("frame_*.csv"))
    isat = min(range(len(t)), key=lambda i: abs(ee[i] - ee_max))
    fidx = min(len(frames) - 1, round(isat / max(1, len(t) - 1) * (len(frames) - 1)))
    xs, vs = [], []
    with open(frames[fidx]) as f:
        for row in csv.reader(f):
            try:
                xs.append(float(row[0])); vs.append(float(row[1]))
            except ValueError:
                continue                                  # header
    b = ax[0][1]
    b.plot(xs[::7], vs[::7], ",", color="#00307d", alpha=0.5)
    b.set_xlabel(r"$x/\lambda_D$"); b.set_ylabel(r"$v_x/v_{th}$")
    b.set_title(f"(b) phase space at saturation ({frames[fidx].name})")

    # (c)+(d) conservation over the 10x run
    tl = long_["time"]
    tot0, q0 = long_["total"][0], long_["charge"][0]
    c = ax[1][0]
    c.plot(tl, [(x - tot0) / tot0 for x in long_["total"]], "k-", lw=0.9)
    c.set_xlabel(r"$t\,\omega_{pe}$"); c.set_ylabel(r"$\Delta W_{tot}/W_{tot}(0)$")
    c.set_title("(c) total-energy drift (6000 steps)")
    c.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    d = ax[1][1]
    d.plot(tl, [(x - q0) / abs(q0) for x in long_["charge"]], "k-", lw=0.9)
    d.set_xlabel(r"$t\,\omega_{pe}$"); d.set_ylabel(r"$\Delta Q/|Q(0)|$")
    d.set_title("(d) net-charge drift (6000 steps)")
    d.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle("ArcWarden verification: two-stream deck, electrostatic branch",
                 fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}\n  gamma_fit={gamma_fit:.4f}  gamma_theory={gamma_th:.4f}  "
          f"ratio={gamma_fit/gamma_th:.3f}")


if __name__ == "__main__":
    main()
