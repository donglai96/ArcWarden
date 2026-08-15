#!/usr/bin/env python3
"""F0-SCAN frozen verdict tool (task #41, 2026-08-15). FROZEN BEFORE any
scan-arm data exists. v2 after user redirect ("not d120, clear chirping
base"): u_par ladder at fixed A=0.52 on giant_x4_atmo40 (nh 0.0178, the
discrete rising-tone element train); anchor = existing giant_x4_atmo40
run (u_par 0.198). d120_ctrl kept in the tables as tool calibrant only
(its eq omega_stop 0.491 reproduces the density-ladder history).

Stations: verdict station = EQUATOR (continuity with the density-ladder
anchor convention: eq omega_stop reproduces the historical 0.49 exactly;
the +-291 off-eq ridge is burst-polluted on the anchor, tracker spread
0.33 -> disqualified as verdict station, reported as secondary only).

Per-arm pre-registered readings (eq station):
  R1 CLASS: FLOOD if W10_early(t<2000) >= 0.235 AND full-run band peak
     Bw >= 1.2e-2 B0. Calibrated on-disk against BOTH user rulings:
     floods lumorph_L12A5 (0.266, 1.54e-2) / L10A6 (0.245, 1.91e-2) vs
     non-flood anchor d120_ctrl (0.225, ~8e-3) — each bar sits between
     its calibrants. DEAD if no contiguous element (dur >= 300/Oe) AND
     full-run band peak Bw < 1e-3 B0. Else ELEMENT.
  R2 omega_birth: tracker first-frame f (seed per arm, declared below) +
     independent first-strong-frame ridge f. Pre-registered windows (Omura
     corridor + full-f floor 1.3e-3): anchor 0.22-0.28 / up16 0.24-0.30 /
     up14 0.26-0.32 / up12 >=0.30 (floor-gated bottom) / up10 >=0.33
     (0.25 SUPPRESSED th>opt, Lu-style) or DEAD.
  R3 endpoint, TWO measures: (a) omega_top = chirp-tracker max over nwin
     {1024,1536,2048}, converged if spread < 0.01 else quote the 1024/1536
     pair (amendment-v2 lesson: 2048 smears short risers) — per-element
     organization metric; (b) omega_stop = 95th pct of the power-thresholded
     (>100x pre-ignition floor power) ridge over the FULL run — corridor
     endpoint, the measure behind the density-ladder 0.49 anchor number.
     HEADLINE judged on omega_stop. Any arm > 0.55?
  R4 Bw peak (band 0.2-0.75) vs local B_opt(omega_top) from the built-in
     Omura table (phase-lock budget check: stall expected where Bw >~ B_opt).
  R5 cross-arm: omega_birth and omega_top monotone non-decreasing as u_par
     falls (corridor prediction); count violations.
REFRAME (user, 2026-08-15, before any arm data): the goal is NOT crossing
0.55 — it is UNDERSTANDING how the initial distribution controls the waves
in this 1D code. The ladder holds A (linear band, fuel ceiling w_m=0.342)
fixed and moves ONLY the nonlinear corridor -> every cross-arm difference
is corridor/nonlinear control, not linear-band control. Four pre-registered
CONTROL LAWS, each falsifiable per arm:
  L1 IGNITION/BIRTH <- corridor threshold vs noise floor (R2 windows).
  L2 SATURATION <- B_opt(omega), f0-DEPENDENT (R4: Bw_sat/B_opt ~ 1;
     Chen-repro benchmark sat ~ B_opt).
  L3 SWEEP RATE <- amplitude ONLY, f0-FREE (R6: per-element ridge slope /
     Omura Eq.88(local Bw, per-arm branch) — in-house benchmark 0.72-0.73x,
     lre-free; prediction: ratio stays ~0.7 across ALL arms even as f0
     changes = sweep is amplitude-controlled, not f0-controlled).
  L4 TRAIN PERIOD <- bounce clock (R7: ACF recurrence of the |B| envelope
     at station -1201 (south mid-lat; eq envelope is slow-modulation-
     swamped on the anchor, -1201 gives the clean first peak 5660 @0.27);
     parabolic-well T_b ∝ lre/Uperp0 -> predicted period RATIO vs anchor =
     Uperp0(anchor)/Uperp0(arm) = 1.24/1.41/1.65/1.98 for up16/14/12/10,
     i.e. predicted lags ~7000/8000/9350/11200 IF the bounce clock rules;
     a u_par-independent lag falsifies L4 and points at a wave clock).
up10 DEAD = valid corridor datum, not scan failure. No optional stopping:
all four arms run to t_end before any verdict (early flood-stop allowed
ONLY by R1 FLOOD on a completed t2000 segment, mirroring lumorph stop rule).

Usage: python3 fscan_upar_gate.py            # all arms present in cwd
       python3 fscan_upar_gate.py d120_up16  # single arm
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_chen2026_fig1 import load_probe
from lumorph_gate import spec, w10_occupancy, follow, element_from_traj
import io
import contextlib
with contextlib.redirect_stdout(io.StringIO()):   # omura import chatter
    from analyze_chen2026_periods import envelope, acf_period, \
        ridge_segments

# arm -> (u_par, tracker seed f, pre-registered birth window lo/hi (hi=None
# means open-ended), B_opt table {w: B_opt} from omura pre-reg run)
ARMS = {
    "d120_ctrl": (0.198, 0.25, (0.22, 0.28)),       # calibrant only
    "giant_x4_atmo40": (0.198, 0.25, (0.22, 0.28)),
    "a40_up16": (0.160, 0.26, (0.24, 0.30)),
    "a40_up14": (0.140, 0.28, (0.26, 0.32)),
    "a40_up12": (0.120, 0.31, (0.30, None)),
    "a40_up10": (0.100, 0.34, (0.33, None)),
}
BOPT = {  # B_opt/B0 at (w/Oe) per arm — frozen from the pre-reg Omura run
    "d120_ctrl": {0.35: 7.2e-3, 0.45: 5.5e-3, 0.55: 3.6e-3, 0.65: 2.1e-3},
    "giant_x4_atmo40":
                 {0.35: 8.5e-3, 0.45: 6.5e-3, 0.55: 4.4e-3, 0.65: 2.5e-3},
    "a40_up16": {0.35: 8.7e-3, 0.45: 7.4e-3, 0.55: 5.2e-3, 0.65: 3.2e-3},
    "a40_up14": {0.35: 8.3e-3, 0.45: 7.8e-3, 0.55: 5.8e-3, 0.65: 3.6e-3},
    "a40_up12": {0.35: 7.3e-3, 0.45: 8.0e-3, 0.55: 6.4e-3, 0.65: 4.1e-3},
    "a40_up10": {0.35: 5.5e-3, 0.45: 7.7e-3, 0.55: 7.0e-3, 0.65: 4.8e-3},
}
W10BAR = 0.235
FLOOD_BW = 1.2e-2
DEAD_BW = 1e-3

# per-arm Omura branch (atmo40 base: nc 0.9822, nh 0.0178, pure-biMax
# Uperp0, T_perp/T_par = 1.52 fixed) for Eq.88 sweep + bounce clock
NC, RATIO, OE = 0.9822, 1.232883, 0.2
LRE = 3326.26


def uperp0_of(arm):
    return float(np.sqrt(np.pi / 2) * RATIO * ARMS[arm][0])


def sweep_eq88(arm, wt, bw):
    """Omura Eq.88 sweep-rate prediction (Ωe² units) at ridge freq wt,
    local amplitude bw (B0 units), with THIS arm's Uperp0."""
    up0 = uperp0_of(arm)
    wpe = np.sqrt(NC)
    disp = lambda w: w * np.sqrt(1 + wpe ** 2 / (w * (OE - w)))
    w = wt * OE
    dw = 1e-6
    Vg = (2 * dw * OE) / (disp(w + dw * OE) - disp(w - dw * OE))
    xi = np.sqrt(w * (OE - w)) / wpe
    chi = 1 / np.sqrt(1 + xi * xi)
    Vp = chi * xi
    A = wt * wt + Vp * Vp
    VR = (wt * wt - np.sqrt(wt ** 4 + A * (1 - wt * wt - up0 ** 2))) / A * Vp
    gam = 1 / np.sqrt(1 - VR * VR - up0 ** 2)
    s0 = chi * up0 / xi
    s1 = gam * (1 - VR / Vg) ** 2
    return 0.4 * s0 * wt * bw / s1


def bopt_at(arm, w):
    ws = sorted(BOPT[arm])
    return float(np.interp(w, ws, [BOPT[arm][x] for x in ws]))


def station(d, off):
    upar, fseed, (blo, bhi) = ARMS[d]
    m, t, by, bz = load_probe(d, off)
    wce = m["wce"]
    z = by + 1j * bz
    tOe = t * wce
    P1, fs1, tt1 = spec(z, t, wce, 1024)

    w10, nstrong = w10_occupancy(P1, fs1, tt1, 0, 2000)
    w10_late, _ = w10_occupancy(P1, fs1, tt1, 2500, 6000)

    traj = follow(P1, fs1, tt1, fseed, 200)
    lo, hi = element_from_traj(traj)
    el = traj[lo:hi + 1]
    dur = el[-1, 0] - el[0, 0]

    # band-limited (0.2-0.75) envelope, full run
    F = np.fft.fft(z)
    fr = np.fft.fftfreq(len(z), d=t[1] - t[0]) * 2 * np.pi / wce
    F[(fr < 0.2) | (fr > 0.75)] = 0
    env = np.abs(np.fft.ifft(F))
    nbox = max(1, int(60 / wce / (t[1] - t[0])))
    env = np.convolve(env, np.ones(nbox) / nbox, mode="same")
    bw_run = float(env.max() / wce)
    w = (tOe > el[0, 0]) & (tOe < el[-1, 0])
    bw = float(env[w].max() / wce) if w.any() else 0.0

    flood = (w10 >= W10BAR) and (bw_run >= FLOOD_BW)
    dead = (dur < 300) and (bw_run < DEAD_BW)
    cls = "FLOOD" if flood else ("DEAD" if dead else "ELEMENT")

    # independent birth: first frame whose band [0.15,0.70] peak exceeds
    # 10x the run's own pre-ignition floor (first 10 frames median)
    band = (fs1 > 0.15) & (fs1 < 0.70)
    pk = P1[:, band].max(axis=1)
    floor = np.median(pk[:10])
    istr = np.nonzero(pk > 100 * floor)[0]      # 10x amplitude = 100x power
    birth_ridge = float(fs1[band][np.argmax(P1[istr[0], band])]) \
        if len(istr) else np.nan
    birth_trk = float(el[0, 1]) if not dead else np.nan
    bok = (not dead) and (birth_trk >= blo) and \
          (bhi is None or birth_trk <= bhi)

    tops = []
    for nwin in (1024, 1536, 2048):
        Pw, fsw, ttw = spec(z, t, wce, nwin)
        tr = follow(Pw, fsw, ttw, fseed, 200)
        l2, h2 = element_from_traj(tr)
        tops.append(float(tr[l2:h2 + 1][:, 1].max()))
    spread = max(tops) - min(tops)
    wtop = float(np.mean(tops)) if spread < 0.01 else \
        float(np.mean(tops[:2]))
    conv = "converged" if spread < 0.01 else "1024/1536 pair (2048 smear)"

    # (b) corridor endpoint: 95th pct of thresholded ridge, full run
    ridge_f = fs1[band][np.argmax(P1[:, band], axis=1)]
    strong = pk > 100 * floor
    wstop = float(np.percentile(ridge_f[strong], 95)) if strong.any() \
        else np.nan

    return dict(arm=d, off=off, upar=upar, cls=cls, w10=w10,
                w10_late=w10_late, birth=birth_trk, bok=bok,
                birth_ridge=birth_ridge, wtop=wtop, wtops=tops, conv=conv,
                wstop=wstop, spread=spread, bw=bw, bw_run=bw_run, dur=dur,
                dead=dead, nstrong=nstrong, blo=blo, bhi=bhi,
                P=P1, fs=fs1, tt=tt1, el=el)


def analyze(d):
    st = [station(d, off) for off in (0.0, 291.0, -291.0)]
    r = st[0]                                   # verdict station = EQUATOR
    print(f"\n--- {d} (u_par {r['upar']:.3f}) ---")
    for s in st:
        tag = "  <- VERDICT" if s is r else "  (secondary)"
        print(f"  [{s['off']:+6.0f}] W10 {s['w10']:.3f}/{s['w10_late']:.3f}"
              f"  Bw_run {s['bw_run']:.2e}  birth {s['birth']:.3f}  "
              f"stop {s['wstop']:.3f}{tag}")
    print(f"R1 CLASS {r['cls']} (dur {r['dur']:.0f}/Oe)")
    bhi = r["bhi"]
    print(f"R2 birth: tracker {r['birth']:.3f} / ridge "
          f"{r['birth_ridge']:.3f}  pre-reg [{r['blo']:.2f},"
          f"{'inf' if bhi is None else f'{bhi:.2f}'}] -> "
          f"{'IN' if r['bok'] else 'OUT'}")
    print(f"R3a omega_top " + "/".join(f"{x:.3f}" for x in r["wtops"]) +
          f" spread {r['spread']:.4f} -> {r['wtop']:.3f} ({r['conv']})")
    print(f"R3b omega_stop (95th pct ridge, full run) = {r['wstop']:.3f}"
          f"{'  ** CROSSES 0.55 **' if r['wstop'] > 0.55 else ''}")
    if not r["dead"]:
        bo = bopt_at(d, r["wstop"])
        print(f"R4 Bw_pk {r['bw']:.2e} vs B_opt({r['wstop']:.2f}) = "
              f"{bo:.2e}  ratio {r['bw'] / bo:.2f} "
              f"(stall expected when >~1)")

    # L3: sweep rate vs Eq.88 (per-arm branch), eq station
    segs = ridge_segments(d)
    if segs:
        rats = [s["slope"] / sweep_eq88(d, s["w0"], s["bw"]) for s in segs]
        r["sweep_ratio"] = float(np.median(rats))
        print(f"R6 sweep: {len(segs)} risers, median slope "
              f"{np.median([s['slope'] for s in segs]):.2e} Oe^2, "
              f"meas/Eq88 = {r['sweep_ratio']:.2f} "
              f"(range {min(rats):.2f}-{max(rats):.2f}; benchmark 0.72-0.73"
              f" = amplitude-controlled, f0-free)")
    else:
        r["sweep_ratio"] = np.nan
        print("R6 sweep: no qualifying risers")

    # L4: train recurrence vs bounce clock, station -1201, win [500, end]
    toe_e, e_env, _ = envelope(d, poff=-1201.2)
    acf = acf_period(toe_e, e_env, 500, toe_e[-1])
    r["acf"] = acf
    pred = uperp0_of("giant_x4_atmo40") / uperp0_of(d)
    if acf:
        print(f"R7 recurrence ACF lags: "
              + ", ".join(f"{l:.0f} ({v:.2f})" for l, v in acf)
              + f"  [bounce-clock predicted period ratio vs anchor "
              f"{pred:.2f}]")
    else:
        print(f"R7 recurrence: no ACF peak (single element or dead) "
              f"[predicted ratio vs anchor {pred:.2f}]")
    return r


def main(arms):
    res = [analyze(d) for d in arms]
    live = [r for r in res if r["cls"] == "ELEMENT"]
    if len(res) > 1:
        order = sorted(res, key=lambda r: -r["upar"])
        vb = sum(1 for a, b in zip(order, order[1:])
                 if not (np.isnan(a["birth"]) or np.isnan(b["birth"]))
                 and b["birth"] < a["birth"] - 1e-9)
        vt = sum(1 for a, b in zip(order, order[1:])
                 if b["cls"] == a["cls"] == "ELEMENT"
                 and b["wstop"] < a["wstop"] - 1e-9)
        print(f"\nR5 monotonicity (u_par falling): birth violations {vb}, "
              f"stop violations {vt} (pre-reg: 0 expected)")
        crossers = [r["arm"] for r in live if r["wstop"] > 0.55]
        if crossers:
            print(f"HEADLINE: 0.55 crossed by {', '.join(crossers)}")
        elif all(r["wstop"] <= 0.50 for r in live):
            print("HEADLINE: all arms stall <= 0.50 -> pre-registered "
                  "failure reading: high-pass at fixed A=1.5 insufficient "
                  "at x4; next axis = A at best u_par (USER decision)")
        else:
            print("HEADLINE: between 0.50 and 0.55 — report as measured")

    n = len(res)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3.4 * n), squeeze=False,
                            constrained_layout=True)
    for ax, r in zip(axs.flat, res):
        Pl = np.log10(r["P"].T + 1e-30)
        ax.pcolormesh(r["tt"], r["fs"], Pl, cmap="turbo",
                      vmin=Pl.max() - 5, vmax=Pl.max(), shading="auto",
                      rasterized=True)
        if r["cls"] == "ELEMENT":
            ax.plot(r["el"][:, 0], r["el"][:, 1], "m.", ms=1.5)
        ax.axhline(0.5, color="w", lw=1.5)
        ax.axhline(0.55, color="w", lw=0.8, ls=":")
        ax.set_ylabel(r"$\omega/\Omega_e$")
        ax.set_title(f"{r['arm']}  u_par {r['upar']:.3f}  {r['cls']}  "
                     f"birth {r['birth']:.3f} top {r['wtop']:.3f} "
                     f"stop {r['wstop']:.3f} Bw {r['bw']:.1e}", fontsize=10)
    axs.flat[-1].set_xlabel(r"$t\,\Omega_e$")
    fig.savefig("fscan_upar_verdict.png", dpi=130)
    print("fig -> fscan_upar_verdict.png")


if __name__ == "__main__":
    arms = sys.argv[1:] or list(ARMS)
    main(arms)
