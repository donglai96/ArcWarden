#!/usr/bin/env python3
"""UB fuel-consumption check (Step-3 TODO, 2026-08-16): species-split
f(v_par) + anisotropy from the expG2_ctrl_r3 final checkpoint (~t15000).

Species split: NOT by index (tile_sort orders the ckpt arrays by cell —
index-block split FALSIFIED empirically: "tail block" is the box right
end, zero eq markers). Split by marker WEIGHT instead: fullf weights are
per-species delta functions, engine w=6.491e-6 vs pancake w=4.467e-6
(ratio 1.453 = (0.0178/28000)/(0.0035/8000) exactly). Threshold 5.5e-6.

Claim under test (Step-3 episodic verdict): pancake free energy is a
finite resource -- QL flattening at the UB-resonant shell
|v_par| ~ 0.05-0.08c and A2 drawdown by t15000. Initial reference is
ANALYTIC (bi-Max x conecut; no t0 ckpt on disk -- caveat recorded).

Usage (from build/): python3 ../scripts/ub_fuel_check.py expG2_ctrl_r3
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../scripts")
from ckpt_fvpar import parse_header, DTS  # noqa: E402

WSPLIT = 5.5e-6          # engine w=6.491e-6 > split > pancake w=4.467e-6
DX, NX = 5041.92 / 19392, 19392
XCUT = 150.0
CH = 50_000_000


def arrays(path):
    with open(path, "rb") as fp:
        step, time, man, off0 = parse_header(fp)
    offs, off = {}, off0
    for name, dt, cnt in man:
        offs[name] = (off, dt, cnt)
        off += DTS[dt] * cnt
    def mm(name):
        o, dt, cnt = offs[name]
        return np.memmap(path, dtype=np.float32, mode="r",
                         offset=o, shape=(cnt,))
    return step, time, mm


def species_stats(mm, engine, bins):
    px, ux, uy, uz, pw = (mm(k) for k in ("px", "pux", "puy", "puz", "pw"))
    xeq = NX / 2.0
    hw = np.zeros(len(bins) - 1)
    s_w = s_vpar2 = s_vper2 = s_upar2 = 0.0
    nsel = 0
    ntot = px.shape[0]
    for a0 in range(0, ntot, CH):
        sl = slice(a0, min(a0 + CH, ntot))
        x = px[sl]
        wsel = (pw[sl] > WSPLIT) if engine else (pw[sl] < WSPLIT)
        m = wsel & (np.abs((x.astype(np.float64) - xeq) * DX) < XCUT)
        if not m.any():
            continue
        a = ux[sl][m].astype(np.float64)
        b = uy[sl][m].astype(np.float64)
        c = uz[sl][m].astype(np.float64)
        w = pw[sl][m].astype(np.float64)
        g = np.sqrt(1.0 + a * a + b * b + c * c)
        v = a / g
        hw += np.histogram(v, bins=bins, weights=w)[0]
        s_w += w.sum()
        s_upar2 += (w * a * a).sum()
        s_vpar2 += (w * v * v).sum()
        s_vper2 += (w * (b * b + c * c) / (g * g)).sum()
        nsel += int(m.sum())
    A_eff = s_vper2 / (2.0 * s_vpar2) - 1.0
    return hw, nsel, s_w, np.sqrt(s_upar2 / s_w), A_eff


def main(d):
    step, time, mm = arrays(f"{d}/ckpt.bin")
    ntot = mm("px").shape[0]
    print(f"{d}: step={step} t={time*0.2:.0f}/We0 markers={ntot/1e6:.1f}M")
    bins = np.linspace(-0.30, 0.30, 401)
    vc = 0.5 * (bins[1:] + bins[:-1])
    out = {}
    for tag, engine in (("engine", True), ("pancake", False)):
        hw, nsel, sw, upar_rms, A_eff = species_stats(mm, engine, bins)
        print(f"  {tag:8s}: eq-sel {nsel/1e6:.1f}M, u_par rms {upar_rms:.4f} "
              f"(loaded {'0.198' if tag=='engine' else '0.060'}), "
              f"A_eff(eq) = {A_eff:.3f} "
              f"(loaded {'0.52' if tag=='engine' else '5.00'})")
        out[tag] = (hw, A_eff)
    # analytic initial pancake f(v_par): bi-Max u_par2=0.06 (conecut acts
    # mostly in pitch angle; v_par marginal shape ~ Maxwellian, caveat)
    f0 = np.exp(-vc ** 2 / (2 * 0.06 ** 2))
    hp = out["pancake"][0]
    sh = np.abs(vc) < 0.20
    f0 *= hp[sh].sum() / f0[sh].sum() * (f0[sh].sum() > 0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax[0].semilogy(vc, hp, "r", label=f"pancake t{time*0.2:.0f}")
    ax[0].semilogy(vc, f0, "k--", label="analytic init (biMax 0.06)")
    for s in (-1, 1):
        ax[0].axvspan(s * 0.05, s * 0.08, color="b", alpha=0.15)
    ax[0].set_xlim(-0.25, 0.25); ax[0].set_xlabel(r"$v_\parallel/c$")
    ax[0].set_title("pancake f(v$_\\parallel$) eq |x|<150 "
                    "(blue = UB-resonant shell)")
    ax[0].legend()
    ax[1].semilogy(vc, out["engine"][0], "g", label="engine")
    ax[1].set_xlim(-0.3, 0.3); ax[1].set_xlabel(r"$v_\parallel/c$")
    ax[1].set_title("engine f(v$_\\parallel$)"); ax[1].legend()
    fig.savefig(f"{d}_fuel_check.png", dpi=130)
    np.savez(f"{d}/fuel_check.npz", bins=bins,
             engine=out["engine"][0], pancake=hp)
    print(f"fig -> {d}_fuel_check.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "expG2_ctrl_r3")
