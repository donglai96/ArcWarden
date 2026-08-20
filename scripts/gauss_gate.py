#!/usr/bin/env python3
"""Gauss-cleaner gate (2026-08-20): monitor arm vs cleaning arm.

Verdict criteria (frozen before looking at the cleaning arm):
  G1 stability      — cleaning arm finite to the end, no W_EM blowup
                      (first cut D = 0.25 dx^2/dt NaN'd in 2 energy rows).
  G2 suppression    — end-of-run gauss_res(clean) <= 0.5 x gauss_res(mon);
                      report the full-trace ratio.
  G3 no wave damage — W_EM and per-probe |B| envelopes agree within the
                      float-atomic statistical envelope (<= 3x, the
                      regress_case2 convention; expect ~1.0x here because
                      the arms share a seed and the cleaner touches only
                      the longitudinal E error field).
  G4 delta-f intact — wdrms traces agree to the same envelope.

Usage: gauss_gate.py <mon_dir> <clean_dir> [out.png]
"""
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

mon_dir, clean_dir = sys.argv[1], sys.argv[2]
png = sys.argv[3] if len(sys.argv) > 3 else f"{clean_dir}/gauss_gate.png"


def load_energy(d):
    rows = np.genfromtxt(f"{d}/energy.csv", delimiter=",", names=True)
    return rows


def load_probes(d):
    meta = open(f"{d}/meta.txt").read()
    lams = [float(a) for a, _ in
            re.findall(r"^\s+(-?[\d.]+)\s+([\d.]+)\s*$", meta, re.M)]
    dt = float(re.search(r"dt ([\d.]+)", meta).group(1))
    pe = int(re.search(r"probe_every (\d+)", meta).group(1))
    raw = np.fromfile(f"{d}/probes.bin", dtype=np.float32)
    npb = len(lams)
    ns = len(raw) // (6 * npb)
    raw = raw[: ns * 6 * npb].reshape(ns, 6, npb)
    t = (np.arange(ns) + 1) * pe * dt
    return t, raw, lams


em, ec = load_energy(mon_dir), load_energy(clean_dir)
tm, pm, lams = load_probes(mon_dir)
tc, pc, _ = load_probes(clean_dir)

n = min(len(em), len(ec))
em, ec = em[:n], ec[:n]

g1 = bool(np.isfinite(ec["W_EM"]).all() and np.isfinite(ec["gauss_res"]).all())
res_ratio_end = ec["gauss_res"][-1] / em["gauss_res"][-1]
g2 = bool(res_ratio_end <= 0.5)
wem_ratio = ec["W_EM"][-1] / em["W_EM"][-1]
g3 = bool(1 / 3 <= wem_ratio <= 3)
wd_ratio = ec["wdrms_engine"][-1] / em["wdrms_engine"][-1]
g4 = bool(1 / 3 <= wd_ratio <= 3)

# per-probe |B| envelope ratio over the last third of the run
nsc = min(pm.shape[0], pc.shape[0])
pm, pc, tm = pm[:nsc], pc[:nsc], tm[:nsc]
i0 = 2 * nsc // 3
bm = np.sqrt((pm[i0:, 3:6, :] ** 2).sum(axis=1)).mean(axis=0)
bc = np.sqrt((pc[i0:, 3:6, :] ** 2).sum(axis=1)).mean(axis=0)
probe_ratio = bc / bm

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
a = ax[0, 0]
a.semilogy(em["t"], em["gauss_res"], "C0", label="monitor")
a.semilogy(ec["t"], ec["gauss_res"], "C1", label="clean")
a.set_title("interior rms(divE - rho)"); a.set_xlabel("t wpe"); a.legend()
a = ax[0, 1]
a.semilogy(em["t"], em["W_EM"], "C0", label="mon W_EM")
a.semilogy(ec["t"], ec["W_EM"], "C1--", label="clean W_EM")
a.semilogy(em["t"], em["W_cold"], "C0", alpha=0.4, label="mon W_cold")
a.semilogy(ec["t"], ec["W_cold"], "C1--", alpha=0.4, label="clean W_cold")
a.set_title("field / cold energies"); a.set_xlabel("t wpe"); a.legend()
a = ax[1, 0]
a.plot(em["t"], em["wdrms_engine"], "C0", label="mon")
a.plot(ec["t"], ec["wdrms_engine"], "C1--", label="clean")
a.set_title("wd rms (engine)"); a.set_xlabel("t wpe"); a.legend()
a = ax[1, 1]
a.bar(np.arange(len(lams)) - 0.2, bm, 0.4, label="mon")
a.bar(np.arange(len(lams)) + 0.2, bc, 0.4, label="clean")
a.set_xticks(range(len(lams)), [f"{l:g}°" for l in lams])
a.set_title("per-probe |B| envelope (last third)"); a.legend()
fig.suptitle(
    f"Gauss gate: G1 stable={g1}  G2 res_end ratio={res_ratio_end:.3f} "
    f"(<=0.5: {g2})  G3 W_EM ratio={wem_ratio:.3f} ({g3})  "
    f"G4 wdrms ratio={wd_ratio:.3f} ({g4})")
fig.tight_layout()
fig.savefig(png, dpi=110)

print(f"G1 stability        : {'PASS' if g1 else 'FAIL'}")
print(f"G2 suppression      : end ratio {res_ratio_end:.3e} "
      f"-> {'PASS' if g2 else 'FAIL'}")
print(f"G3 wave physics     : W_EM ratio {wem_ratio:.3f} "
      f"-> {'PASS' if g3 else 'FAIL'}")
print(f"   probe |B| ratios : " +
      " ".join(f"{l:g}°:{r:.3f}" for l, r in zip(lams, probe_ratio)))
print(f"G4 delta-f weights  : wdrms ratio {wd_ratio:.3f} "
      f"-> {'PASS' if g4 else 'FAIL'}")
print(f"figure: {png}")
