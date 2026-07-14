#!/usr/bin/env python3
"""ArcWarden L-shell plan M0 — memory/time budgeter (plan v3 §13).

Estimates device memory and wall-clock for a planned run and REFUSES budgets
above the safety threshold. Numbers follow the plan's amortized accounting:
static marker state 28-36 B plus runtime amortization (sort scratch, migration
queues, injection pool, deposit temporaries, checkpoint staging, solver
workspace) => 45-80 B/marker.

Usage:
  budgeter.py --nx 4096 --ny 1024 --ppc 64 --dt 0.04 --nsteps 300000
  budgeter.py --markers 3e8 --nsteps 3e5           # direct marker count
Options: --bytes-per-marker 60  --vram-gb 32  --guard 0.90
         --sd 1.0 --eps-l 0.0   (recorded scaling metadata, M5.5+)
"""
import argparse
import sys

THROUGHPUT = [("conservative", 1.5e8), ("baseline (M9 target)", 4.0e8),
              ("stretch", 9.0e8)]


def human_time(sec):
    if sec < 3600:
        return f"{sec/60:.1f} min"
    if sec < 172800:
        return f"{sec/3600:.1f} h"
    return f"{sec/86400:.1f} days"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=0)
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--ppc", type=float, default=0)
    ap.add_argument("--markers", type=float, default=0)
    ap.add_argument("--nsteps", type=float, required=True)
    ap.add_argument("--dt", type=float, default=0.0)
    ap.add_argument("--bytes-per-marker", type=float, default=60.0,
                    help="amortized B/marker (plan: 45-80)")
    ap.add_argument("--grid-arrays", type=int, default=24,
                    help="persistent per-cell field/scratch arrays (FP32 eq.)")
    ap.add_argument("--vram-gb", type=float, default=32.0)
    ap.add_argument("--guard", type=float, default=0.90,
                    help="refuse if projected memory exceeds this fraction")
    ap.add_argument("--sd", type=float, default=1.0, help="drift scaling S_d")
    ap.add_argument("--eps-l", type=float, default=0.0,
                    help="cross-L applicability parameter (record only)")
    a = ap.parse_args()

    cells = max(1, a.nx * a.ny)
    markers = a.markers if a.markers > 0 else cells * a.ppc
    if markers <= 0:
        ap.error("give --markers or --nx/--ny/--ppc")

    marker_gb = markers * a.bytes_per_marker / 1e9
    grid_gb = cells * a.grid_arrays * 4 / 1e9
    total_gb = marker_gb + grid_gb
    frac = total_gb / a.vram_gb

    print("ArcWarden budgeter (plan v3 §13)")
    print(f"  grid: {a.nx} x {a.ny} = {cells:.3g} cells;  markers: {markers:.3g}"
          f"  ({markers/cells:.1f}/cell)")
    print(f"  memory: markers {marker_gb:.2f} GB @ {a.bytes_per_marker:.0f} B/marker"
          f" + grid {grid_gb:.2f} GB = {total_gb:.2f} GB"
          f"  ({100*frac:.1f}% of {a.vram_gb:.0f} GB)")
    if a.dt > 0:
        print(f"  time span: {a.nsteps*a.dt:.3g} Omega_e0^-1  (dt={a.dt})")
    print(f"  scaling metadata: S_d={a.sd}  eps_L={a.eps_l}  (MANDATORY in run meta)")
    pu = markers * a.nsteps
    print(f"  particle-updates: {pu:.3g}")
    for name, rate in THROUGHPUT:
        print(f"    {name:24s} {rate:.1e}/s -> {human_time(pu/rate)}")

    if frac > a.guard:
        print(f"REFUSED: projected memory {100*frac:.1f}% exceeds guard"
              f" {100*a.guard:.0f}% — shrink the run or raise --vram-gb")
        return 1
    print("OK: within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
