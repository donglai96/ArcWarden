#!/usr/bin/env bash
# Regenerate every Section-5 (GPU performance) number from decks alone,
# in one session with one binary. Results -> <outdir>/perf_sweep.csv.
#
# Usage: scripts/repro/perf_sweep.sh <build_dir> [outdir] [--tend=T] [--full]
#   --tend=T   physical end time per run (default 60; smoke ~40 s/point)
#   --full     paper-grade: tend=3000 (case-7 full length; ~35 min for the
#              fused point alone, several hours for the whole sweep)
#
# Sweep points (all derived from decks/benchmarks/eaw_case7_tiled.ini by sed):
#   ablation:  tile_sort=0 | tile_sort=20+tile_migrate_fused=0 | tile_sort=20
#   cadence:   tile_sort = 10 / 20 / 40
# The base deck and derived variants share physics exactly; field-energy
# lines in each run log are the cross-check (they must match bit-for-bit).
set -euo pipefail

BUILD=${1:?usage: perf_sweep.sh <build_dir> [outdir] [--tend=T] [--full]}
OUT=${2:-perf_sweep_out}
TEND=60
for a in "$@"; do
  case "$a" in
    --tend=*) TEND=${a#--tend=} ;;
    --full)   TEND=3000 ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DECK=$ROOT/decks/benchmarks/eaw_case7_tiled.ini
BIN=$BUILD/eaw2d_yee
[ -x "$BIN" ] || { echo "error: $BIN not built" >&2; exit 1; }

mkdir -p "$OUT"
CSV=$OUT/perf_sweep.csv
{
  echo "# git_commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "# binary_md5: $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "# driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "# date: $(date -Is)"
  echo "point,tile_sort,tile_migrate_fused,tend,wall_s,psteps_per_s"
} > "$CSV"

run_point() {  # name tile_sort fused
  local name=$1 ts=$2 fused=$3
  local deck=$OUT/deck_$name.ini
  # tile_migrate_fused must land INSIDE the [field] section (the parser is
  # section-scoped) — insert it on the tile_sort line, never append to EOF.
  sed -e "s/^tile_sort *=.*/tile_sort = $ts\ntile_migrate_fused = $fused/" \
      "$DECK" > "$deck"
  [ "$ts" = 0 ] && sed -i '/^tile_sort\|^tile_migrate_fused/d' "$deck"
  grep -A20 '^\[field\]' "$deck" | grep -q "tile_migrate_fused = $fused" || [ "$ts" = 0 ] || {
    echo "error: knob failed to land in [field] section of $deck" >&2; exit 1; }
  echo "== $name (tile_sort=$ts fused=$fused tend=$TEND)"
  local log=$OUT/$name.log
  "$BIN" "$deck" "$OUT/run_$name" --tend="$TEND" | tee "$log" | tail -1
  # runner prints: "done: <wall> s (<rate> particle-steps/s)"
  local wall rate
  wall=$(grep -oP 'done: \K[0-9.]+' "$log")
  rate=$(grep -oP '\(\K[0-9.e+]+(?= particle-steps/s)' "$log")
  echo "$name,$ts,$fused,$TEND,$wall,$rate" >> "$CSV"
}

# --- ablation chain (paper Table: flat -> +tiled deposit -> +fused migrate)
run_point flat            0  1
run_point tiled_nofuse    20 0
run_point tiled_fused     20 1
# --- sort-cadence dial
run_point cadence10       10 1
run_point cadence40       40 1

echo; echo "== field-energy cross-check (last line of each run):"
for f in "$OUT"/*.log; do printf '%-24s %s\n' "$(basename "$f" .log)" "$(grep -E '^t=' "$f" | tail -1)"; done

# quantified energy-consistency summary: relative difference of total field
# energy vs the tiled_fused reference at sample times, per point
python3 - "$OUT" << 'PY'
import glob, os, sys
out = sys.argv[1]
def hist(name):
    fs = glob.glob(os.path.join(out, f"run_{name}", "*_hist.txt"))
    if not fs: return None
    d = {}
    for ln in open(fs[0]):
        if ln.startswith('#'): continue
        v = ln.split()
        if len(v) < 7: continue
        d[float(v[0])] = sum(float(x) for x in v[1:7])   # total field energy
    return d
ref = hist("tiled_fused")
names = [os.path.basename(f)[4:] for f in glob.glob(os.path.join(out, "run_*"))]
lines = ["# relative |W - W_fused|/W_fused of total field energy, sample times",
         "point," + ",".join(f"t={t:g}" for t in (100, 300, 600, 900, 1500, 2999))]
for n in sorted(names):
    if n == "tiled_fused": continue
    h = hist(n)
    if not h or not ref: continue
    row = [n]
    for t in (100, 300, 600, 900, 1500, 2999):
        tt = min(ref, key=lambda x: abs(x - t)); th = min(h, key=lambda x: abs(x - t))
        row.append(f"{abs(h[th]-ref[tt])/ref[tt]:.2e}")
    lines.append(",".join(row))
open(os.path.join(out, "energy_consistency.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo; column -s, -t "$CSV"
