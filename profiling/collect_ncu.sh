#!/usr/bin/env bash
# Nsight Compute collection for the paper (run from build/; needs perf-counter
# permission — see README.md; typically: sudo ./collect_ncu.sh).
# Produces profiling/yee_tiled_kernels.ncu-rep and profiling/darwin_kernels.ncu-rep.
set -euo pipefail
cd "$(dirname "$0")/../build"
NCU=/usr/local/cuda-13.3/bin/ncu

# Fused tiled particle kernel (the paper's central kernel) + Yee updates.
$NCU --set full --launch-count 3 \
     -k 'regex:k_push_esirkepov_tiled|k_binomial|yee' \
     -o ../profiling/yee_tiled_kernels --force-overwrite \
     ./eaw2d_yee ../decks/eaw_case7_tiled.ini ncu_scratch_yee --tend=1

# Darwin branch: deposit + push + moment kernels.
$NCU --set full --launch-count 3 \
     -k 'regex:deposit|push|amu|dcu' \
     -o ../profiling/darwin_kernels --force-overwrite \
     ./arcsim ../decks/darwin_bench.ini ncu_scratch_darwin --nsteps=3
echo "done: profiling/*.ncu-rep"
