# Profiling artifacts (paper evidence — open in the GUIs)

- `darwin_step.nsys-rep` — Nsight Systems timeline, Darwin branch
  (`decks/darwin_bench.ini`, 625², 156.25M particles, 300 steps).
  Open: `nsys-ui darwin_step.nsys-rep`
- `yee_tiled_step.nsys-rep` — Nsight Systems timeline, Yee tiled branch
  (`decks/eaw_case7_tiled.ini`, 207 steps). Open with `nsys-ui`.
  Headline numbers (nsys stats): fused `k_push_esirkepov_tiled` = 87.7%
  of GPU time, median 9.3 ms/step; sort 12.1 ms every 20 steps
  (0.64 ms/step); full Maxwell update + 9 filter passes ≈ 0.05 ms/step.
- `*.sqlite` — nsys stats exports of the same reports (queryable).
- `*_util.txt` — 1 Hz `nvidia-smi` utilization/power samples taken during
  the same runs (darwin: 98–100% @ ~485 W once stepping).
- `*.ncu-rep` — Nsight Compute kernel reports (open: `ncu-ui <file>`).
  ncu needs GPU perf-counter permission: either run the collection with
  sudo, or enable for all users once:
    `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo update-initramfs -u`  (then reboot)
  Collection commands are in `collect_ncu.sh`.

Generated 2026-07-16 on RTX 5090 (sm_120, 96 MB L2), CUDA 13.3,
Nsight Systems 2026.1.3. What to look at for the paper (§ Profiling):
- nsys timeline: 4–5 wide kernels per step, no inter-kernel gaps, no
  host round-trips outside diagnostics.
- ncu on `k_push_esirkepov_tiled`: SOL memory/compute %, achieved
  occupancy, **L2 hit rate of the field gathers** (the design-rule-1
  number), shared-memory atomic throughput.
