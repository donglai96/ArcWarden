#!/usr/bin/env python3
"""Statistical non-regression gate for the Chen 2026 case-2 chirp path.

GPU float atomics make runs non-deterministic bit-wise (verified
2026-07-24: two runs of the SAME binary differ from step 1), so the gate
is STATISTICAL EQUIVALENCE: the candidate-vs-reference envelope must not
exceed the same-build run-to-run envelope by more than MARGIN.

Protocol (three short runs, ~seconds each):
    ./chirp2d decks/chen2026_case2.ini A   --fullf --nsteps=5000   # ref build
    ./chirp2d decks/chen2026_case2.ini A2  --fullf --nsteps=5000   # ref build again
    ./chirp2d decks/chen2026_case2.ini B   --fullf --nsteps=5000   # candidate build
    python3 scripts/regress_case2.py A A2 B

Baseline envelope at 5000 steps (2026-07-24, RTX 5090): WE ~4e-6,
WB ~1.3e-5, bline rms ~3.6e-4, probe rms ~1.9e-4. The envelope GROWS
with nsteps (linear-stage amplification of atomic-ordering noise) — always
regenerate the same-build envelope, never reuse these numbers at other
step counts.
"""
import sys, glob, os
import numpy as np

MARGIN = 3.0  # candidate envelope may exceed same-build envelope by this factor


def envelope(d1, d2):
    out = {}
    a = np.genfromtxt(f'{d1}/energy.csv', delimiter=',', names=True)
    b = np.genfromtxt(f'{d2}/energy.csv', delimiter=',', names=True)
    out['WE'] = float(np.max(np.abs(a['WE'] - b['WE']) / np.abs(a['WE'])))
    out['WB'] = float(np.max(np.abs(a['WB'] - b['WB']) / np.abs(a['WB'])))
    worst = 0.0
    for f in sorted(glob.glob(f'{d1}/bline_*.bin')):
        x = np.fromfile(f, dtype=np.float32)
        y = np.fromfile(f'{d2}/' + os.path.basename(f), dtype=np.float32)
        n = np.linalg.norm(x)
        if n > 0:
            worst = max(worst, float(np.linalg.norm(x - y) / n))
    out['bline_rms'] = worst
    x = np.fromfile(f'{d1}/probe.bin', dtype=np.float32)
    y = np.fromfile(f'{d2}/probe.bin', dtype=np.float32)
    out['probe_rms'] = float(np.linalg.norm(x - y) / np.linalg.norm(x))
    return out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    ref, ref2, cand = sys.argv[1:4]
    same = envelope(ref, ref2)
    cross = envelope(ref, cand)
    ok = True
    print(f'{"metric":>10s} {"same-build":>12s} {"candidate":>12s} {"ratio":>8s}')
    for k in same:
        ratio = cross[k] / max(same[k], 1e-300)
        flag = '' if ratio <= MARGIN else '  <-- FAIL'
        if ratio > MARGIN:
            ok = False
        print(f'{k:>10s} {same[k]:12.3e} {cross[k]:12.3e} {ratio:8.2f}{flag}')
    print('PASS: candidate statistically indistinguishable from reference'
          if ok else f'FAIL: candidate exceeds {MARGIN}x same-build envelope')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
