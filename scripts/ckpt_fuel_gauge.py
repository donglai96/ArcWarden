#!/usr/bin/env python3
"""Fuel-gauge from checkpoints: equatorial hot-electron f(u) across elements.

Parses the M0 checkpoint (checkpoint.hpp schema), reads px + pux/puy/puz
strided, selects equatorial hot particles (|x-xc| < HALFWIN), and reports
the perp/par temperatures, anisotropy A = T_perp/T_par, and reduced
f(u_par) — the free-energy reservoir that elements tap and the lull refills.
"""
import struct
import sys
import numpy as np

DT_SIZE = {0: 4, 1: 8, 2: 4, 3: 8, 4: 8}
DT_NP = {0: np.float32, 1: np.float64, 2: np.int32, 3: np.uint64, 4: np.float32}

def parse_header(fn):
    with open(fn, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        assert magic == 0x57435241, "bad magic"
        rng, step = struct.unpack("<Qq", f.read(16))
        time, sd, eps = struct.unpack("<ddd", f.read(24))
        for _ in range(2):  # deck, git_hash
            n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
        na = struct.unpack("<I", f.read(4))[0]
        man = []
        for _ in range(na):
            nl = struct.unpack("<I", f.read(4))[0]
            name = f.read(nl).decode()
            dtype = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            man.append((name, dtype, count))
        data_start = f.tell()
    return step, time, man, data_start

def offsets(man, data_start):
    off = {}; pos = data_start
    for name, dtype, count in man:
        off[name] = (pos, dtype, count)
        pos += count * DT_SIZE[dtype]
    return off

def read_strided(fn, off, name, stride):
    pos, dtype, count = off[name]
    npdt = DT_NP[dtype]; itemsize = DT_SIZE[dtype]
    n = count // stride
    out = np.empty(n, npdt)
    with open(fn, "rb") as f:
        # read in chunks to keep memory bounded
        CH = 2_000_000
        idx = 0
        for c0 in range(0, n, CH):
            c1 = min(c0 + CH, n)
            # seek+read each strided element is slow; instead read contiguous
            # blocks and subsample. Read stride*(c1-c0) elements from the
            # right file offset.
            f.seek(pos + c0 * stride * itemsize)
            block = np.fromfile(f, npdt, (c1 - c0) * stride)
            out[c0:c1] = block[::stride][: c1 - c0]
    return out

def main():
    fn = sys.argv[1]
    xc = float(sys.argv[2]) if len(sys.argv) > 2 else 650.0
    halfwin = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0   # +-60 c/wpe ~ +-2.6deg
    dx = 0.26
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    step, time, man, ds = parse_header(fn)
    off = offsets(man, ds)
    # position: stored as float cell-x? px is x in cell units * ? Check: loader
    # uses x in CELL units [0,nx). Physical x = px * dx.
    px = read_strided(fn, off, "px", stride)
    xphys = px * dx
    sel = np.abs(xphys - xc) < halfwin
    ux = read_strided(fn, off, "pux", stride)[sel]
    uy = read_strided(fn, off, "puy", stride)[sel]
    uz = read_strided(fn, off, "puz", stride)[sel]
    w = read_strided(fn, off, "pw", stride)[sel] if "pw" in off else np.ones(sel.sum())
    uperp = np.sqrt(uy * uy + uz * uz)
    Tpar = np.average(ux * ux, weights=w)
    Tperp = 0.5 * np.average(uperp * uperp, weights=w)
    A = Tperp / Tpar
    print(f"t={time*0.2:7.1f}/We0  step={step:7d}  N_eq={sel.sum():8d}  "
          f"Tpar={Tpar:.5f}  Tperp={Tperp:.5f}  A=Tperp/Tpar={A:.4f}")
    # return the reduced f(u_par) for plotting
    np.savez(fn + ".fuel.npz", t=time*0.2, ux=ux, uperp=uperp, w=w,
             Tpar=Tpar, Tperp=Tperp, A=A)

if __name__ == "__main__":
    main()
