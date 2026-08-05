#!/usr/bin/env python3
"""f(v_par) at the equator from ckpt.bin particle state (plan sec 5 particle
condition). Parses the ARCW checkpoint manifest, memmaps px/pux/puy/puz/pw,
selects |x_phys| < XCUT around the equator, histograms w-weighted v_par=ux/gamma.
Usage: ckpt_fvpar.py DIR [DIR...]  -> per-dir npz + printed manifest info."""
import sys, struct
import numpy as np

def parse_header(fp):
    b = fp.read(4+4+8+8+8+8+8)
    magic, ver, seed, step, time, sd, eps = struct.unpack("<IIqqddd", b)
    assert magic == 0x57435241, hex(magic)
    for _ in range(2):                       # deck, git_hash
        n, = struct.unpack("<Q", fp.read(8)); fp.seek(n, 1)
    na, = struct.unpack("<I", fp.read(4))
    man = []
    for _ in range(na):
        nl, = struct.unpack("<I", fp.read(4))
        name = fp.read(nl).decode()
        dt, cnt = struct.unpack("<IQ", fp.read(12))
        man.append((name, dt, cnt))
    return step, time, man, fp.tell()

DTS = {0: 4, 1: 8, 2: 4, 4: 8}
XCUT = 150.0     # +- around equator, physical units (~ +-145 probe arc)

def fvpar(path, dx, nx, bins):
    with open(path, "rb") as fp:
        step, time, man, off0 = parse_header(fp)
    offs = {}
    off = off0
    for name, dt, cnt in man:
        offs[name] = (off, dt, cnt)
        off += DTS[dt] * cnt
    np_ = offs["px"][2]
    def mm(name):
        o, dt, cnt = offs[name]
        return np.memmap(path, dtype=np.float32, mode="r", offset=o, shape=(cnt,))
    xeq = nx / 2.0
    hw = np.zeros(len(bins) - 1)
    CH = 50_000_000
    px, ux, uy, uz, pw = mm("px"), mm("pux"), mm("puy"), mm("puz"), mm("pw")
    nsel = 0
    for i0 in range(0, np_, CH):
        sl = slice(i0, min(i0 + CH, np_))
        x = px[sl]
        m = np.abs((x.astype(np.float64) - xeq) * dx) < XCUT
        if not m.any(): continue
        a, b, c = ux[sl][m].astype(np.float64), uy[sl][m].astype(np.float64), uz[sl][m].astype(np.float64)
        w = pw[sl][m].astype(np.float64)
        v = a / np.sqrt(1.0 + a*a + b*b + c*c)
        hw += np.histogram(v, bins=bins, weights=w)[0]
        nsel += m.sum()
    return step, time, np_, nsel, hw

if __name__ == "__main__":
    dx, nx = 5041.92/19392, 19392
    bins = np.linspace(-0.3, 0.3, 401)
    for d in sys.argv[1:]:
        step, time, np_, nsel, h = fvpar(f"{d}/ckpt.bin", dx, nx, bins)
        np.savez(f"{d}/fvpar_eq.npz", bins=bins, f=h)
        print(f"{d}: step={step} t={time*0.2:.0f}/We0 markers={np_/1e6:.0f}M eq-sel={nsel/1e6:.1f}M -> fvpar_eq.npz")
