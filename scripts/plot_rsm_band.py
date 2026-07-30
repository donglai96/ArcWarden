#!/usr/bin/env python3
"""RSM V4 band-run analysis (rsm_band outputs).

Usage: python3 scripts/plot_rsm_band.py <outdir> [png_prefix]

Panels:
  1. energy: WB (m0) and 2*W1 (m1) vs time — growth engaged in both channels
  2. m0 omega-k spectrum from bline By+iBz (signed k_par, whistler branch)
  3. m1 omega-k spectrum from m1line B1z (complex coefficient — signed omega
     directly; V6 forensics: P1 shown SEPARATELY from P0, never mixed)
  4. f(vpar) evolution (fhist) — Landau plateau monitor at +-Vp
  5. |E1x| (m1 E_par) x-t map — the Landau channel activity
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_meta(d):
    m = {}
    for line in open(os.path.join(d, 'meta.txt')):
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in ('species', 'probe_ix'):
            try: m[parts[0]] = float(parts[1])
            except ValueError: pass
    return m

def main():
    d = sys.argv[1]
    pref = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, 'rsm_band')
    m = read_meta(d)
    nx = int(m['nx']); dx = m['dx']; dt = m['dt']
    ble = int(m['bline_every']); wce = m['wce']; k1 = m['rsm_k1']
    nb = int(m['nb']); vmax = m['vmax']

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # --- energy ---
    e = np.genfromtxt(os.path.join(d, 'energy.csv'), delimiter=',', names=True)
    ax = axs[0, 0]
    ax.semilogy(e['time'] * wce, e['WB'], label='WB (m0)')
    ax.semilogy(e['time'] * wce, e['W1'], label='2W1 (m1)')
    ax.set_xlabel('t $\\Omega_e$'); ax.legend(); ax.set_title('field energy')

    # --- line stacks ---
    bl = sorted(glob.glob(os.path.join(d, 'bline_*.bin')))
    ml = sorted(glob.glob(os.path.join(d, 'm1line_*.bin')))
    nt = len(bl)
    B0c = np.zeros((nt, nx), complex)   # m0 By+iBz
    B1z = np.zeros((nt, nx), complex)   # m1 B1z coefficient
    E1x = np.zeros((nt, nx), complex)
    for s, f in enumerate(bl):
        a = np.fromfile(f, dtype=np.float32)
        B0c[s] = a[:nx] + 1j * a[nx:2*nx]
    for s, f in enumerate(ml):
        a = np.fromfile(f, dtype=np.float32).reshape(4, nx, 2)
        B1z[s] = a[1, :, 0] + 1j * a[1, :, 1]
        E1x[s] = a[2, :, 0] + 1j * a[2, :, 1]

    tw = np.hanning(nt)[:, None]
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2 * np.pi
    w = np.fft.fftshift(np.fft.fftfreq(nt, ble * dt)) * 2 * np.pi
    wsel = (w >= 0) & (w <= 1.5 * wce)

    def wk(F):
        S = np.fft.fftshift(np.fft.fft2(F * tw))
        return np.abs(S[np.ix_(wsel, np.ones(nx, bool))]) ** 2

    for ax, F, ttl in ((axs[0, 1], B0c, 'P0: m0 By+iBz'),
                       (axs[0, 2], B1z, 'P1: m1 B1z (separate!)')):
        P = wk(F)
        ax.pcolormesh(k, w[wsel] / wce, np.log10(P + 1e-30), cmap='inferno',
                      shading='auto')
        ax.axhline(0.5, color='c', lw=0.5, ls='--')
        ax.set_xlim(-2, 2); ax.set_xlabel('$k_\\parallel$ c/$\\omega_{pe}$')
        ax.set_ylabel('$\\omega/\\Omega_e$'); ax.set_title(ttl)

    # --- f(vpar) ---
    fh = sorted(glob.glob(os.path.join(d, 'fhist_*.bin')))
    ax = axs[1, 0]
    v = np.linspace(-vmax, vmax, nb, endpoint=False) + vmax / nb
    for f in [fh[0], fh[len(fh)//2], fh[-1]] if len(fh) >= 3 else fh:
        h = np.fromfile(f, dtype=np.float64)
        ax.semilogy(v, h + 1e-12, label=os.path.basename(f)[6:12])
    ax.set_xlim(-0.3, 0.3); ax.set_xlabel('$v_\\parallel/c$')
    ax.legend(fontsize=7); ax.set_title('f(vpar) — plateau monitor')

    # --- |E1x| x-t ---
    ax = axs[1, 1]
    tt = np.arange(nt) * ble * dt * wce
    ax.pcolormesh(np.arange(nx) * dx, tt, np.log10(np.abs(E1x) + 1e-30),
                  cmap='viridis', shading='auto')
    ax.set_xlabel('x c/$\\omega_{pe}$'); ax.set_ylabel('t $\\Omega_e$')
    ax.set_title('log10 |E1x| (m1 E$_\\parallel$)')

    # --- m1 k spectrum growth ---
    ax = axs[1, 2]
    for frac, lab in ((0.2, 'early'), (0.6, 'mid'), (1.0, 'late')):
        s = min(nt - 1, int(frac * nt) - 1)
        ax.semilogy(k, np.abs(np.fft.fftshift(np.fft.fft(B1z[s]))) ** 2 + 1e-30,
                    label=lab)
    ax.set_xlim(-2, 2); ax.set_xlabel('$k_\\parallel$'); ax.legend()
    ax.set_title('|B1z(k)|$^2$ snapshots')

    fig.suptitle(f'{d}  (k1={k1:.3f}, wce={wce})')
    fig.tight_layout()
    out = pref + '_summary.png'
    fig.savefig(out, dpi=130)
    print('wrote', out)

if __name__ == '__main__':
    main()
