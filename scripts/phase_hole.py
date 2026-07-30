#!/usr/bin/env python3
"""Resonant-electron phase space (zeta, u_par) from a checkpoint — hunt phase holes.

zeta = gyrophase(atan2(uz,uy)) - wave phase(atan2(Bz,By) at particle x).
Trapped electrons cluster/deplete at a fixed zeta near the resonance velocity
v_res = (w - Omega_e)/k  (<0, counter-streaming). A rotating hole in (zeta,u_par)
= electromagnetic electron hole = Omura's chirping engine.

Usage: phase_hole.py <ckpt.bin> <bline.bin> [xc=650] [halfwin=120]
"""
import struct, sys
import numpy as np

DT_SIZE = {0:4,1:8,2:4,3:8,4:8}; DT_NP={0:np.float32,1:np.float64,2:np.int32,3:np.uint64,4:np.float32}

def parse(fn):
    with open(fn,"rb") as f:
        magic,ver=struct.unpack("<II",f.read(8)); assert magic==0x57435241
        f.read(16); f.read(24)
        for _ in range(2):
            n=struct.unpack("<Q",f.read(8))[0]; f.read(n)
        na=struct.unpack("<I",f.read(4))[0]; man=[]
        for _ in range(na):
            nl=struct.unpack("<I",f.read(4))[0]; name=f.read(nl).decode()
            dt=struct.unpack("<I",f.read(4))[0]; c=struct.unpack("<Q",f.read(8))[0]
            man.append((name,dt,c))
        ds=f.tell()
    off={}; pos=ds
    for name,dt,c in man: off[name]=(pos,dt,c); pos+=c*DT_SIZE[dt]
    # also recover time
    with open(fn,"rb") as f:
        f.read(8); f.read(16); import struct as st
        t=st.unpack("<d",f.read(8))[0]
    return off,t

def rd(fn,off,name,stride):
    pos,dt,c=off[name]; npdt=DT_NP[dt]; it=DT_SIZE[dt]; n=c//stride
    out=np.empty(n,npdt)
    with open(fn,"rb") as f:
        CH=4_000_000
        for c0 in range(0,n,CH):
            c1=min(c0+CH,n); f.seek(pos+c0*stride*it)
            blk=np.fromfile(f,npdt,(c1-c0)*stride); out[c0:c1]=blk[::stride][:c1-c0]
    return out

def main():
    ck=sys.argv[1]; bl=sys.argv[2]
    xc=float(sys.argv[3]) if len(sys.argv)>3 else 650.0
    halfwin=float(sys.argv[4]) if len(sys.argv)>4 else 120.0
    dx=0.26; nx=5000; stride=4
    off,t=parse(ck)
    px=rd(ck,off,"px",stride); xphys=px*dx
    sel=np.abs(xphys-xc)<halfwin
    ux=rd(ck,off,"pux",stride)[sel]; uy=rd(ck,off,"puy",stride)[sel]; uz=rd(ck,off,"puz",stride)[sel]
    xp=xphys[sel]
    # wave field at each particle x (nearest grid node)
    b=np.fromfile(bl,dtype=np.float32,count=2*nx); By,Bz=b[:nx],b[nx:]
    gi=np.clip((xp/dx).astype(int),0,nx-1)
    phw=np.arctan2(Bz[gi],By[gi])
    php=np.arctan2(uz,uy)
    zeta=np.mod(php-phw,2*np.pi)
    np.savez(ck+".phase.npz",t=t*0.2,zeta=zeta,ux=ux,uperp=np.sqrt(uy*uy+uz*uz),
             xp=xp-xc)
    print(f"{ck}: t={t*0.2:.0f}/We0  N_res_window={sel.sum()}  |Bw|_eq~{np.sqrt((By[gi]**2+Bz[gi]**2).mean()):.3e}")

if __name__=="__main__": main()
