#!/usr/bin/env python3
"""Batch phase-hole hunt across the dense giant_e12 ckpt series.

For every ckpt_<step>.bin (with its matching bline_<step/100>.bin):
  - build (zeta, u_par) for equatorial resonant electrons
  - trapping index T(t) = |m=1 Fourier coeff| of the zeta-distribution in the
    resonance band, normalized (0 = flat, >0 = phase-organized / trapping)
Then plot T(t) with WB(t), and the (zeta,u_par) delta-f map at the peak-T ckpt.
"""
import glob, os, struct, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT_SIZE={0:4,1:8,2:4,3:8,4:8}; DT_NP={0:np.float32,1:np.float64,2:np.int32,3:np.uint64,4:np.float32}
def parse(fn):
    with open(fn,"rb") as f:
        assert struct.unpack("<II",f.read(8))[0]==0x57435241
        f.read(16); t=struct.unpack("<d",f.read(8))[0]; f.read(16)
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
    return off,t
def rd(fn,off,name,stride):
    pos,dt,c=off[name]; npdt=DT_NP[dt]; it=DT_SIZE[dt]; n=c//stride; out=np.empty(n,npdt)
    with open(fn,"rb") as f:
        for c0 in range(0,n,4_000_000):
            c1=min(c0+4_000_000,n); f.seek(pos+c0*stride*it)
            blk=np.fromfile(f,npdt,(c1-c0)*stride); out[c0:c1]=blk[::stride][:c1-c0]
    return out

d=sys.argv[1] if len(sys.argv)>1 else "giant_e12"
xc=650.0; dx=0.26; nx=5000; stride=4; halfwin=100.0
VRES=(-0.30,-0.10)   # resonance band u_par
cks=sorted(glob.glob(d+"/ckpt_*.bin"),key=lambda s:int(s.split("_")[-1].split(".")[0]))
ts=[]; Tr=[]; best=None; bestT=-1
for ck in cks:
    step=int(ck.split("_")[-1].split(".")[0]); bl=f"{d}/bline_{step//100:06d}.bin"
    if not os.path.exists(bl): continue
    off,t=parse(ck)
    px=rd(ck,off,"px",stride)*dx; sel=np.abs(px-xc)<halfwin
    ux=rd(ck,off,"pux",stride)[sel]; uy=rd(ck,off,"puy",stride)[sel]; uz=rd(ck,off,"puz",stride)[sel]
    xp=px[sel]
    b=np.fromfile(bl,dtype=np.float32,count=2*nx); By,Bz=b[:nx],b[nx:]
    gi=np.clip((xp/dx).astype(int),0,nx-1)
    zeta=np.mod(np.arctan2(uz,uy)-np.arctan2(Bz[gi],By[gi]),2*np.pi)
    band=(ux>VRES[0])&(ux<VRES[1])
    zb=zeta[band]
    if len(zb)<1000: continue
    # m=1 Fourier amplitude of zeta distribution (trapping/bunching index)
    c1=np.mean(np.exp(1j*zb)); Tidx=np.abs(c1)
    ts.append(t*0.2); Tr.append(Tidx)
    if Tidx>bestT and t*0.2>800: bestT=Tidx; best=(ck,bl,t*0.2)
    print(f"t={t*0.2:6.0f}/We0  |Bw|_win={np.sqrt((By[gi]**2+Bz[gi]**2).mean()):.2e}  trapping_index={Tidx:.4f}  Nres={band.sum()}")
ts=np.array(ts); Tr=np.array(Tr)

# figure
fig,ax=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
e=np.genfromtxt(d+"/energy.csv",delimiter=",",names=True)
ax[0,0].plot(ts,Tr,"ko-",ms=4); ax[0,0].set_ylabel("trapping index |⟨e^{iζ}⟩|"); ax[0,0].grid(alpha=.3)
ax00b=ax[0,0].twinx(); ax00b.semilogy(e["time"]*0.2,e["WB"],"r-",alpha=.6); ax00b.set_ylabel("WB",color="r")
ax[0,0].axvspan(1000,1650,color="C0",alpha=.12); ax[0,0].axvspan(2000,2500,color="C1",alpha=.12)
ax[0,0].set_xlabel("t [1/We0]"); ax[0,0].set_title("phase-organization vs time (elements shaded)")

if best:
    ck,bl,tb=best; off,_=parse(ck)
    px=rd(ck,off,"px",stride)*dx; sel=np.abs(px-xc)<halfwin
    ux=rd(ck,off,"pux",stride)[sel]; uy=rd(ck,off,"puy",stride)[sel]; uz=rd(ck,off,"puz",stride)[sel]
    xp=px[sel]; b=np.fromfile(bl,dtype=np.float32,count=2*nx); By,Bz=b[:nx],b[nx:]
    gi=np.clip((xp/dx).astype(int),0,nx-1)
    zeta=np.mod(np.arctan2(uz,uy)-np.arctan2(Bz[gi],By[gi]),2*np.pi)
    H,ze,ue=np.histogram2d(zeta,ux,bins=[60,80],range=[[0,2*np.pi],[-0.6,0.3]])
    Hn=H/np.maximum(H.mean(0,keepdims=True),1)   # delta-f: normalize per u_par row
    im=ax[0,1].pcolormesh(ze,ue,(Hn-1).T,cmap="RdBu_r",vmin=-0.4,vmax=0.4,rasterized=True)
    plt.colorbar(im,ax=ax[0,1],pad=.01,label="δf/⟨f⟩_ζ")
    for vv in VRES: ax[0,1].axhline(vv,ls=":",c="k",lw=.7)
    ax[0,1].set_xlabel("ζ [rad]"); ax[0,1].set_ylabel("u$_\\parallel$/c")
    ax[0,1].set_title(f"(ζ,u∥) δf at peak trapping t={tb:.0f}/We0 (dotted=resonance band)")
    # raw counts too
    im2=ax[1,0].pcolormesh(ze,ue,H.T,cmap="viridis",rasterized=True)
    plt.colorbar(im2,ax=ax[1,0],pad=.01,label="counts")
    ax[1,0].set_xlabel("ζ [rad]"); ax[1,0].set_ylabel("u$_\\parallel$/c"); ax[1,0].set_title("raw (ζ,u∥)")
    # zeta profile in resonance band
    band=(ux>VRES[0])&(ux<VRES[1])
    hz,_=np.histogram(zeta[band],bins=60,range=(0,2*np.pi))
    ax[1,1].plot(np.linspace(0,2*np.pi,60),hz/hz.mean(),"b-")
    ax[1,1].axhline(1,c="k",lw=.6); ax[1,1].set_xlabel("ζ [rad]"); ax[1,1].set_ylabel("f(ζ)/⟨f⟩ in res band")
    ax[1,1].set_title(f"ζ-profile at resonance (depth = trapping)"); ax[1,1].grid(alpha=.3)
fig.suptitle(f"{d}: resonant-electron phase space — phase-hole hunt",fontsize=13)
fig.savefig(d+"_phase_hole.png",dpi=120); print("wrote",d+"_phase_hole.png","peak trapping at",best[2] if best else None)
