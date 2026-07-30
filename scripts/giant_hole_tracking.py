#!/usr/bin/env python3
"""Phase-space-hole tracking for giant_e12, the Tao-2017 way (plot_chirp1d_
hole_tracking.py recipe): NARROW field-aligned window (single local Omega),
a resonant u_perp band, (zeta,u_par) with the zeta-average removed per u_par,
and the hole center tracked against the relativistic resonant momentum
u_R(omega(t)) from the measured local ridge frequency.

Fixes the earlier mistakes: (1) wide +-100 window -> narrow single-Omega window;
(2) all u_perp -> resonant band; (3) fixed u_par band -> track u_R(omega(t)).
"""
import struct, sys
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
            nl=struct.unpack("<I",f.read(4))[0]; nm=f.read(nl).decode()
            dt=struct.unpack("<I",f.read(4))[0]; c=struct.unpack("<Q",f.read(8))[0]; man.append((nm,dt,c))
        ds=f.tell()
    off={}; pos=ds
    for nm,dt,c in man: off[nm]=(pos,dt,c); pos+=c*DT_SIZE[dt]
    return off,t
def rd(fn,off,nm,st):
    pos,dt,c=off[nm]; npdt=DT_NP[dt]; it=DT_SIZE[dt]; n=c//st; out=np.empty(n,npdt)
    with open(fn,"rb") as f:
        for c0 in range(0,n,4_000_000):
            c1=min(c0+4_000_000,n); f.seek(pos+c0*st*it)
            b=np.fromfile(f,npdt,(c1-c0)*st); out[c0:c1]=b[::st][:c1-c0]
    return out

d="giant_e12"; nx=5000; dx=0.26; xc=650.; Om0=0.2; wpe=1.0
HCEN=float(sys.argv[1]) if len(sys.argv)>1 else -55.0   # window center (c/wpe, south)
HWID=float(sys.argv[2]) if len(sys.argv)>2 else 20.0    # window half-width ~2 wavelengths
UP0,UP1=0.30,0.65                                        # resonant u_perp band
st=1                                                     # all particles (full-f is noisy)
# local Omega ratio from dipole (small-lat: ~1)
import math
def lat_of_s(s):
    sgn=math.copysign(1,s); s=abs(s); lo,hi=0.,40.
    lre=1330.504
    fn=lambda l: lre*0.5*(math.sin(math.radians(l))*math.sqrt(1+3*math.sin(math.radians(l))**2)+math.asinh(math.sqrt(3)*math.sin(math.radians(l)))/math.sqrt(3))
    for _ in range(50):
        mid=.5*(lo+hi); lo,hi=(mid,hi) if fn(mid)<s else (lo,mid)
    return sgn*.5*(lo+hi)
lam=lat_of_s(HCEN); sl=math.sin(math.radians(abs(lam)))
Wratio=math.sqrt(1+3*sl*sl)/math.cos(math.radians(abs(lam)))**6
Omloc=Om0*Wratio    # local Omega_e in wpe units

# ridge omega(t) from equator probe STFT (in Omega_e0 units)
meta={}
for l in open(d+"/meta.txt"):
    k,v=l.split(None,1); meta.setdefault(k,[]).append(v.strip())
npr=int(meta["nprobe"][0]); dt=float(meta["dt"][0]); pe=int(meta["probe_every"][0])
ix=[int(v) for v in meta["probe_ix"]]; offs=[(i+0.5)*dx-xc for i in ix]
eqp=min(range(npr),key=lambda p:abs(offs[p]-HCEN))
raw=np.fromfile(d+"/probe.bin",dtype=np.float32); nrec=raw.size//(2*npr)
S=(raw[:nrec*2*npr].reshape(nrec,npr,2)); sig=S[:,eqp,0]+1j*S[:,eqp,1]
from scipy.signal import stft
f,tt,Z=stft(sig,fs=2*np.pi/pe/dt,nperseg=512,noverlap=512-64,return_onesided=False,boundary=None)
pos=f>=0; fpos=f[pos]/Om0; P=np.abs(Z[pos])**2; tc=tt*2*np.pi*Om0
selb=(fpos>=0.15)&(fpos<=0.9)
def ridge_at(tq):
    j=np.argmin(abs(tc-tq)); return fpos[selb][np.argmax(P[selb,j])]
def kcold(w_wpe):  # cold whistler R-mode, wpe=1,c=1
    return np.sqrt(w_wpe*w_wpe + w_wpe/(Omloc - w_wpe))
def uR(wOe, up=0.5):
    w=wOe*Om0   # to wpe units
    k=kcold(w); v=-0.2
    for _ in range(80):
        vp2=(up/math.sqrt(1+up*up))**2
        g=1.0/math.sqrt(max(1-min(v*v+vp2,0.985),1e-3))
        v=(w-Omloc/g)/k
    vp2=(up/math.sqrt(1+up*up))**2
    g=1.0/math.sqrt(max(1-min(v*v+vp2,0.985),1e-3))
    return v*g

# times through element 1
steps=[35000,40000,42500,45000,47500,50000]
fig,axes=plt.subplots(1,len(steps)+1,figsize=(3.1*(len(steps)+1),3.6),squeeze=False)
ht,hu=[],[]
for c,step in enumerate(steps):
    off,t=parse(f"{d}/ckpt_{step}.bin"); t*=0.2
    b=np.fromfile(f"{d}/bline_{step//100:06d}.bin",dtype=np.float32,count=2*nx); By,Bz=b[:nx],b[nx:]
    px=rd(f"{d}/ckpt_{step}.bin",off,"px",st)*dx; gi=np.clip((px/dx).astype(int),0,nx-1)
    ux=rd(f"{d}/ckpt_{step}.bin",off,"pux",st); uy=rd(f"{d}/ckpt_{step}.bin",off,"puy",st); uz=rd(f"{d}/ckpt_{step}.bin",off,"puz",st)
    up=np.hypot(uy,uz)
    slc=(np.abs(px-xc-HCEN)<HWID)&(up>UP0)&(up<UP1)
    zeta=np.mod(np.arctan2(uz[slc],uy[slc])-np.arctan2(Bz[gi[slc]],By[gi[slc]]),2*np.pi)
    wom=ridge_at(t); ur=uR(wom,0.5)
    H,xe,ye=np.histogram2d(zeta/np.pi,ux[slc],bins=[40,70],range=[[0,2],[-0.5,0.5]])
    Hm=(H-H.mean(0,keepdims=True))/np.maximum(H.mean(0,keepdims=True),1)  # frac delta-f
    ax=axes[0,c]; lim=np.percentile(np.abs(Hm),99) or 1
    ax.pcolormesh(xe,ye,Hm.T,cmap="RdBu_r",vmin=-lim,vmax=lim,rasterized=True)
    ax.axhline(ur,color="lime",lw=1.4,ls="--")      # k>0 (north-going) resonance
    ax.axhline(-ur,color="magenta",lw=1.4,ls=":")   # k<0 (south-going) resonance
    ax.set_xlabel("ζ/π"); ax.set_title(f"t={t:.0f} ω={wom:.2f}\nN={slc.sum()}",fontsize=9)
    if c==0: ax.set_ylabel("u∥/c")
    band=(xe[:-1]+np.diff(xe)/2>0.6)&(xe[:-1]+np.diff(xe)/2<1.4)
    prof=Hm[band].mean(0); uc=ye[:-1]+np.diff(ye)/2; m=(uc>-0.35)&(uc<0.35)
    ht.append(t); hu.append(uc[m][np.argmin(prof[m])])
ax=axes[0,len(steps)]
tt2=np.linspace(min(ht)-50,max(ht)+50,50)
ax.plot(tt2,[uR(ridge_at(x),0.5) for x in tt2],"g-",label="u_R(ω(t))")
ax.plot(ht,hu,"ko--",label="hole center")
ax.set_xlabel("t [1/We0]"); ax.set_ylabel("u∥/c"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("hole tracks sweeping resonance?",fontsize=9)
fig.suptitle(f"giant_e12 element 1: (ζ,u∥) δf, NARROW window lat≈{lam:.1f}° (Ω_loc={Wratio:.3f}), u⊥∈[{UP0},{UP1}] — Tao-2017 method",fontsize=11)
fig.tight_layout(); fig.savefig(d+"_hole_tracking.png",dpi=130)
print("hole centers:",[f"{u:.3f}" for u in hu]); print("u_R:",[f"{uR(ridge_at(t),0.5):.3f}" for t in ht])
print("wrote",d+"_hole_tracking.png")
