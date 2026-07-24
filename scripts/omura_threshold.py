import numpy as np

# Case II parameters, our units: wpe_total=1, Oe=0.2
Oe = 0.2
nc = 0.9822                    # cold density -> cold wpe
wpe = np.sqrt(nc)
nh = 0.0178
Utpar  = 0.19783565            # c units
Utperp = 0.24390817
rho, kap = 1.0, 0.3
Q, tau = 0.5, 0.5
Lam = 1.0                      # constant cold density model
# mean perp speed of the subtracted bi-Max (matches Omura's V_perp0=0.45c example)
Uperp0 = np.sqrt(np.pi/2)*Utperp*(1-rho*kap**1.5)/(1-rho*kap)
wph_t = np.sqrt(nh)/Oe         # hot plasma freq / Oe

def disp_k(w):                 # w in wpe units -> k in wpe/c
    return w*np.sqrt(1 + wpe**2/(w*(Oe-w)))/1.0  # n = sqrt(1+wpe^2/(w(Oe-w))), k = n w / c
def branch(wt):                # wt = omega/Oe
    w = wt*Oe
    k = disp_k(w)
    dw = 1e-6
    Vg = (2*dw*Oe)/(disp_k(w+dw*Oe)-disp_k(w-dw*Oe))
    xi = np.sqrt(w*(Oe-w))/wpe
    chi = 1/np.sqrt(1+xi*xi)
    Vp = chi*xi
    # relativistic VR, eq (23), v_perp = Uperp0
    A = wt*wt + Vp*Vp
    VR = (wt*wt - np.sqrt(wt**4 + A*(1-wt*wt-Uperp0**2)))/A * Vp
    gam = 1/np.sqrt(1-VR*VR-Uperp0**2)
    s2 = (gam*wt*Uperp0**2 - (2 + Lam*chi**2*(1-gam*wt)/(1-wt))*VR*Vp)/(2*xi*chi)
    return k, Vg, xi, chi, Vp, VR, gam, s2

def thresholds(lre):
    at = 112.5/lre**2          # a~ = a c^2/Oe^2, a = 4.5/lre^2
    out = {}
    for wt in (0.20, 0.25, 0.30, 0.40):
        k, Vg, xi, chi, Vp, VR, gam, s2 = branch(wt)
        th = (100*np.pi**3*gam**4*xi)/(wt*wph_t**4*(chi*Uperp0)**5) \
             * (at*s2*Utpar/Q)**2 * np.exp(gam**2*VR**2/Utpar**2)
        op = 0.8*np.pi**-2.5 * (Q*Vp*Vg*Uperp0*wph_t**2)/(tau*wt*Utpar**2) \
             * (1-VR/Vg) * np.exp(-gam**2*VR**2/(2*Utpar**2))
        out[wt] = (th, op)
    return out

print(f"Uperp0 = {Uperp0:.4f}c   wph/Oe = {wph_t:.4f}")
for name, lre in [("x5 ", 2661.008), ("x10", 1330.504), ("x20", 665.252)]:
    r = thresholds(lre)
    print(f"\n{name} (lre={lre:.0f}, a~={112.5/lre**2:.3e}):")
    for wt,(th,op) in r.items():
        tag = "  << th > opt: SUPPRESSED" if th > op else ""
        print(f"  w/Oe={wt:.2f}: B_th/B0 = {th:.3e}   B_opt/B0 = {op:.3e}{tag}")
