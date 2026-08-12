# Learning Note — Step 11: The Boris Pusher (`pusher.hpp`)

> Goal: understand **how particles move** in a PIC code — gathering the field to
> each particle (the inverse of deposition) and advancing it with the **Boris
> algorithm**, the standard energy-conserving mover for charged particles in E and
> B. Also: why gather must reuse the deposit stencil, how `if constexpr` makes the
> magnetic path free when unused, and how the velocity update is factored so the
> deferred Step-9 half-step rollback falls out for free.

---

## 0. What Step 11 is, in one paragraph

After the solver produces `Ex, Ey` on the grid, each particle must feel that field
and move. Step 11 builds `Pusher<Cfg>` and `boris_push_kernel<Cfg>` in
`include/pic/pusher.hpp`, namespace `arc`: it **gathers** E at the particle (CIC
interpolation), runs the **Boris** velocity update, and moves the position. It also
lands the leapfrog half-step rollback that Step 9 reserved. The test follows a single
particle in (A) a uniform E field and (B) a pure magnetic field, checking both
against exact analytic motion.

---

## 1. Gather — deposition run backwards

Deposition (Step 10) *scatters* one particle's charge to 4 grid cells. **Gather** does
the reverse: it *reads* the field at those same 4 cells and interpolates to the
particle, with the **same CIC weights**:

```cpp
__device__ void gather_E(const FieldViews& f, const CicStencil& st, float& Ex, float& Ey) {
    Ex = Ey = 0;
    for (int k = 0; k < 4; ++k) { Ex += st.w[k]*f.Ex[st.cell[k]]; Ey += st.w[k]*f.Ey[st.cell[k]]; }
}
```

It calls the *same* `cic_stencil` (grid.hpp) the depositor uses. This is not a style
preference — if gather and scatter used even slightly different weights, a particle
would exert a net force on *itself* (the field its own charge created wouldn't
interpolate back symmetrically), producing a spurious self-acceleration. Sharing the
one stencil makes that impossible (plan §16). This is the single most important
correctness rule tying Steps 10 and 11 together.

---

## 2. The Boris algorithm — why it, and how it works

We must integrate `du/dt = (q/m)(E + v×B)`. The naive approach (Euler) drifts in
energy: in a pure magnetic field, where the particle *should* circle forever at
constant speed, Euler spirals out. **Boris** is the standard fix. Its trick is to
split the step into three exact sub-steps:

```
1. half electric kick:   u⁻ = u + (q/m)(dt/2) E
2. magnetic rotation:    u⁺ = rotate(u⁻) by the B field over dt   (a pure rotation)
3. half electric kick:   u_new = u⁺ + (q/m)(dt/2) E
```

The key property: step 2 is a **pure rotation**, so it changes the velocity's
*direction* but never its *magnitude*. In a magnetic field (no E), Boris therefore
conserves speed — hence energy — *exactly*, for any timestep. That's why every PIC
code uses it. The rotation is done without trig via the Boris `t`/`s` vectors:

```cpp
t = (q/m)(dt/2) B;   s = 2t/(1+|t|²);
u' = u⁻ + u⁻ × t;    u⁺ = u⁻ + u' × s;     // algebraically an exact rotation
```

The rotation angle per step is `θ = 2·atan(|t|) = 2·atan(ω_c dt/2)` where
`ω_c = |(q/m)B|` is the **cyclotron frequency** — for small `dt`, `θ/dt → ω_c`, with a
tiny `O((ω_c dt)²)` frequency error. The test checks the measured per-step angle
against this exact Boris value (matches to <1e-6) and `ω_eff` against `ω_c` (0.99999).

### 2.1 The magnetic path is compile-time optional

```cpp
if constexpr (Cfg::has_b0) { /* t/s rotation */ }
```

`has_b0` is a `SimConfig` constant (Step 4). For a purely electrostatic build the
whole rotation block is **removed from the generated code** — zero runtime cost.
(With `has_b0=true` but `B0={0,0,0}`, the rotation degenerates to the identity, so the
same kernel handles the unmagnetized case correctly — which is how Case A runs.)

---

## 3. The position move (cell units again)

After the velocity update, leapfrog advances position by a full step. Velocities are
physical, positions are in **cell units** (Step 9), so the displacement is divided by
the cell size:

```cpp
p.x[t] += ux * dt / dx;     // v = u (γ≡1); cell-unit move = v·dt/dx
p.y[t] += uy * dt / dy;
```

`uz` exists (the field can accelerate out of plane via B) but doesn't move the
position in this 2D code. Wrapping back into `[0,nx)×[0,ny)` and recomputing `cell` is
**not** done here — that's `Particles::migrate`, called right after the push in the
main loop (Step 13). Keeping the move and the wrap separate keeps each kernel simple.

---

## 4. One velocity update, reused for the rollback

The velocity update is factored into `boris_velocity_update<Cfg>(..., dt_eff)`. The
full push calls it with `dt_eff = dt` then moves the position. The **leapfrog
half-step rollback** — reserved back in Step 9 — calls the *same* function with
`dt_eff = -dt/2` and skips the position move:

```cpp
static void half_step_back(...) {   // set u at t = -dt/2 (after the first field solve)
    boris_half_back_kernel<Cfg><<<...>>>;   // boris_velocity_update(-dt/2), no move
}
```

Why it was deferred to here: the rollback needs E **gathered at the particle**, with
the exact CIC weights of the Pusher's gather. Implementing it in Step 9 would have
meant a second, independent gather — the precise drift the shared-stencil rule
forbids. With gather now living in the Pusher, the backstep is three lines and
provably consistent. For `B0=0` it reduces to `u -= (q/m)(dt/2)E`; for `B0≠0` it's the
backward half-Boris (E half-kick + the rotation evaluated at `-dt/2`), which Step-15
magnetized runs need so the energy curve has no first-step offset.

---

## 5. How we verified Step 11

**Case A — uniform E, B0=0.** A particle from rest in constant `E0` should reach
`u = (q/m)E0·t` exactly (leapfrog adds `(q/m)E0·dt` each step, no B). After 100 steps:
`ux = -0.5` to <1e-5, `uy = 0`. ✓

**Case B — gyration, E=0, Bz=1, q/m=-1 ⇒ ω_c=1.** Over ~one cyclotron period:
- per-step rotation = `2·atan(ω_c dt/2)` (Boris-exact) to <1e-6;
- `ω_eff = 0.99999 ≈ ω_c`;
- speed `|u|` conserved to `3.6e-7` (the energy-exactness of Boris);
- gyroradius `= 1.0000` vs analytic `v⊥/ω_c = 1`. ✓

CTest 6/6; `compute-sanitizer --leak-check full` → 0 leaked / 0 errors.

---

## 6. Concepts you learned in Step 11 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Gather | interp grid field → particle | inverse of deposit; **same** CIC weights |
| Shared stencil rule | gather reuses `cic_stencil` | no spurious self-force (plan §16) |
| Boris algorithm | half-E → rotate → half-E | energy-exact mover for E+B |
| Rotation is magnitude-preserving | step 2 is a pure rotation | speed/energy conserved in B |
| `t`/`s` vectors | trig-free exact rotation | `θ = 2 atan(ω_c dt/2)` |
| Cyclotron frequency | `ω_c = |(q/m)B|` | sets gyration rate & radius `v⊥/ω_c` |
| `if constexpr (has_b0)` | compile-time B path | electrostatic build pays nothing |
| Position in cell units | move = `v·dt/dx` | consistent with deposit/gather |
| Factored velocity update | `boris_velocity_update(dt_eff)` | reused for the `-dt/2` rollback |

---

## 7. One-sentence summary

Step 11's `Pusher<Cfg>` gathers E to each particle with the depositor's exact CIC
stencil, advances velocity with the energy-conserving Boris scheme (half-E → trig-free
B rotation behind `if constexpr (has_b0)` → half-E) and position by `v·dt/dx`, and —
by factoring the velocity update over an arbitrary `dt_eff` — also delivers Step 9's
deferred `-dt/2` leapfrog rollback; verified exactly against uniform-E acceleration
and against cyclotron gyration (Boris-exact frequency, energy to 3.6e-7, radius 1.0000).
