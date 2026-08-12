# Learning Note — Step 9: Particles & Quiet Loading (`particles.hpp`)

> Goal: understand **how PIC stores its particles** (a flat structure-of-arrays,
> not array-of-structs, and why), what "positions in cell units" buys you, and the
> single most important loading idea — the **quiet start** (low-discrepancy
> sampling) that replaces noisy RNG with a smooth, reproducible distribution. Also:
> why deposit/push are *not* methods here, and why the leapfrog half-step rollback
> is deferred to Step 11.

---

## 0. What Step 9 is, in one paragraph

The plasma is millions of macro-particles, each with a position and a momentum. Step
9 builds the container that stores them and the routine that loads them into a
thermal (Maxwellian) state — `include/pic/particles.hpp`, namespace `arc`:
`Particles` (owns the SoA arrays `x, y, ux, uy, uz, cell`), `ParticleViews` (the POD
handle pack for kernels), `initialize` (quiet Maxwellian load), and `migrate`
(periodic wrap + recompute cell). Deposit and push live elsewhere (Steps 10–11);
`Particles` only owns storage, loading, and migration.

---

## 1. SoA vs. AoS — why structure-of-arrays

You could store particles as an array of structs (AoS): `struct P {float x,y,ux,...;};
P parts[N];`. PIC uses the opposite — **structure of arrays** (SoA): one array per
attribute, `x[N]`, `y[N]`, `ux[N]`, … The reason is GPU memory coalescing: when 32
neighboring threads each read `x[t]`, SoA makes those 32 reads contiguous in memory
(one cache transaction); AoS scatters them by `sizeof(P)` (many transactions). SoA is
the difference between using and wasting memory bandwidth, and PIC is bandwidth-bound.

```cpp
struct Particles {
    DeviceArray<float> x, y;        // position (cell units)
    DeviceArray<float> ux, uy, uz;  // momentum  u = γv  (γ≡1 in v1)
    DeviceArray<int>   cell;        // owning cell index
    std::size_t n = 0;
};
struct ParticleViews {              // POD pack for kernels
    DeviceView<float> x, y, ux, uy, uz;
    DeviceView<int>   cell;
    int n;
};
```

`Particles` owns (host-side `DeviceArray`s); `ParticleViews` borrows (the POD that
rides into kernels). Same owner/view split as every container before it.

---

## 2. Two conventions baked into the data

- **Positions in *cell units*.** `x ∈ [0,nx)`, `y ∈ [0,ny)` — measured in cells, not
  physical length. So the CIC cell a particle sits in is just `floor(x)`, and the
  interpolation weight is the fraction `x − floor(x)`. Deposit and gather read this
  directly with **no `dx` division in the hot path** — fewer flops per particle per
  step, and one less place to get a unit wrong.
- **Momentum, not velocity.** `ux,uy,uz` store `u = γv`. v1 is non-relativistic so
  `γ ≡ 1` and `u = v`, but the *storage contract* is momentum (design §3), so a later
  relativistic push is a local change, not a data-layout migration.

---

## 3. The responsibility boundary (why no deposit/push here)

`Particles` deliberately has **only** `allocate / initialize / migrate / views`.
Charge deposition and the Boris push are *not* methods — they are `Depositor<Cfg>`
and `Pusher<Cfg>` (Steps 10–11). The payoff: when you later swap the storage backend
(flat SoA → tiled chunk pool for locality), `Particles` changes internally and the
algorithm layer doesn't move at all. Storage and algorithms are decoupled.

---

## 4. The quiet start (the heart of this step)

### 4.1 Why not just use a random number generator?

Sampling N velocities from `numpy.random.normal` gives a Maxwellian — plus **shot
noise** that scales like `1/√N`. In a PIC run that noise seeds spurious fluctuations
and muddies the very instabilities (Langmuir, two-stream) we want to measure. A
**quiet start** replaces random draws with a deterministic **low-discrepancy
sequence** that fills the distribution far more evenly, cutting the noise by orders of
magnitude for the same N — and it's reproducible.

### 4.2 Stratified positions

Particles are placed `ppc` per cell. Within each cell, the sub-cell offset comes from
a **van der Corput sequence** (the radical inverse of the index in a prime base):

```cpp
__host__ __device__ double radical_inverse(unsigned i, unsigned base) {
    double inv = 1.0/base, f = inv, r = 0.0;
    while (i) { r += (i % base) * f; i /= base; f *= inv; }
    return r;                       // a point in (0,1), spread evenly as i grows
}
// particle s within cell (i,j):
x = i + radical_inverse(s+1, 2);    // base 2 for the x offset
y = j + radical_inverse(s+1, 3);    // base 3 for the y offset
```

Base 2 and base 3 (coprime) give a 2D point set that tiles the cell uniformly — no
clumping, exactly `ppc` per cell. (The test confirms every cell holds exactly `ppc`.)

### 4.3 Quiet Maxwellian velocities via inverse-CDF

To turn a uniform quantile `q ∈ (0,1)` into a Gaussian sample, invert the normal CDF.
For a Maxwellian with standard deviation `vth`, `Φ⁻¹(q) = √2 · erfinv(2q − 1)`, so:

```cpp
double qx = radical_inverse(t+1, 2);    // low-discrepancy quantiles per component
double qy = radical_inverse(t+1, 3);    // (bases 2,3,5 → a 3D Hammersley-like set)
double qz = radical_inverse(t+1, 5);
ux = vd + √2·vth · erfinv(2qx − 1);     // drift vd on x; thermal spread vth
uy =      √2·vth · erfinv(2qy − 1);
uz =      √2·vth · erfinv(2qz − 1);
```

Feeding the *low-discrepancy* quantiles (instead of RNG uniforms) through `erfinv`
produces a Maxwellian that is smooth and nearly noise-free. The position uses index
`s` (within-cell) while the velocity uses the global index `t`, so position and
velocity are decorrelated. `vd` is a bulk drift (zero for a plain thermal plasma, set
nonzero to build counter-streaming beams for two-stream later).

> The single-k density perturbation (displace particles to make `ρ ∝ 1+ε cos kx`,
> for seeding Langmuir/two-stream) is a documented hook; v1's default load is uniform.

---

## 5. `migrate` — periodic wrap + recompute cell

After particles move, some leave the box; on a periodic domain they re-enter the far
side, and their `cell` index must be recomputed:

```cpp
float x = fmodf(p.x[t], nx); if (x < 0) x += nx;   // wrap into [0,nx)
float y = fmodf(p.y[t], ny); if (y < 0) y += ny;
int ci = floorf(x), cj = floorf(y);
if (ci >= nx) ci = nx-1;  if (cj >= ny) cj = ny-1; // guard x==nx-ε rounding
p.cell[t] = g.idx(ci, cj);
```

`fmodf` (vs. the cheaper two-branch `wrap` in `Grid`) is used here because migrate
must tolerate a particle that jumped *several* periods. v1 just rewraps in place; a
chunk-pool reshuffle for memory locality is a later optimization behind the same
`migrate` interface.

---

## 6. Why the leapfrog half-step rollback is deferred to Step 11

Leapfrog integration wants `u` at the *half* step while `x` is at the *whole* step,
so initialization must push velocities back half a step: for `B0=0`, a `−dt/2`
electrostatic kick; for `B0≠0`, a backward half-Boris (with the B rotation). But that
kick needs the field **gathered at each particle**, using the *exact same CIC weights*
as the Pusher's gather. Implementing a second gather here would risk the classic
PIC bug where deposit and gather use subtly different weights (plan §16). So the
rollback interface is reserved in Step 9 and *implemented in Step 11* alongside the
shared gather. Loading itself is field-free, so Step 9 stands on its own.

---

## 7. How we verified Step 9

1. **Stratification / positions:** every particle in `[0,nx)×[0,ny)`, `cell ==
   idx(⌊x⌋,⌊y⌋)`, and every cell holds **exactly `ppc`** particles.
2. **Maxwellian statistics:** per-component `mean ≈ −1e-4` (quiet → tiny), `std =
   0.9999 ≈ vth`, and `fraction(|v|<vth) = 0.6827` — the exact Gaussian value
   (`erf(1/√2)`). RNG loading would be visibly noisier at this N.
3. **Energy:** `KE/particle = 1.4992 ≈ 1.5 vth²` (½ per dim × 3 dims).
4. **migrate:** after shoving every particle several periods out, all return to range
   with a consistent `cell`.

CTest 4/4; `compute-sanitizer --leak-check full` → 0 leaked / 0 errors.

---

## 8. Concepts you learned in Step 9 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| SoA vs AoS | one array per attribute | coalesced GPU reads; bandwidth is the bottleneck |
| Positions in cell units | x∈[0,nx) | CIC cell = floor(x); no `dx` in the hot path |
| Momentum storage | `u = γv`, γ≡1 in v1 | relativistic retrofit is local later |
| Responsibility boundary | no deposit/push in Particles | swap storage backend without touching algorithms |
| Quiet start | low-discrepancy load, not RNG | noise ∝ far below 1/√N; reproducible |
| van der Corput / radical inverse | even (0,1) sequence in a base | stratified positions, Hammersley velocities |
| Inverse-CDF sampling | `√2·vth·erfinv(2q−1)` | uniform quantile → Gaussian velocity |
| `migrate` | periodic wrap + recompute cell | `fmodf` tolerates multi-period jumps |
| Deferred half-step rollback | needs Pusher's CIC gather | avoid a second, drifting gather (Step 11) |

---

## 9. One-sentence summary

Step 9 stores particles as a coalescing-friendly SoA (`x,y` in cell units, `ux,uy,uz`
as momentum, plus a `cell` index) and loads them with a **quiet start** — stratified
van der Corput positions and an inverse-CDF Maxwellian on low-discrepancy quantiles —
giving a smooth, reproducible thermal plasma (std=0.9999, frac=0.6827, KE/particle
=1.4992) with `migrate` for periodic wrapping, while leaving deposit/push and the
field-dependent half-step rollback to Steps 10–11.
