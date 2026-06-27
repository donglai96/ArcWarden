# Learning Note — Step 8: Electrostatic Spectral Solver (`solver_es.hpp`)

> Goal: understand **how charge density becomes the electric field in Fourier
> space** — the whole `rho → E` pipeline — and the three subtle points that make it
> *correct*: dividing by `k²` (with k=0 and Nyquist handled), the `-i k` derivative,
> and where normalization lives. This is the physics payoff of Steps 5–7; the
> single-sine test pins it to an exact analytic answer.

---

## 0. What Step 8 is, in one paragraph

`ElectrostaticSpectralSolver::solve(sources, fields, engine, rp, stream)` takes the
deposited charge density `rho` and fills the field `Ex, Ey` — by transforming to
Fourier space, solving Poisson's equation mode-by-mode, applying the gradient, and
transforming back. It lives in `include/pic/solver_es.hpp`, namespace `arc`, and
*borrows* the Step-6 `SpectralEngine` (it owns no FFT itself). The test
`tests/test_poisson_sine.cu` drives it with `rho = sin(x)` and checks the result
against the exact answer `Ex = -cos(x)`, `Ey = 0`.

---

## 1. Background: electrostatics in three lines

The electrostatic field comes from a potential `φ` that satisfies Poisson's equation:

```
∇²φ = -ρ/ε0        (Poisson)
E   = -∇φ          (field is minus the gradient of the potential)
```

In real space these are coupled PDEs. On a **periodic** grid we use the Step-5 fact
that `∇ → i k` in Fourier space: `∇²φ → -k²φ_k` and `∇φ → i k φ_k`. So:

```
-k² φ_k = -ρ_k/ε0   →   φ_k = ρ_k / (ε0 k²)
E_k = -i k φ_k       →   Ex_k = -i kx φ_k ,  Ey_k = -i ky φ_k
```

A coupled PDE collapses to **independent per-mode arithmetic**. That's the entire
reason spectral PIC is fast and accurate on periodic domains.

---

## 2. The pipeline (three stages)

```cpp
void solve(const Sources& src, Fields& fld, SpectralEngine& eng,
           const RunParams& rp, cudaStream_t s) const {
    auto& ws = eng.ws();
    eng.r2c(src.rho.data(), ws.rho_k.data(), s);                 // 1. rho -> rho_k
    es_poisson_field_kernel<<<...,s>>>(ws.rho_k.view(), ws.phi_k.view(),
        ws.Ex_k.view(), ws.Ey_k.view(), eng.kgrid(),
        SpectralFormFactor{rp.eps0}, nx, ny);                    // 2. k-space solve
    eng.c2r(ws.Ex_k.data(), fld.Ex.data(), s);                   // 3. Ex_k -> Ex
    eng.c2r(ws.Ey_k.data(), fld.Ey.data(), s);                   //    Ey_k -> Ey
}
```

- **Borrows the engine.** The solver holds no plan and no buffers; it uses the
  engine's workspace (`rho_k/phi_k/Ex_k/Ey_k`). This keeps all FFT ownership in one
  place (design §7.2) and means a future Darwin solver reuses the same engine.
- **Host orchestration, one call per step.** Unlike the deposit/push *kernels* (hot,
  compile-time-fixed, Step 11), the solver is a host object invoked once per step, so
  it can be a plain concrete class — polymorphism here would be free but isn't needed
  yet.

---

## 3. The k-space kernel — three things it must get right

```cpp
const double k2 = kg.k2(i,j);
const double g  = ff.green(k2);            // 1/(eps0 k²), and 0 at k²=0
cufftComplex phi = rho_k[t] * g;           // phi_k
// Nyquist force-zero, then:
Ex_k = -i kx phi  =>  (kx*phi.y, -kx*phi.x);
Ey_k = -i ky phi  =>  (ky*phi.y, -ky*phi.x);
```

Each thread handles one half-spectrum element `(i,j)` (recovered from the flat
thread id via `i = t % nkx`, `j = t / nkx`). Three correctness points:

### 3.1 The k=0 (DC) mode — divide-by-zero *and* physics

At `i=j=0`, `k²=0`, so `1/k²` is infinite. `SpectralFormFactor::green` returns `0`
there, which sets `φ_{k=0}=0`. This is not a hack: the k=0 Fourier mode is the
spatial **average**. Forcing it to zero removes the DC offset, which for a periodic
electrostatic plasma is exactly the **neutralizing background** — a uniform charge
density has no self-consistent periodic field, so the average potential is gauge-free
and conventionally set to 0 (design §7.2, UPIC's `POIS2`).

### 3.2 The Nyquist mode — zero the *force*

For an even-length axis, the highest mode (`kx = nx/2` or `ky = ny/2`) is the
**Nyquist** frequency. It's a pure-real mode (it has no distinguishable sign of
frequency), and applying a first-derivative operator `-i k` to it produces a force
the grid can't represent faithfully — it injects spurious high-frequency noise. The
standard spectral remedy is to **zero the force** on Nyquist modes (we keep `φ` but
set `Ex_k = Ey_k = 0` there). The kernel detects it with
`(nx%2==0 && i==nx/2) || (ny%2==0 && j==ny/2)`.

### 3.3 The `-i k` multiply, in components

cuFFT stores a complex number as `cufftComplex {x, y}` = `(real, imag)`. Multiplying
`φ = a + i b` by `-i k`:

```
-i k (a + i b) = -i k a - i² k b = k b - i k a   =>   (k*b, -k*a)
```

So `Ex_k.x = kx*phi.y`, `Ex_k.y = -kx*phi.x`. Getting this component bookkeeping wrong
is the classic spectral-PIC sign bug; the single-sine test exists precisely to catch
it.

> The kernel is `template<class FF>` so it has *vague linkage* and is ODR-safe to
> define in a header (same trick as the Step-6 scale kernel). `eps0` flows in from
> `RunParams` via `SpectralFormFactor{rp.eps0}` — the Green's function stays the one
> place `1/k²` is written.

---

## 4. Where normalization lives (recap)

The solver never writes a `1/N` factor: the **inverse transform** owns it.
`SpectralEngine::c2r` applies `1/(nx*ny)` after cuFFT's unnormalized C2R (Step 6). So
`solve` just calls `c2r` twice and the fields come out correctly scaled. One factor,
one place — the contract from `config.hpp`/`grid.hpp`. (Note `c2r` is destructive on
its input, which is fine because `Ex_k` and `Ey_k` are separate buffers consumed once.)

---

## 5. The test: an exact analytic check

Drive the solver with a single Fourier mode and you get a closed-form answer to
compare against. With `ρ(x) = sin(k1 x)`, `k1 = 2π/Lx`, and `ε0 = 1`:

```
φ = sin(k1 x) / k1²          (from φ = ρ/(ε0 k²), k = k1)
E = -dφ/dx = -cos(k1 x) / k1
```

Choosing `Lx = 2π` makes `k1 = 1`, so the expected field is `Ex(x) = -cos(x)`,
`Ey = 0`. (Hand-checking the discrete transform: `rho_k[i=1] = (0, -N/2)`, `green=1`,
the `-i k` multiply gives `Ex_k[1] = (-N/2, 0)`, and the normalized inverse yields
exactly `-cos(x)` — so the test verifies amplitude *and* phase, not just shape.)

Results: `max|Ex − (−cos x)| = 9.2e-8`, peak `|Ex| = 1.0000`, and `Ey ≡ 0` exactly
(every `ky = 0` for this x-only mode). CTest 3/3 green; `compute-sanitizer
--leak-check full` reports 0 leaked / 0 errors.

> Why `Ey` is *identically* zero: `ρ` depends only on x, so only `j=0` modes are
> nonzero, where `ky = 0` ⇒ `Ey_k = 0` ⇒ `Ey = 0`. A nice exact secondary check.

---

## 6. Concepts you learned in Step 8 (cheat sheet)

| Concept | What it is | Why it matters here |
|---|---|---|
| Poisson in k-space | `φ_k = ρ_k/(ε0 k²)` | coupled PDE → per-mode division |
| `∇ → i k` | derivative is multiply in Fourier space | `E_k = -i k φ_k` |
| DC mode zeroing | `green(0)=0` | avoids 1/0 *and* sets neutralizing background |
| Nyquist force-zero | drop `-i k` on the pure-real top mode | kills spurious grid-scale noise |
| `-i k` in components | `(k·imag, -k·real)` | the classic sign bug; test guards it |
| Borrow, don't own | solver uses engine's plan+workspace | FFT ownership stays in one place |
| Host orchestration | one call/step, concrete class OK | vs. compile-time-fixed hot kernels |
| Single-mode analytic test | exact `Ex=-cos x`, `Ey=0` | validates amplitude + phase + sign |

---

## 7. One-sentence summary

Step 8's `ElectrostaticSpectralSolver` does the full `rho → E` by borrowing the
SpectralEngine: forward-FFT `rho`, solve `φ_k = ρ_k/(ε0 k²)` per mode (zeroing the DC
mode for neutrality and the Nyquist force for stability), apply `E_k = -i k φ_k`, and
inverse-FFT (with the engine's `1/N`) — verified to 9e-8 against the exact
single-sine solution `Ex = -cos x`, `Ey = 0`.
