# C++ 2D3V Electrostatic -> Spectral Darwin PIC 开发计划

这份文档的目标不是复刻 UCLA/UPIC 的 Fortran 代码，而是吸收它的架构思想，用 C++ 写一个自己的、可逐步扩展的 PIC code：

```text
第一阶段: 2D3V electrostatic spectral PIC
第二阶段: 2D3V magnetized electrostatic PIC
第三阶段: 2D3V spectral Darwin PIC
第四阶段: 可选 simplified full-EM solver 作为对照
```

推荐定位：

```text
Modern C++ single-GPU-ready 2D3V spectral PIC code,
starting from electrostatic Poisson solve,
designed to grow into a Darwin solver.
```

第一版不要追求 production performance。第一版的唯一目标是：结构正确、物理闭环、诊断可信、以后能自然加入 Darwin 的 `J`、`B`、transverse projection 和 spectral correction。

---

## 1. 从 UCLA / UPIC spectral PIC 借鉴什么

UCLA UPIC / UPIC-2.0 的核心启发是架构，不是具体语法。你给的目录是：

```text
UPIC-2.0/mpbeps2
```

这个目录里有 3 个 reference applications：

```text
mpbeps2.f90   electrostatic
mpbbeps2.f90  electromagnetic
mpdbeps2.f90  Darwin
```

它们共用一批 trusted components：初始化、particle push、charge/current deposition、spectral field solver、FFT、sorting/reordering、diagnostics。你自己的 C++ code 应该学习这种组件边界，而不是照搬 Fortran/MPI/OpenMP 的实现细节。

1. PIC 主循环仍然是标准结构：

```text
deposit sources -> solve fields -> gather fields -> push particles -> diagnostics
```

2. spectral solver 把空间微分放到 Fourier space：

```text
d/dx -> i kx
d/dy -> i ky
```

3. field quantities 可以放在同一个 collocated mesh 上，不需要 Yee staggered grid。

4. electrostatic / longitudinal field 每步直接由 `rho` 解 Gauss/Poisson：

```text
div E = rho / eps0
E = -grad phi
laplacian phi = -rho / eps0
```

5. Darwin / spectral EM 以后需要 `rho` 和 `J`，并在 k-space 做：

```text
longitudinal projection
transverse projection
k-space filtering
continuity / Gauss correction
```

6. 单卡 2D code 的 FFT 成本可接受，开发难度远低于 multi-GPU distributed FFT。

所以你的代码应该分成两个 engine：

```text
Particle engine:
    particle storage
    deposition
    field gather
    particle push
    migration / boundary

Spectral field engine:
    real-space source arrays
    R2C FFT
    k-space solve / filter / projection
    C2R FFT
    field normalization
```

这两个 engine 通过 `Sources2D` 和 `Fields2D` 连接。这样第一版只用 `rho, Ex, Ey`，以后 Darwin 可以加入 `Jx, Jy, Jz, Bx, By, Bz, E_T`，不用推翻代码结构。

---

## 2. UPIC-2.0 `mpbeps2` 代码实际结构

### 2.1 先读这些文件

如果你要基于 `mpbeps2` 写自己的 C++ 版本，阅读顺序建议是：

```text
mpbeps2/README
mpbeps2.source/input2mod.f90
mpbeps2.source/mpbeps2.f90
mpbeps2.source/mpdbeps2.f90
mpbeps2.source/libmppush2.f
mpbeps2.source/libmpcurd2.f
mpbeps2.source/libmpdpush2.f
mpbeps2.source/libmpfield2.f
mpbeps2.source/libmpsort2.f
Periodic/mp_pfields2.f90
```

对应关系：

```text
input2mod.f90:
    namelist 参数、默认值、单位、grid size、particle shape、diagnostic interval

mpbeps2.f90:
    2D electrostatic reference application

mpdbeps2.f90:
    2-1/2D Darwin reference application

libmppush2.f:
    electrostatic charge deposition / electrostatic particle push

libmpcurd2.f:
    current deposition for 2-1/2D EM/Darwin

libmpdpush2.f:
    Darwin 需要的 acceleration density 和 momentum flux deposition

libmpfield2.f:
    Poisson solver、transverse projection、magnetic field、transverse E solver

libmpsort2.f / ompplib2.f90:
    tile particle reorder / movement

Periodic/mp_pfields2.f90:
    不含粒子的 field-only test，最适合用来验证你自己的 spectral field solver
```

### 2.2 不要照搬的部分

UPIC-2.0 的代码目标是 MPI/OpenMP CPU framework。你第一版 C++/CUDA-ready code 不应该照搬这些复杂性：

```text
不要先写 MPI y-partition
不要先写 non-uniform partition field manager
不要先写 Fortran packed FFT layout
不要先写完整 restart/diagnostic file system
不要先写所有 density profile 和所有 boundary option
```

你的第一版应该是：

```text
single process
single full 2D periodic domain
one FFT layout
one particle shape: CIC / linear
one or two species
minimal CSV diagnostics
```

### 2.3 必须学习的部分

UPIC 值得学习的是这些结构：

```text
1. main program 只编排流程，低层操作放进 library/module
2. particles 和 fields 分开
3. real-space field arrays 和 k-space field arrays 分开
4. particle 初始化可以先进入 linear buffer，再转成 tiled buffer
5. 每个 tile 有自己的 particle count
6. push 时记录离开 tile 的粒子，再统一 reorder
7. spectral solver 里有 form factor / Green function
8. field-only test 独立于 particle loop
```

### 2.4 UPIC 数组到 C++ 结构的映射

UPIC 的 Fortran 数组：

```text
part(idimp,maxnp)
ppart(idimp,nppmx0,mxyp1)
kpic(mxyp1)
ncl(8,mxyp1)
ihole(2,ntmaxp+1,mxyp1)
qe(nxe,nypmx), qi(nxe,nypmx)
fxye(ndim,nxe,nypmx)
qt(nye,kxp), fxyt(ndim,nye,kxp)
ffc(nyh,kxp)
```

建议映射成 C++：

```text
part(idimp,maxnp)
    -> optional ParticleScratch, only for initialization/debug/restart

ppart(idimp,nppmx0,mxyp1)
    -> first version: ParticleSoA flat arrays
    -> later: ParticleTiles / CellChunks backend

kpic(mxyp1)
    -> tile_particle_count[tile]

ncl(8,mxyp1)
    -> outgoing_count[tile][direction]

ihole(2,ntmaxp+1,mxyp1)
    -> leaving_particles / holes buffer

qe, qi
    -> rho_e, rho_i, rho_total inside Sources2D

cue, cui
    -> current Jx/Jy/Jz inside Sources2D

dcu, amu
    -> DarwinSources: dJdt and momentum_flux

fxye / fxyze
    -> longitudinal electric field E_L

cus
    -> transverse electric field E_T retained between steps

exyze
    -> total electric field E = E_L + E_T

bxyze
    -> magnetic field B

qt, fxyt, cut, dcut, amut, exyt, bxyt
    -> SpectralArrays2D

ffc, ffe
    -> SpectralFormFactor / GreenFunction table
```

Fortran 的 `ppart(idimp,n,tile)` 是 array-of-components inside tile。C++/CUDA 第一版建议用 flat SoA：

```text
x[p], y[p], ux[p], uy[p], uz[p]
```

但是逻辑接口要像 UPIC 的 tiled particles：

```cpp
particles.deposit_charge(...);
particles.deposit_current(...);
particles.push(...);
particles.reorder_or_wrap(...);
```

这样以后可以把 backend 从 flat SoA 换成 tile/chunk pool，而主循环不需要改。

### 2.5 `idimp=4` 和 `idimp=5` 的选择

UPIC 的 `mpbeps2` electrostatic 主程序里：

```text
idimp = 4
particle = x, y, vx, vy
```

UPIC 的 `mpdbeps2` Darwin 主程序里：

```text
idimp = 5
particle = x, y, vx, vy, vz
```

你的目标是 2D3V electrostatic first, Darwin later，所以不要跟 `mpbeps2` 一样用 `idimp=4`。你应该从第一天就用：

```text
x, y, ux, uy, uz
```

即使第一版 electrostatic 只会更新 `ux, uy`，`uz` 也必须存在。这样外加 `B0`、Boris pusher、Darwin current `Jz`、whistler physics 都可以自然接上。

### 2.6 `mpbeps2` electrostatic 主循环

`mpbeps2.f90` 的核心流程可以压缩成：

```text
for each timestep:
    zero qe
    deposit electron charge -> qe
    add guard cells

    if mobile ions:
        zero qi
        deposit ion charge -> qi
        add guard cells

    qe = qe + qi

    FFT(qe) -> qt
    mppois2(qt, ffc) -> fxyt, electric energy
    inverse FFT(fxyt) -> fxye
    copy guard cells for fxye

    push electrons with fxye
    reorder electrons by tile

    if mobile ions:
        push ions with fxye
        reorder ions by tile

    diagnostics
```

你的第一版 C++ 主循环应该直接模仿这个顺序，但删掉 MPI guard/partition 复杂性：

```text
zero rho
deposit rho
subtract mean or add neutralizing background
R2C FFT rho -> rho_k
solve Poisson in k-space
C2R FFT E_k -> E
gather E and push particles
periodic wrap
diagnostics
```

### 2.7 `mpdbeps2` Darwin 主循环

`mpdbeps2.f90` 的核心比 electrostatic 多很多 source：

```text
cue   = current density J
dcu   = acceleration density / time derivative of current source
amu   = momentum flux
cus   = transverse electric field E_T, retained between steps
fxyze = longitudinal electric field E_L
exyze = total electric field E
bxyze = magnetic field B
```

Darwin 主循环的高层顺序是：

```text
deposit current J
deposit charge rho
rho_k -> solve E_L
J_k -> transverse projection -> solve B
add external B0
E_total = E_L + old E_T

deposit dcu and amu using E_total and B
dcu/amu -> transverse dJ/dt
solve new E_T
E_total = E_L + new E_T

optional inner iterations ndc:
    redeposit J, dcu, amu
    recompute B and E_T
    update E_total

push particles with E_total and B
reorder particles
diagnostics
```

第一版不要实现这些。但你的数据结构要能容纳这些名字，否则以后会重写。

### 2.8 `ffc` / `ffe` 的意义

UPIC 的 field solver 不只是简单 `1/k^2`。它把 spectral Green's function 和 finite-size particle shape factor 放进 form factor array：

```text
ffc:
    Poisson Green function
    particle shape factor s(k)

ffe:
    Darwin transverse electric solver Green function
    particle shape factor s(k)
```

在 `libmpfield2.f` 里，electrostatic force/charge 的形式是：

```text
E_k ~ -i k * g(k) * s(k) * rho_k
g(k) ~ affp * s(k) / k^2
s(k) = exp(-((kx ax)^2 + (ky ay)^2) / 2)
```

你的第一版可以先令：

```text
s(k) = 1
g(k) = 1 / (eps0 * k^2)
```

但代码结构上应该保留：

```cpp
struct SpectralFormFactor;
```

以后再加入 UPIC 风格的 smoothing/filter，不要把 `1/k^2` 写死在多个 solver 里。

---

## 3. 第一版的物理范围

第一阶段只写：

```text
Dimension:
    2D in space: x, y
    3D in velocity: ux, uy, uz

Boundary:
    periodic in x and y

Species:
    kinetic electrons first
    immobile neutralizing ion background first

Fields:
    electrostatic self-field Ex, Ey
    Ez = 0 in first electrostatic version
    optional external B0 reserved for next stage

Numerics:
    CIC shape first
    spectral Poisson solver
    leapfrog particle update first
    Boris pusher interface reserved

Precision:
    float for hot particle / field arrays first
    double for diagnostics and reductions
```

注意：这比 UPIC 的 `mpbeps2` electrostatic 多一个 velocity component。这样做是为了直接对齐 `mpdbeps2` 的 2-1/2D Darwin 粒子布局。

第一版暂时不要做：

```text
no open boundary
no multi-GPU
no high-order shape
no full Darwin field solve
no charge-conserving current deposition
no AMR
no complicated collision model
```

原因很简单：你现在最需要的是一个能跑通、能验证、能扩展的最小闭环。

---

## 4. 第一周从哪里开始

### Step 0: 先定 normalization

不要一边写代码一边改单位。第一版建议使用 normalized units：

```text
eps0 = 1
electron mass me = 1
elementary charge magnitude e = 1
omega_pe reference = 1
grid coordinate stored in cell units
```

粒子位置建议存成 cell coordinate：

```text
x in [0, nx)
y in [0, ny)
```

而不是物理长度。这样 deposition 和 periodic wrap 最简单：

```cpp
i = floor(x);
fx = x - i;
x += vx * dt / dx;
```

物理波数和谱求解仍然通过 `dx, dy, Lx, Ly` 计算。

### Step 1: 建 C++ 项目骨架

先写纯 C++ host reference，再接 CUDA/cuFFT。host reference 可以很慢，但它能帮你验证每个 kernel。

推荐目录：

```text
include/arcwarden/
  config.hpp
  array.hpp
  grid.hpp
  particles.hpp
  fields.hpp
  sources.hpp
  deposition.hpp
  pusher.hpp
  spectral_poisson.hpp
  diagnostics.hpp
  simulation.hpp

src/
  main.cpp
  simulation.cpp
  deposition_cpu.cpp
  pusher_cpu.cpp
  spectral_poisson_cpu.cpp
  diagnostics.cpp

tests/
  test_deposition.cpp
  test_poisson.cpp
  test_push.cpp
```

等 CPU reference 跑通后，再加：

```text
src/cuda/
  deposition.cu
  pusher.cu
  spectral_poisson.cu
  cufft_plan.cu
```

### Step 2: 先写 Poisson solver 单元测试

不要从 particle push 开始。先验证 field solver：

```text
rho(x,y) = sin(2 pi x / Lx)
```

解析解：

```text
phi_k = rho_k / k^2
Ex_k = -i kx phi_k
Ey_k = -i ky phi_k
```

你应该先确认：

```text
k index 正确
FFT normalization 正确
k = 0 mode 被正确设置为 0
Ex, Ey 的符号正确
```

### Step 3: 再写 CIC deposition

先只 deposit `rho`：

```text
particle charge -> four neighboring grid points
periodic wrap
sum(rho) equals total particle charge + background charge
```

第一版可以用 global `atomicAdd`。性能差没关系，结构先对。

### Step 4: 最后接 particle push

第一版 electrostatic leapfrog：

```text
u^{n+1/2} = u^{n-1/2} + (q/m) E^n dt
x^{n+1}   = x^n + u^{n+1/2} dt
```

虽然 electrostatic self-field 只有 `Ex, Ey`，粒子仍然存 `ux, uy, uz`。`uz` 第一版不变，但这个设计能直接接上外加 `B0` 和 Darwin/EM pusher。

---

## 5. 第一版主循环

主循环应该从第一天就长这样：

```cpp
void Simulation::step() {
    sources.zero();

    particles.deposit_charge(sources, grid);
    sources.add_neutralizing_background(params);

    field_solver.solve(fields, sources, grid);

    particles.push(fields, grid, params);
    particles.apply_periodic_boundary(grid);

    diagnostics.maybe_compute(step_id, particles, fields, sources);
}
```

以后 Darwin 版本只把中间几步扩展成：

```cpp
particles.deposit_sources(sources, grid);  // rho + J
field_solver.solve(fields, sources, grid); // electrostatic or Darwin
particles.push(fields, grid, params);      // E + B
```

不要让主循环知道 solver 内部是 Poisson、Darwin 还是 Yee。

---

## 6. 第一阶段最重要的数据结构

### 6.1 基础类型

第一版不要过度 template 化，但要把类型集中放在一个地方：

```cpp
namespace aw {

using Real = float;
using Index = int;
using Count = int;
using DiagnosticReal = double;

enum class ShapeOrder {
    CIC = 1,
    TSC = 2
};

enum class Boundary {
    Periodic
};

enum class SolverKind {
    ElectrostaticSpectral,
    DarwinSpectral,
    YeeEM
};

} // namespace aw
```

第一版 hot path 可以写死 CIC、periodic、2D3V。枚举存在的意义是让接口以后不需要改名。

### 6.2 Grid2D

`Grid2D` 是全局几何信息，应该很小，可以按值传入 kernel：

```cpp
struct Grid2D {
    int nx = 0;
    int ny = 0;

    Real dx = 1;
    Real dy = 1;
    Real lx = 1;
    Real ly = 1;

    int real_size() const { return nx * ny; }
    int complex_ny() const { return ny / 2 + 1; }
    int complex_size() const { return nx * complex_ny(); }

    int real_index(int i, int j) const {
        i = wrap(i, nx);
        j = wrap(j, ny);
        return i * ny + j;
    }

    int complex_index(int i, int jk) const {
        return i * complex_ny() + jk;
    }

    static int wrap(int i, int n) {
        i %= n;
        return i < 0 ? i + n : i;
    }
};
```

建议 real-space layout 用 row-major：

```text
index = i * ny + j
```

只要全代码一致即可。以后如果为了 CUDA coalescing 想换 layout，也只改 `Grid2D` 和访问 helper。

### 6.3 Device/Host Array wrapper

不要在 physics code 里到处出现 `new/delete`、`cudaMalloc/cudaFree`。

CPU reference 可以先写：

```cpp
template<class T>
class HostArray {
public:
    explicit HostArray(std::size_t n = 0) : data_(n) {}

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    std::size_t size() const { return data_.size(); }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    void resize(std::size_t n) { data_.resize(n); }
    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

private:
    std::vector<T> data_;
};
```

CUDA 版本再写同样接口的 `DeviceArray<T>`：

```cpp
template<class T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t n);
    ~DeviceArray();

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept;
    DeviceArray& operator=(DeviceArray&& other) noexcept;

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return n_; }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};
```

这样 CPU/GPU backend 可以共享大部分上层接口。

### 6.4 SpeciesParams

species 的常数不要重复存在每个粒子里：

```cpp
struct SpeciesParams {
    Real charge = -1;
    Real mass = 1;
    Real weight = 1;       // physical particles per macro-particle
    Real qm() const { return charge / mass; }
};
```

第一版只有 electrons：

```text
charge = -1
mass = 1
weight = n0 * cell_volume / particles_per_cell
```

以后加入 mobile ions 时，再创建第二个 `ParticleSoA`。

### 6.5 ParticleSoA

粒子必须用 SoA，不要用 AoS。

推荐第一版：

```cpp
struct ParticleSoA {
    HostArray<Real> x;
    HostArray<Real> y;
    HostArray<Real> ux;
    HostArray<Real> uy;
    HostArray<Real> uz;

    HostArray<int> cell;   // 第一版可选，但建议保留

    SpeciesParams species;
    int count = 0;

    void resize(int n) {
        count = n;
        x.resize(n);
        y.resize(n);
        ux.resize(n);
        uy.resize(n);
        uz.resize(n);
        cell.resize(n);
    }
};
```

为什么不用：

```cpp
struct Particle { float x, y, ux, uy, uz; };
std::vector<Particle> particles;
```

因为 GPU 上 deposition/gather/push 都会连续读取同一属性。SoA 更容易 coalescing，也更容易以后接 sorting、cell list、chunk pool。

### 6.6 position 的存法

建议存 cell coordinate：

```text
x = physical_x / dx
y = physical_y / dy
```

范围：

```text
0 <= x < nx
0 <= y < ny
```

periodic wrap：

```cpp
inline Real wrap_pos(Real x, int n) {
    x = std::fmod(x, static_cast<Real>(n));
    return x < 0 ? x + n : x;
}
```

push 时：

```cpp
x[p] += ux[p] * dt / grid.dx;
y[p] += uy[p] * dt / grid.dy;
```

这样 deposit 时没有除法：

```cpp
int i = floor(x[p]);
Real fx = x[p] - i;
```

### 6.7 为什么第一版保留 `cell`

第一版可以不排序，但保留：

```cpp
cell[p] = i * ny + j;
```

用途：

```text
debug: 检查粒子是否越界
diagnostics: 统计每个 cell PPC
future GPU: sort by cell / tile
future chunk pool: 从 flat SoA 迁移到 cell chunks
```

不要第一版就写 chunk pool。chunk pool 是优化结构，不是正确性结构。

### 6.8 Sources2D

第一版只需要 `rho`，但数据结构要为 Darwin 预留 `J`：

```cpp
struct Sources2D {
    HostArray<Real> rho;

    // Darwin / EM later
    HostArray<Real> jx;
    HostArray<Real> jy;
    HostArray<Real> jz;

    void resize(const Grid2D& grid, bool allocate_current) {
        rho.resize(grid.real_size());
        if (allocate_current) {
            jx.resize(grid.real_size());
            jy.resize(grid.real_size());
            jz.resize(grid.real_size());
        }
    }

    void zero_charge() {
        rho.fill(Real(0));
    }

    void zero_current() {
        jx.fill(Real(0));
        jy.fill(Real(0));
        jz.fill(Real(0));
    }
};
```

第一版可以 `allocate_current = false`。接口上保留 `J`，内存上不强迫浪费。

### 6.9 Fields2D

electrostatic 第一版只用 `Ex, Ey`，但 field container 应该长得像 EM code：

```cpp
struct Fields2D {
    HostArray<Real> ex;
    HostArray<Real> ey;
    HostArray<Real> ez;

    HostArray<Real> bx;
    HostArray<Real> by;
    HostArray<Real> bz;

    void resize(const Grid2D& grid, bool allocate_magnetic) {
        ex.resize(grid.real_size());
        ey.resize(grid.real_size());
        ez.resize(grid.real_size());

        if (allocate_magnetic) {
            bx.resize(grid.real_size());
            by.resize(grid.real_size());
            bz.resize(grid.real_size());
        }
    }
};
```

第一版 `ez` 可以全 0，但保留它有两个好处：

```text
1. pusher interface 从一开始就是 3V/EM-ready
2. Darwin 后面有 transverse electric field，Ez 可能不再总是 0
```

### 6.10 SpectralArrays2D

spectral solver 的复杂数组单独放，不要混进 `Fields2D`：

```cpp
struct Complex {
    Real re;
    Real im;
};

struct SpectralArrays2D {
    HostArray<Complex> rho_k;

    HostArray<Complex> ex_k;
    HostArray<Complex> ey_k;
    HostArray<Complex> ez_k;

    // Darwin / EM later
    HostArray<Complex> jx_k;
    HostArray<Complex> jy_k;
    HostArray<Complex> jz_k;
    HostArray<Complex> bx_k;
    HostArray<Complex> by_k;
    HostArray<Complex> bz_k;

    void resize(const Grid2D& grid, bool allocate_darwin) {
        const int n = grid.complex_size();
        rho_k.resize(n);
        ex_k.resize(n);
        ey_k.resize(n);
        ez_k.resize(n);
        if (allocate_darwin) {
            jx_k.resize(n);
            jy_k.resize(n);
            jz_k.resize(n);
            bx_k.resize(n);
            by_k.resize(n);
            bz_k.resize(n);
        }
    }
};
```

CPU reference 可以用你自己的 naive DFT 或 FFTW；GPU 版本用 cuFFT 的 `R2C/C2R` layout。

### 6.11 SpectralFormFactor

UPIC 的 `ffc/ffe` 把 Green's function 和 particle shape factor 放在一起。C++ 里建议拆清楚：

```cpp
struct SpectralFormFactor {
    HostArray<Real> shape;          // s(k)
    HostArray<Real> poisson_green;  // 1 / (eps0 * k^2), or UPIC-style affp*s/k^2
    HostArray<Real> darwin_green;   // later: transverse E denominator

    Real ax = 0;
    Real ay = 0;

    void resize(const Grid2D& grid, bool allocate_darwin) {
        const int n = grid.complex_size();
        shape.resize(n);
        poisson_green.resize(n);
        if (allocate_darwin) {
            darwin_green.resize(n);
        }
    }
};
```

第一版可以：

```text
shape(k) = 1
poisson_green(k) = 1 / (eps0 * k^2)
```

但是接口保留 `shape`，以后加入 UPIC 风格 smoothing：

```text
shape(k) = exp(-((kx ax)^2 + (ky ay)^2) / 2)
```

### 6.12 SimulationParams

所有 runtime 参数集中放：

```cpp
struct SimulationParams {
    Real dt = 0.1f;
    int nsteps = 1000;

    int particles_per_cell = 100;
    Real density0 = 1.0f;
    Real temperature = 0.01f;

    bool neutralizing_background = true;
    bool external_b0_enabled = false;
    Real b0x = 0;
    Real b0y = 0;
    Real b0z = 0;

    ShapeOrder shape = ShapeOrder::CIC;
    Boundary boundary = Boundary::Periodic;
    SolverKind solver = SolverKind::ElectrostaticSpectral;
};
```

不要把 `dt`、`dx`、`q/m` 这些东西散落在 kernel 参数里。上层传一个 compact params，底层只读。

---

## 7. CIC deposition 设计

第一版使用 nodal collocated mesh：

```text
grid point (i,j) at x = i dx, y = j dy
i = 0 ... nx - 1
j = 0 ... ny - 1
periodic, no duplicated endpoint
```

对每个 particle：

```cpp
int i0 = floor(xp);
int j0 = floor(yp);
Real fx = xp - i0;
Real fy = yp - j0;

int i1 = i0 + 1;
int j1 = j0 + 1;

Real wx0 = 1 - fx;
Real wx1 = fx;
Real wy0 = 1 - fy;
Real wy1 = fy;
```

deposit:

```cpp
rho(i0,j0) += q * w * wx0 * wy0 / cell_volume;
rho(i1,j0) += q * w * wx1 * wy0 / cell_volume;
rho(i0,j1) += q * w * wx0 * wy1 / cell_volume;
rho(i1,j1) += q * w * wx1 * wy1 / cell_volume;
```

periodic wrap 由 `Grid2D::real_index()` 处理。

第一版测试：

```text
1 particle total charge conserved
many particles total charge conserved
uniform electron + uniform ion background -> rho_k nonzero modes near noise level
```

---

## 8. Spectral Poisson solver 设计

第一版每步：

```text
rho real -> rho_k
rho_k(k=0) = 0
for each k:
    phi_k = poisson_green(k) * shape(k) * rho_k
    Ex_k = -i kx phi_k
    Ey_k = -i ky phi_k
    Ez_k = 0
Ex_k, Ey_k -> Ex, Ey real
normalize inverse FFT by 1 / (nx * ny)
```

第一版可以让：

```text
shape(k) = 1
poisson_green(k) = 1 / (eps0 * k^2)
```

但函数签名建议从一开始接收 `SpectralFormFactor`，因为 UPIC 的 `mppois2` 实际上就是通过 `ffc` 使用 Green's function 和 finite-size shape factor。

k-space kernel 的逻辑：

```cpp
void solve_poisson_k(SpectralArrays2D& spec,
                     const SpectralFormFactor& form,
                     const Grid2D& grid) {
    for (int i = 0; i < grid.nx; ++i) {
        Real kx = physical_kx(i, grid);
        for (int jk = 0; jk < grid.complex_ny(); ++jk) {
            Real ky = physical_ky(jk, grid);
            Real k2 = kx * kx + ky * ky;
            int m = grid.complex_index(i, jk);

            if (k2 == 0) {
                spec.ex_k[m] = {0, 0};
                spec.ey_k[m] = {0, 0};
                spec.ez_k[m] = {0, 0};
                continue;
            }

            Complex rho = spec.rho_k[m];
            Real green = form.poisson_green[m];
            Real shape = form.shape[m];
            Complex phi = rho * (green * shape);

            spec.ex_k[m] = multiply_minus_i_k(phi, kx);
            spec.ey_k[m] = multiply_minus_i_k(phi, ky);
            spec.ez_k[m] = {0, 0};
        }
    }
}
```

`physical_kx`:

```cpp
inline Real physical_kx(int i, const Grid2D& g) {
    int n = (i <= g.nx / 2) ? i : i - g.nx;
    return Real(2) * pi<Real>() * n / g.lx;
}
```

`ky` 对 R2C 的 last dimension 只需要：

```cpp
ky = 2 pi jk / Ly, jk = 0 ... ny/2
```

注意：

```text
cuFFT inverse transform 不会自动除以 nx * ny
k = 0 mode 必须置 0，否则 periodic Poisson 没有唯一解
总电荷不为 0 时，先加 neutralizing background 或减去 mean(rho)
```

---

## 9. Field gather 和 pusher

gather 使用和 deposit 同一套 CIC weights。第一版：

```cpp
Ep = sum_grid E(i,j) * weight(i,j)
```

electrostatic push：

```cpp
ux[p] += species.qm() * ex_p * dt;
uy[p] += species.qm() * ey_p * dt;
// uz unchanged in pure electrostatic

x[p] += ux[p] * dt / grid.dx;
y[p] += uy[p] * dt / grid.dy;
```

但函数签名从一开始就设计成 EM-ready：

```cpp
void push_particles(ParticleSoA& particles,
                    const Fields2D& fields,
                    const Grid2D& grid,
                    const SimulationParams& params);
```

下一阶段加入 `B0` 后，把内部替换成 Boris pusher：

```text
half E kick
B rotation
half E kick
position update
```

主循环不需要改。

---

## 10. Diagnostics 必须第一版就写

不要等代码变大后再补 diagnostics。第一版至少写：

```text
total charge
mean rho
kinetic energy
electric field energy
total energy
max |E|
max particle speed
particles outside domain count
rho spectrum optional
```

建议：

```cpp
struct Diagnostics {
    DiagnosticReal charge_total = 0;
    DiagnosticReal kinetic_energy = 0;
    DiagnosticReal electric_energy = 0;
    DiagnosticReal total_energy = 0;
    DiagnosticReal max_abs_e = 0;
};
```

电场能量：

```text
0.5 * eps0 * sum(E^2) * dx * dy
```

动能：

```text
0.5 * mass * weight * sum(ux^2 + uy^2 + uz^2)
```

第一版每 10 或 100 步输出 CSV：

```text
step,time,charge,ke,ee,total,max_e,max_u
```

---

## 11. 正确性验证顺序

按这个顺序做，不要跳：

### Test 1: Grid indexing

```text
wrap(-1,n) == n-1
wrap(n,n) == 0
real_index periodic correct
complex_size == nx * (ny/2 + 1)
```

### Test 2: CIC total charge

单粒子、多粒子、边界附近粒子都要测：

```text
sum(rho) * dx * dy == total macro charge
```

### Test 3: Poisson analytic mode

输入：

```text
rho = sin(2 pi x / Lx)
```

验证：

```text
Ex amplitude and phase
Ey near zero
```

### Test 4: quiet plasma no field

均匀电子 + neutralizing background，速度为 0：

```text
rho mean removed
E remains near roundoff / particle noise
particles do not move if velocity is zero
```

### Test 5: cold Langmuir oscillation

给电子一个小位移扰动：

```text
frequency near omega_pe
energy exchange between KE and EE
total energy reasonably bounded
```

### Test 6: two-stream instability

两个 counter-streaming electron beams：

```text
early growth rate compared with theory / reference
phase space forms vortices
```

只有 Test 1-5 过了，才开始 GPU 优化。

---

## 12. 以后如何扩展到 Darwin

第一阶段这些设计是专门为 Darwin 留口子的：

```text
ParticleSoA:
    already 2D3V

Sources2D:
    rho now
    Jx/Jy/Jz later

Fields2D:
    Ex/Ey now
    Ez/Bx/By/Bz later

SpectralArrays2D:
    rho_k now
    J_k, E_k, B_k later

Pusher:
    function signature already accepts E and B

FieldSolver:
    electrostatic solver can be replaced by Darwin solver
```

Darwin stage 的主变化：

```text
deposit rho and J
rho_k gives longitudinal electric field E_L
J_k is projected into transverse current J_T
solve magnetostatic / inductive fields in k-space
construct E = E_L + E_T
construct B
push particles with E and B
```

k-space projection helper 要提前设计：

```cpp
struct KVector {
    Real kx;
    Real ky;
    Real kz; // zero for 2D spatial code
    Real k2;
};

inline Vec3Complex longitudinal_project(Vec3Complex a, KVector k);
inline Vec3Complex transverse_project(Vec3Complex a, KVector k);
inline Real spectral_filter(KVector k, FilterParams params);
```

2D3V 的含义是：

```text
spatial k = (kx, ky, 0)
vector fields still have x, y, z components
current J has Jx, Jy, Jz
B has Bx, By, Bz
```

所以不要把 vector field 写成只有两个 component。

---

## 13. CUDA/GPU 路线

GPU 优化分三层做。

### GPU Level 1: correctness baseline

```text
flat ParticleSoA
one thread per particle
global atomicAdd for rho
cuFFT R2C/C2R
one thread per k mode for Poisson
one thread per particle for gather/push
```

这个版本不快，但容易验证。

### GPU Level 2: tile shared deposition

```text
sort particles by cell or tile occasionally
one CUDA block owns one tile
deposit into shared memory tile buffer
flush tile buffer to global rho
```

这是常规高性能 PIC baseline。

### GPU Level 3: high-PPC / Darwin-oriented particle container

当你进入 whistler / chorus / high-PPC 问题，再考虑：

```cpp
constexpr int CHUNK = 128;

struct ParticleChunk {
    Real x[CHUNK];
    Real y[CHUNK];
    Real ux[CHUNK];
    Real uy[CHUNK];
    Real uz[CHUNK];
    uint16_t count;
    int next;
};

struct CellList {
    int first_chunk;
    int count;
};
```

但这不是第一版。第一版先用 flat SoA，把接口设计好：

```cpp
particles.deposit_charge(...)
particles.push(...)
particles.apply_periodic_boundary(...)
```

以后 `ParticleSoA` backend 可以换成 chunk pool，而主循环不变。

---

## 14. 推荐 C++ 接口形状

不要一开始用 inheritance-heavy OOP。推荐轻量 class + free function：

```cpp
class Simulation {
public:
    Simulation(Grid2D grid, SimulationParams params);

    void initialize();
    void step();
    void run();

private:
    Grid2D grid_;
    SimulationParams params_;

    ParticleSoA electrons_;
    Sources2D sources_;
    Fields2D fields_;
    SpectralArrays2D spectral_;
    Diagnostics diagnostics_;
};
```

solver 可以先写成普通 class：

```cpp
class ElectrostaticSpectralSolver {
public:
    void solve(Fields2D& fields,
               Sources2D& sources,
               SpectralArrays2D& spectral,
               const Grid2D& grid);
};
```

以后 Darwin 加：

```cpp
class DarwinSpectralSolver {
public:
    void solve(Fields2D& fields,
               Sources2D& sources,
               SpectralArrays2D& spectral,
               const Grid2D& grid,
               Real dt);
};
```

如果后面想做 compile-time dispatch，可以再 template 化。第一版不要把 template 搞复杂。

---

## 15. 推荐实现里程碑

### Milestone 0: 编译骨架

输出：

```text
CMake builds
main.cpp can create Grid2D and SimulationParams
unit test framework works
```

### Milestone 1: CPU reference Poisson

输出：

```text
rho analytic input
Ex/Ey analytic comparison
CSV or small printout
```

### Milestone 2: CPU reference PIC loop

输出：

```text
ParticleSoA
CIC deposit
spectral Poisson
CIC gather
leapfrog push
periodic boundary
diagnostics CSV
```

### Milestone 3: physical validation

输出：

```text
Langmuir oscillation
two-stream instability
energy diagnostics plot
rho/E spectrum plot
```

### Milestone 4: CUDA baseline

输出：

```text
DeviceArray
CUDA deposition with atomicAdd
cuFFT wrapper
CUDA k-space Poisson kernel
CUDA gather/push
CPU vs GPU small-case comparison
```

### Milestone 5: magnetized electrostatic

输出：

```text
external B0
Boris pusher
gyro motion test
E x B drift test
upper-hybrid / magnetized plasma validation
```

### Milestone 6: Darwin preparation

输出：

```text
Jx/Jy/Jz deposition
J_k FFT
transverse projection helper
k-space filter helper
continuity diagnostic
```

### Milestone 7: first Darwin solver

输出：

```text
E_L from rho
B from transverse current / vector potential
inductive E_T
whistler dispersion test
compare with electrostatic and theory
```

---

## 16. 第一版最容易犯的错误

1. FFT inverse 忘记除以 `nx * ny`。

2. `k = 0` mode 没有特殊处理。

3. particle position 一会儿用 physical units，一会儿用 cell units。

4. deposit 和 gather 的 CIC stencil 不一致。

5. 忘记 neutralizing ion background，导致 periodic Poisson 不可解。

6. 把 2D3V 写成 2D2V，后面 Darwin/whistler 要推翻。

7. 太早写 chunk pool、shared memory deposition、CUDA Graph。

8. 没有 diagnostics，导致 simulation 看起来在跑但不知道对不对。

---

## 17. 最小可运行 demo 的目标参数

先用小网格：

```text
nx = 64
ny = 64
particles_per_cell = 16 or 64
dt = 0.05 / omega_pe
dx = dy = 1
periodic
electrons + neutralizing background
```

初始速度：

```text
Maxwellian with small thermal velocity
or zero velocity for cold plasma test
```

第一批 demo：

```text
demo_poisson_sine
demo_quiet_plasma
demo_langmuir
demo_two_stream
```

这些 demo 比一开始写复杂 input parser 更重要。

---

## 18. 最终架构原则

你要避免的是写出一个只能跑 electrostatic 的 dead-end toy code。关键原则：

```text
1. particle always 2D3V
2. fields always vector-capable
3. sources reserve rho + J
4. solver is replaceable
5. k-space operations are centralized
6. diagnostics are first-class
7. first backend is simple flat SoA
8. optimized particle container comes after validation
```

一句话：

```text
先用最朴素的 flat-SoA + CIC + spectral Poisson 跑通 2D3V electrostatic PIC，
同时把 Sources/Fields/Pusher/Solver 接口设计成 Darwin-ready。
```

---

## 19. 参考资料

这些资料用于提炼架构方向：

1. UCLA Plasma Simulation Group, UPIC-2.0 `mpbeps2` directory.  
   https://github.com/UCLA-Plasma-Simulation-Group/UPIC-2.0/tree/master/mpbeps2

2. UPIC-2.0 `mpbeps2.source/mpbeps2.f90`, electrostatic reference application.  
   https://github.com/UCLA-Plasma-Simulation-Group/UPIC-2.0/blob/master/mpbeps2/mpbeps2.source/mpbeps2.f90

3. UPIC-2.0 `mpbeps2.source/mpdbeps2.f90`, Darwin reference application.  
   https://github.com/UCLA-Plasma-Simulation-Group/UPIC-2.0/blob/master/mpbeps2/mpbeps2.source/mpdbeps2.f90

4. UPIC-2.0 `mpbeps2.source/libmpfield2.f`, spectral field solver routines.  
   https://github.com/UCLA-Plasma-Simulation-Group/UPIC-2.0/blob/master/mpbeps2/mpbeps2.source/libmpfield2.f

5. UPIC-2.0 `mpbeps2/Periodic/mp_pfields2.f90`, field-only periodic test.  
   https://github.com/UCLA-Plasma-Simulation-Group/UPIC-2.0/blob/master/mpbeps2/Periodic/mp_pfields2.f90

6. Peicheng Yu et al., "Modeling of Laser wakefield acceleration in Lorentz boosted frame using EM-PIC code with spectral solver", arXiv:1310.2622.  
   https://arxiv.org/abs/1310.2622

7. Xin An et al., "Unified view of nonlinear wave structures associated with whistler-mode chorus", arXiv:1901.00953.  
   https://arxiv.org/abs/1901.00953

8. Dmytro Sydorenko et al., "Improved algorithm for a two-dimensional Darwin particle-in-cell code", arXiv:2409.19559.  
   https://arxiv.org/abs/2409.19559

9. V. K. Decyk, UCLA PIC Framework / UPIC reference as cited by UPIC-EMMA papers: Computer Physics Communications 177, 95 (2007).
