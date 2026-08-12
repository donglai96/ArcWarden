# 第一步：2D3V 电静电谱方法 GPU PIC —— 类设计计划

> 本文是 Darwin 项目的**第一步**实现计划：2D3V、周期边界、电子动理学、离子固定中性背景、谱方法（FFT）求解 Poisson、CIC 插值、Boris pusher（含可选外加均匀 B₀）。
> 纯 GPU、单卡、C++20 + CUDA。
> 参考：本仓库 `gpu_darwin_pic_plan.md` 路线（Level 1 → Level 2）与 UPIC-2.0 `mpbeps2` 模块划分
> （`mpbeps2` 静电主程序 + `modmpush2` / `modmpfft2` / `modmpfield2` 等模块）。

---

## 0. 本步目标界定

**只做这一件事**：2D3V、周期边界、电子动理学、离子固定中性背景、谱方法（FFT）求解 Poisson、CIC 插值、Boris pusher（含可选外加均匀 B₀，为 Level 2 留口）。

```
deposit ρ → FFT → 谱方法解 φ/E → IFFT → gather E → push → migrate → diagnostics
```

对应 UPIC 的 `mpbeps2`：电荷沉积 → FFT → Poisson 解 → 反 FFT → 粒子推进。
我们把它做成可向 Darwin（加 J、dcu、amu 沉积和横向场求解）平滑扩展的骨架。

> 关键设计原则（来自 plan §10/§18）：**物理参数运行时可配，hot kernel 编译期固定**；
> **不要让 99% 普通粒子为 1% 异常粒子付分支成本**。

---

## 0.1 架构风格分层（最重要的设计哲学）

> 这是回答「OOP / C++ 写得够不够 sophisticated」的核心：**GPU PIC 的复杂度不在继承树，而在 (1) RAII 所有权、(2) 编译期 policy 模板、(3) 数据布局 / host-device 边界。** 深继承 + 满屏 virtual 在 hot path 上是反模式。每一层用对应的风格，而不是所有东西都套同一种 OOP。

| 层 | 代表类型 | 风格 | 为什么这样 | 本文章节 |
|---|---|---|---|---|
| 资源 / 所有权 | `DeviceArray`、`CufftPlan2D`、`CudaStream`、`SpectralEngine` | **正经 OOP + RAII**，move-only、禁拷贝 | 生命周期 / 所有权是真问题，必须封装 | §2, §7.1 |
| 编译期配置 | `SimConfig`、`DepositPolicy` | **template policy**，编译期 dispatch | hot path 要零开销、无虚调用、kernel 内联 | §3 |
| 数据容器 | `Particles`、`Sources`、`Fields`、`SpectralWorkspace`、`*Views` | **POD-ish struct + SoA + owning/View 分离** | device 边界要可值传 POD；SoA 要 coalescing | §2.1, §5, §6 |
| 算法 / kernel | `deposit_*_kernel`、`boris_push_kernel`、k-solve | **free `__global__` + 模板 policy** | 内联、无虚表、无间接跳转 | §8 |
| host 编排 | `Simulation`、solver 选择(`std::variant`) | **组合式 OOP**；host-only `std::variant`(或 virtual) | 每步一次，清晰优先；多态开销可忽略 | §7.2, §10 |

**继承基本只出现在最后一层，而且很浅。** 真正难、也真正现代 C++ 的是上面三层——owning/non-owning 分离、policy 模板、SoA 布局。这比深继承难得多。

---

## 1. 目录与分层

```
ArcWarden/
  include/pic/
    config.hpp          // 编译期+运行期配置
    device_array.hpp    // RAII GPU 内存
    cuda_utils.hpp      // 错误检查、stream、event
    grid.hpp            // 实空间/k空间网格几何
    particles.hpp       // 粒子容器
    fields.hpp          // 场容器(实空间+k空间)
    fft.hpp             // cuFFT 封装
    depositor.hpp       // ρ(后续 J) 沉积接口
    pusher.hpp          // Boris/Darwin pusher 接口
    solver.hpp          // FieldSolver 基类
    solver_es.hpp       // 静电谱求解器(本步)
    diagnostics.hpp
    simulation.hpp      // 主循环编排
  src/
    main.cpp
    particles.cu
    deposit.cu
    push.cu
    solver_es.cu
    fft.cu
    diagnostics.cu
  tests/                // 单元 + 物理验证
```

分层思想（plan §9）：**局部粒子引擎 + 全局谱场引擎**。粒子 kernel 局部，FFT/k-space 全局。

---

## 2. 基础设施类（先写，最稳）

### 2.1 `DeviceArray<T>` + `DeviceView<T>` —— owning / non-owning 分离

这是本项目**最重要的一个 OOP 决策**：把「拥有 + 管生命周期」和「device 上被值传进 kernel」拆成两个类型。

```cpp
// ---- owning：host 端、有行为、RAII、move-only ----
template<class T>
class DeviceArray {
public:
    explicit DeviceArray(size_t n);                     // cudaMalloc(普通,长生命周期)
    ~DeviceArray();                                     // cudaFree
    DeviceArray(DeviceArray&&) noexcept;                // 可移动
    DeviceArray& operator=(DeviceArray&&) noexcept;
    DeviceArray(const DeviceArray&) = delete;           // 禁拷贝(所有权唯一)
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* data();  const T* data() const;
    size_t size() const;
    void zero(cudaStream_t s);                          // cudaMemsetAsync

    DeviceView<T>      view();         // 传进 kernel 的轻量句柄
    DeviceView<const T> view() const;
private:
    T* ptr_ = nullptr; size_t n_ = 0;
};

// ---- non-owning：device 端、POD、trivially copyable、可值传 ----
template<class T>
struct DeviceView {
    T*  ptr = nullptr;
    int n   = 0;
    __device__ T&       operator[](int i)       { return ptr[i]; }
    __device__ const T& operator[](int i) const { return ptr[i]; }
};
```

要点：

- **`DeviceArray` 永远不进 kernel**；进 kernel 的只有 `DeviceView`（POD、可按值拷贝、无析构、无所有权）。
- 这样 host 端有清晰的 RAII / 所有权语义，device 端有零开销的裸指针访问，两边职责不混。
- 容器（`Particles`/`Sources`/`Fields`/`SpectralWorkspace`）持有 `DeviceArray`；它们的 `views()` 方法打包出一组 `DeviceView` 传给 kernel（见 §5/§6）。

**分配策略按生命周期分两类（不要一刀切 async）：**

| 类别 | 例子 | 分配方式 | 理由 |
|---|---|---|---|
| 长生命周期（init 分配一次、teardown 释放） | 粒子、`Sources`、`Fields`、`SpectralWorkspace` | **`DeviceArray` = 普通 `cudaMalloc/cudaFree`** | 简单、无 stream 生命周期陷阱、与 CUDA Graph capture 不打架 |
| 每步临时 scratch | 归约缓冲、临时排序 buffer | 以后另建 `AsyncScratchBuffer`（`cudaMallocAsync`+mempool） | 真正需要 stream-ordered 复用时才上 |

> 之前「全程 async」是过度设计：`cudaFreeAsync` 要求析构时 stream/mempool 仍存活，长存数组用它反而引入生命周期不确定性。**v1 几乎没有 per-step scratch**（FFT workspace 长存、归约用预分配 buffer），所以 `AsyncScratchBuffer` 列「以后」，v1 不建。`DeviceArray` 一律普通 malloc/free。

### 2.2 `CudaStream` / `CudaEvent` / `CUDA_CHECK`

封装 stream、event、统一错误检查宏。后续 CUDA Graph capture（plan §12）依赖单一 stream。

### 2.3 `CufftPlan2D` —— FFT 封装

```cpp
class CufftPlan2D {
public:
    CufftPlan2D(int nx, int ny, int batch);   // 一次创建，反复用
    void r2c(const float* in, cufftComplex* out, cudaStream_t s);
    void c2r(const cufftComplex* in, float* out, cudaStream_t s);
private:
    cufftHandle plan_r2c_, plan_c2r_;
};
```

**每步绝不重建 plan、绝不重复分配临时 buffer**（plan §11）。

**R2C layout 现在就钉死（写进 `grid.hpp` / `fft.hpp` 顶部注释）：**

```text
real layout  = [ny][nx]          // 行优先, x 连续
real index   = j * nx + i        // 与 Grid::idx 一致
R2C output   = [ny][nx/2 + 1]    // cuFFT 折叠最快维(x) => nkx = nx/2+1
cuFFT plan   = cufftPlan2d(&p, ny, nx, ...)   // 注意顺序(ny,nx): 最后一维 nx 最快
```

x 是连续维 → cuFFT 折叠 x → `nkx = nx/2+1`（与 §4 `KGrid` 自洽）。Darwin 加 batched FFT / transverse projection 时全靠这套约定一致，务必锁死。

---

## 3. 配置类 `SimConfig`

**这是体现 modern C++ 功力的核心层**：用编译期 policy 把 hot path 的策略（shape、沉积方式、精度）绑死，让 kernel 内联、零虚调用；用运行期结构体放物理参数。**sophistication 在这里，不在继承树。**

```cpp
enum class ShapeOrder { CIC, TSC };

// ---- 沉积策略：编译期 policy，不是 device 上的 virtual ----
struct AtomicGlobalDeposit { /* v0: 每线程一粒子 + global atomicAdd */ };
struct SharedTileDeposit   { /* v1: 一个 CTA 一个 tile + shared atomic */ };
struct CellOwnedDeposit    { /* v2: 一个 warp 一个 cell, high-PPC reduction */ };

// ---- 编译期配置：把 hot path 决策全部固定下来 ----
template<int Dim, int VelDim, ShapeOrder Shape,
         typename Real, bool HasB0,
         class DepositPolicy = AtomicGlobalDeposit>
struct SimConfig {
    static constexpr int dim  = Dim;        // 2
    static constexpr int vdim = VelDim;     // 3
    static constexpr ShapeOrder shape = Shape; // CIC
    static constexpr bool has_b0 = HasB0;
    using real    = Real;                   // float
    using deposit = DepositPolicy;          // 换沉积实现只改这一行
};
using Cfg = SimConfig<2,3,ShapeOrder::CIC,float,true, AtomicGlobalDeposit>;

enum class NormMode { OmegaPeUnity };   // ωpe=1, ε0=1, me=1, |e|=1

// 运行期：物理/数值（device kernel 按值接收）。
// 注意：几何(nx,ny,dx,dy,Lx,Ly,idx)全归 §4 Grid，这里不重复存，避免两处 drift。
struct RunParams {
    double dt;
    double n0, vth, vd;   // 密度、热速、漂移
    double wpe, wce;      // 等离子体/回旋频率
    float  B0[3];         // 外加均匀磁场方向
    int    ppc;           // particles per cell
    long   nsteps, dump_every;

    // —— 归一化闭合所需（缺这些 ωpe=1 很容易不闭合，见 §14.2）——
    NormMode norm = NormMode::OmegaPeUnity;
    double eps0   = 1.0;
    double qm     = -1.0;     // q/m (电子 = -1)
    double weight = 1.0;      // 宏粒子权重 = n0*dx*dy/ppc，决定 ρ 标度
    double ax, ay;            // particle shape 平滑尺度 s(k)=exp(-(kx·ax)²/2 …)
    double k_filter;          // 谱滤波截止(plan §4.2)
    unsigned long rng_seed;   // 粒子加载可复现
};
```

**归一化 contract（开工前钉死，保证 ωpe=1）**：

```text
weight = n0·dx·dy / ppc            # 宏粒子权重(物理粒子/宏粒子)
rho   += q·weight·CIC / (dx·dy)    # 电荷密度(CIC 权重和=1)
phi_k  = rho_k / (eps0·k²)         # 周期 Poisson; k=0 与 Nyquist 置零(§7.2)
E_k    = -i·k·phi_k                # = -i·k·rho_k/(eps0·k²)
C2R 后 × 1/(nx·ny)                 # cuFFT 不自动归一化
```

自洽性验算（均匀情形）：总电荷 = `ppc·nx·ny · q · (n0·dx·dy/ppc)` = `q·n0·Area` → `ρ = q·n0`；配 `eps0=me=|e|=1` 得 `ωpe² = n0 = 1`。这套式子要逐行写进 `solver_es` 与 `deposit` 注释，别让标度散落在各 kernel。

**速度/动量语义钉死**：粒子分量 `ux,uy,uz` 一律按**动量** `u = γv` 存储，第一版非相对论令 `γ≡1`。原因：whistler/chorus 的共振电子可达相对论能量，retrofit 相对论代价高，所以一开始就用 `u=γv` 语义、文档写死，而不是「先 v 再改 u」。gather→push 内部用 `v = u/γ` 取速度。

为什么用 template policy 而不是 virtual：

- device 上虚函数表 + 虚调用是性能灾难，且跨 host/device 边界很别扭。
- policy 在编译期选定 → kernel **完全内联**、无分支、无间接跳转（plan §10「hot kernels compile-time fixed」）。
- 换沉积实现（v0→v1→v2）只改 `Cfg` 的一个类型参数，**主循环和 kernel 签名都不动**。

Hot path 固定：2D3V、CIC、周期、float 粒子/场、double 诊断、单卡。

---

## 4. 网格类 `Grid` 与 `KGrid`

```cpp
struct Grid {       // 实空间
    int nx, ny;
    double dx, dy, Lx, Ly;
    __host__ __device__ int idx(int i,int j) const { return j*nx+i; }
};
struct KGrid {      // k空间(R2C 半谱)
    int nkx, nky;   // nkx = nx/2+1
    double dkx, dky;
    __device__ double kx(int i) const;
    __device__ double ky(int j) const;  // 注意 j>ny/2 取负
    __device__ double k2(int i,int j) const;
};
```

`Grid` 是**几何唯一真源**（nx,ny,dx,dy,Lx,Ly + `idx`）：deposit/push/migrate/diagnostics 都收它，`RunParams` 不再重复存几何（§3），避免两处 drift。`KGrid` 负责 `kx, ky, k2`，供 Poisson `φ_k = ρ_k / (ε0 k²)`、`E_k = -i k φ_k`、以及谱 filter（plan §4.2）；由 `SpectralEngine` 持有（§7.1）。

---

## 5. 粒子容器 `Particles`（本步关键决策）

**第一版建议：SoA 扁平数组 + cell 索引**，**先不做 chunk pool**（plan §7.1：第一版不要每步 full sort）。
但接口预留 migrate，方便后续换成 chunk pool。

```cpp
// 传进 kernel 的 POD 句柄包(只有 DeviceView,无所有权)
struct ParticleViews {
    DeviceView<float> x, y, ux, uy, uz;
    DeviceView<int>   cell;
    int n;
};

// 只管 storage / 生命周期 / 加载 / 迁移；不含沉积/推进算法
struct Particles {
    DeviceArray<float> x, y;          // 位置(cell 单位, plan §4 Step 0)
    DeviceArray<float> ux, uy, uz;    // 动量 u=γv(γ≡1 in v1), 见 §3
    DeviceArray<int>   cell;          // 所属 cell（排序/沉积用）
    size_t n;

    ParticleViews views();            // 打包 DeviceView 给 kernel

    void initialize(const RunParams&, const Grid&, cudaStream_t); // 加载 + 半步回退
    void migrate(const Grid&, cudaStream_t);         // 第一版=周期 wrap + 重算 cell
};
```

**leapfrog 初始化半步回退（区分 B0）**：leapfrog 要求 `u` 在半步、`x` 在整步，初始化须把 `u` 回退半步：
- `B0=0`：回退 = 用初始 E 做 `−dt/2` 的静电 kick。
- `B0≠0`：**不能只做 E kick**，要做 **backward half Boris**（含 B 旋转），否则 §11 S10（E×B、upper-hybrid）初值就错。
- v1 若先只验 `B0=0` 的能量守恒，可暂时只实现静电回退，但 `initialize` 接口要预留 B0 分支。
不做回退的话能量守恒曲线第一步就有系统性 offset，会被误判为格式不守恒。

**职责边界（重要）**：`Particles` **只负责** storage / `initialize` / `migrate` / `views()`。
沉积和推进**不**是 `Particles` 的方法，而是 `Depositor<Cfg>` / `Pusher<Cfg>`（§8）。
这样以后把 backend 从 flat SoA 换成 tile/chunk pool 时，`Particles` 内部变了，算法层完全不动。

`Particles` 是 host 端 owning 对象（持有 `DeviceArray`、管生命周期）；`views()` 产出 `ParticleViews`（一组 POD `DeviceView`）才是真正传进 `__global__` 的东西。容器**不**进 kernel，这就是 §2.1 owning/non-owning 分离落到业务对象上的样子。

**沉积策略分层**（plan §8，本步先做能跑的，留优化口）：

- v0：`one thread = one particle` + global `atomicAdd` ρ（最简，先验证物理对）。
- v1：tile shared memory + shared atomic（OSIRIS-like baseline，§8.1）。
- v2（后续）：warp-private buffer / cell-owned reduction（§8.2/§8.3）。

> 本步**先实现 v0/v1 把物理跑对**，把沉积 kernel 藏在 `Depositor` 接口后，后面替换不动主循环。

---

## 6. 源 / 场 / 谱 workspace —— 三分容器

不要把 `rho / E / rho_k / phi_k` 混在一个 `Fields` 里（Darwin 一上来就会爆炸）。按 UPIC 的 `qe`(源) / `fxye`(场) / `qt,fxyt`(谱) 三分法拆开（plan §5.8–5.10）：

```cpp
// 传进 kernel 的 POD 句柄包
struct SourceViews { DeviceView<float> rho; /* later: jx,jy,jz,... */ };
struct FieldViews  { DeviceView<float> Ex, Ey; /* later: Ez,Bx,By,Bz */ };

// 实空间源：deposit 写入
struct Sources {
    DeviceArray<float> rho;
    // Darwin later: jx, jy, jz, dcu*, amu*
    SourceViews views();
    void zero(cudaStream_t);              // 每步清零
};

// 实空间场：gather/push 读取
struct Fields {
    DeviceArray<float> Ex, Ey;            // 本步只需 E(静电); Ez≡0
    // Darwin later: Ez, Bx, By, Bz
    FieldViews views();
};

// k 空间 workspace：solver 内部使用，长生命周期(init 分配一次)
struct SpectralWorkspace {
    DeviceArray<cufftComplex> rho_k;
    DeviceArray<cufftComplex> phi_k;
    DeviceArray<cufftComplex> Ex_k, Ey_k;
    // Darwin later: jx_k.., ex_k/ey_k/ez_k, bx_k..
};
```

对应的 `*Views` 各自打包传 kernel。Darwin 扩展时只在各结构体里加字段，**主循环和职责边界不变**（plan §14）。`SpectralWorkspace` 由 `SpectralEngine` 持有（见 §7）。

---

## 7. `SpectralEngine` + 场求解器（FFT 所有权集中，solver 借用）

### 7.1 `SpectralEngine` —— 拥有 FFT plan 与 k 空间 workspace

FFT plan 只依赖 `(nx,ny,batch)`，**与物理无关**；Darwin 要对多个场做 batched FFT，所以 plan 必须**一份复用**，不能让每个 solver 各自持有。把它收进一个 host 端 owning 对象：

```cpp
class SpectralEngine {
    CufftPlan2D         fft_;        // owning: 一次创建反复用
    SpectralWorkspace   ws_;         // owning: k 空间缓冲(长生命周期)
    KGrid               kgrid_;
    SpectralFormFactor  ff_;         // owning: UPIC ffc/ffe 思想, v1 = shape(k)≡1
public:
    SpectralEngine(const Grid&, int batch);
    void r2c(const float* in, cufftComplex* out, cudaStream_t);
    void c2r(const cufftComplex* in, float* out, cudaStream_t);
    SpectralWorkspace& ws();
    const KGrid& kgrid() const;
    const SpectralFormFactor& form_factor() const;
};
```

- **`SpectralFormFactor`（plan §2.8 的 `ffc/ffe`）**：把 Green 函数 `g(k)` 和粒子 shape 平滑 `s(k)` 收在一处。v1 令 `s(k)=1, g(k)=1/(ε0 k²)`，但**别把 `1/k²` 写死在多个 solver 里**；以后加 smoothing/filter 只改这里。
- **batched FFT 提醒**：`Ex_k/Ey_k/Ez_k` 若是三个独立 `DeviceArray`，cuFFT batched 未必自然可用。v1 **分开调用**即可；以后要 batched，再把向量分量存成 **component-major 连续 buffer**（一块内存 `[comp][ny][nkx]`）。这是 Darwin 多场 FFT 的优化口，不是 v1 必需。

### 7.2 场求解器（host 编排层，可多态；借用 SpectralEngine，做完整 `rho→E`）

求解器是 **host 编排层**对象，每步只调一次，多态开销可忽略——这是少数可用 OOP 多态的地方。

> **v1 不必急着抽象**：第一步只有一个 solver，直接用 concrete `ElectrostaticSpectralSolver` 最干净（连 variant 都不用）。等 `DarwinSpectralSolver` 出现、真有「选哪个」需求时，再升级成下面的 `std::variant`。两种写法 `solve(...)` 签名一致，升级零摩擦。

`std::variant` 静态多态（无虚表、无堆分配）是 Darwin 阶段的目标形态。**solver 不拥有 FFT，借用 `SpectralEngine`**，并负责完整 `rho → E` 流程（避免主循环里散落 FFT 调用，消除旧 §10 里不存在的 `fft_` 成员 bug）：

```cpp
// 推荐：variant 静态多态
using AnySolver = std::variant<ElectrostaticSpectralSolver,
                               DarwinSpectralSolver /* 后续 */>;
// 调用： std::visit([&](auto& s){ s.solve(sources, fields, spectral, rp, stream); }, solver_);

class ElectrostaticSpectralSolver {           // 无需继承
public:
    // 完整 rho -> E：r2c -> k-space Poisson -> c2r -> normalize
    void solve(const Sources& src, Fields& fld,
               SpectralEngine& spectral,       // 借用,不拥有
               const RunParams& rp, cudaStream_t s);
    // 内部 launch 的 k-space kernel(编译期固定逻辑):
    //   1. ρ_k *= filter(k)               (谱滤波, plan §4.2)
    //   2. φ_k = ρ_k / (ε0 * k2)
    //      - k=0 置 0      ← 去 DC, 同时实现中性背景(见 §13)
    //      - Nyquist 置 0  ← kx=nx/2(或 ky=ny/2): 一阶导算子对纯实 Nyquist 模病态
    //   3. Ex_k = -i*kx*φ_k; Ey_k = -i*ky*φ_k
};
```

> device 上的 hot kernel **绝不**走多态。多态只在 host 端「选哪个 solver」；solver 内部 kernel 仍是编译期固定逻辑（§8）。

**两个特殊模都要置零**：
- `k=0`（k²=0）：去 DC，**同时就是周期 ES 的中性化**（见 §13），UPIC `POIS2` 标准做法。
- **Nyquist 模**（偶数长度 R2C 的 `kx=nx/2` / `ky=ny/2`）：该模纯实，乘 `i·k` 求一阶导在网格上不可表示，会引入伪高频。谱方法标准做法是对一阶导算子把 Nyquist 力置零。

---

## 8. `Depositor` 与 `Pusher`（编译期 policy，**不是** device 多态）

这是 §3 `SimConfig` policy 真正落地的地方。沉积/推进是 hot path，**绝不能用 virtual**。做法是 free `__global__` kernel + 模板 policy，策略由 `Cfg` 在编译期选定、kernel 内联展开。

```cpp
// ---- 沉积 kernel：写入 Sources(ρ),不是 Fields；策略经模板参数注入 ----
template<class Cfg, class DepositPolicy = typename Cfg::deposit>
__global__ void deposit_charge_kernel(ParticleViews p, SourceViews src,
                                      Grid g, RunParams rp);

// host 端薄封装：只负责选 launch 配置 + launch，不含多态
template<class Cfg>
struct Depositor {
    static void charge(Particles&, Sources&, const Grid&, const RunParams&,
                       cudaStream_t);
    // Darwin 时新增: current<Cfg>(...), dcu_amu<Cfg>(...)
};

// ---- 推进 kernel：内部先 gather(需要 Grid 做 idx/wrap) 再 Boris ----
template<class Cfg>
__global__ void boris_push_kernel(ParticleViews p, FieldViews f,
                                  Grid g, RunParams rp);

template<class Cfg>
struct Pusher {
    // 收 Grid：gather CIC 要 idx(i,j) + periodic wrap
    static void boris(Particles&, const Fields&, const Grid&, const RunParams&,
                      cudaStream_t);
};
```

> 注意职责：**deposit 写 `Sources`（ρ），不写 `Fields`（E）**——拆 §6 后这是两个不同容器。gather 是 `push` 的第一步、从 `Fields` 读、用与 deposit **同一套 CIC 权重**（plan §16 第 4 条）。

要点：

- **换沉积实现（v0 AtomicGlobal → v1 SharedTile → v2 CellOwned）只改 `Cfg` 的 `DepositPolicy` 类型参数**，`deposit_charge_kernel` 内部 `if constexpr (is_same<DepositPolicy, ...>)` 编译期选路，主循环与 kernel 签名都不动。
- `Cfg::has_b0` 决定 Boris 是否做磁场旋转——也是 `if constexpr`，无 B0 时整段被编译掉，零运行期分支。
- Pusher 现在就写 Boris（带 `B0`），这样 Level 2「磁化静电」零成本到位（plan §13 Level 2）。

> 对比 §7：solver 用 host 多态（每步一次，便宜）；depositor/pusher 用编译期 policy（每粒子每步，必须零开销）。**同一份代码里，多态的层和编译期固定的层是分开的——这正是这个项目「modern C++」的体现，而非继承深度。**

---

## 9. 诊断类 `Diagnostics`

```cpp
class Diagnostics {
public:
    // 解耦：只收需要的 views/数据，不依赖 Simulation(避免循环依赖)
    // 收 Sources：ρ 谱 / 网格侧总电荷诊断需要 ρ(及 ρ_k)
    void maybe_compute(long step, ParticleViews p, const Sources& src,
                       const Fields& f, const RunParams& rp, cudaStream_t);
    // 能量(动能/场能)、总电荷、ρ 谱、E 场快照、相空间(x,vx)
private:
    // double 精度归约缓冲
};
```

**用 double 做能量/电荷归约**（plan §17）。本步必须能输出：场能 + 动能守恒曲线、总电荷守恒。

---

## 10. 编排类 `Simulation`（主循环）

```cpp
class Simulation {
    RunParams      p_;
    Grid           grid_;
    Particles      particles_;
    Sources        sources_;
    Fields         fields_;
    SpectralEngine spectral_;     // 拥有 FFT plan + k 空间 workspace(§7.1)
    ElectrostaticSpectralSolver solver_;   // v1 直接 concrete; Darwin 阶段再换 std::variant(§7.2)
    Diagnostics    diag_;
    CudaStream     stream_;
public:
    void init();
    void step(long n) {
        sources_.zero(stream_);
        Depositor<Cfg>::charge(particles_, sources_, grid_, p_, stream_);

        // solver 内部完整 rho -> E（含 FFT/Poisson/IFFT/normalize + 中性化）
        solver_.solve(sources_, fields_, spectral_, p_, stream_);

        // push 内部先 gather E(收 grid_, 与 deposit 共用同一套 CIC 权重) 再 Boris
        Pusher<Cfg>::boris(particles_, fields_, grid_, p_, stream_);
        particles_.migrate(grid_, stream_);

        // 解耦：只传它需要的 views(含 sources_ 用于 ρ 谱/电荷诊断)，不传 *this
        diag_.maybe_compute(n, particles_.views(), sources_, fields_, p_, stream_);
    }
    void run();   // 后续可对 step() 做 CUDA Graph capture(plan §12)
};
```

要点：

- **FFT 不再散落主循环**：`fft_` 那个不存在的成员已消除，整个 `rho→E` 收进 `solver_.solve(...)`，借用 `spectral_`。
- **v1 用 concrete `solver_`**（不是 `std::variant`）——只有一个 solver 时最干净；Darwin 出现再升 `std::visit`（§7.2）。
- **`Depositor<Cfg>` 写 `sources_`、`Pusher<Cfg>` 收 `grid_`**：deposit 写 ρ(Sources)，push 内部 gather 从 Fields 读、要 grid 做 idx/wrap，与 deposit **共用同一套 CIC 权重**（plan §16 第 4 条，经典 bug 源）。
- **`Diagnostics` 解耦**：传 `ParticleViews + Sources + Fields + RunParams`，不传 `const Simulation&`（否则 Simulation/Diagnostics 互相依赖）。ρ 谱诊断需要 `sources_`。

这就是 plan §14 的统一主循环，Darwin 只是把 `solver_` 升成 variant 并增加 `Depositor<Cfg>::current`。

---

## 11. 实施顺序（按依赖，逐步可验证）

| 阶段 | 内容 | 验证 |
|---|---|---|
| S1 | `DeviceArray`/`CudaStream`/`CUDA_CHECK`/`CufftPlan2D` | FFT 往返 `c2r(r2c(f))≈f` |
| S2 | `Grid`/`KGrid`/`RunParams`/`SimConfig` | 单元测试 kx/ky/k2、周期取负 |
| S3 | `Particles` 初始化 + Maxwellian 加载 | 速度分布直方图 = 解析 |
| S4 | `Depositor::charge` v0(global atomic) | 总电荷 = N·q；均匀分布→ρ≈常数 |
| S5 | `ElectrostaticSpectralSolver` | **单一正弦模式** ρ=sin(2πx/Lx) → 验 E_k 振幅/相位(不要用 Debye 屏蔽,那是动理学响应) |
| S6 | `Pusher::boris`(无 B0) | 单粒子在已知 E 下轨迹正确 |
| S7 | 接通 `Simulation::step` 全链路 | **Langmuir 振荡** ω≈ω_pe |
| S8 | 诊断：能量/电荷守恒 | 能量漂移有界 |
| S9 | 物理验证 | **two-stream 不稳定性**增长率对解析 |
| S10 | 加 B0(Boris 已支持) | E×B 漂移、upper-hybrid (Level 2) |

S7/S9 通过即代表第一步成功。之后才进入 Darwin（加 `J/dcu/amu` 沉积 + Darwin k-space 求解 + 电流谱修正，plan §13 Level 3）。

---

## 12. 为 Darwin 预留的扩展点（现在就留口，别返工）

1. `Sources` 加 `Jx/Jy/Jz, dcu, amu`；`Fields` 加 `Ez/Bx/By/Bz`；`SpectralWorkspace` 加对应 k 版本（先不分配）。
2. `AnySolver` variant 加一个 `DarwinSpectralSolver`，`solve(...)` 签名不变（借用同一 `SpectralEngine`）。
3. `Depositor<Cfg>` 加 `current()/dcu_amu()` 方法（同样编译期 policy）。
4. `Pusher<Cfg>` 留 `darwin()` 变体（含横向 E + B 的迭代修正）。
5. 连续性方程谱修正（plan §4.3）：`i k·Ĵ = -(ρ̂ⁿ⁺¹-ρ̂ⁿ)/Δt`，作为后续 cheap deposition + 纵向修正方案。

---

## 13. 第一步要做的关键决策（建议默认）

- **粒子数据结构**：第一版 SoA 扁平数组，**不上 chunk pool**（先把物理跑对，再按 plan §7 优化）。
- **沉积**：先 `AtomicGlobalDeposit` → 再 `SharedTileDeposit`，**通过 `Cfg::deposit` 编译期 policy 切换**（§3/§8），主循环不动。
- **多态边界**：device hot path 一律编译期 policy / `if constexpr`，**零 virtual**；host 编排层（solver 选择）才用 `std::variant` 或 virtual（§7）。
- **owning/non-owning 分离**：容器持有 `DeviceArray`，kernel 只收 `*Views`（POD `DeviceView`）（§2.1）。
- **速度/动量**：分量按动量 `u=γv` 存，v1 令 γ≡1（§3）；whistler/chorus 电子可相对论，避免 retrofit。
- **精度**：粒子/场 float，诊断 double。
- **边界**：周期（FFT 天然匹配）。
- **离子 / 中性背景**：固定均匀背景**不 deposit 离子**。周期 ES 的中性化 == solver 里**把 `ρ_k[k=0]` 置零**（Poisson 本来就要去 DC）。不需要单独减 mean、也不需要 ion 数组。（电子 q=-1 时若真要显式背景，是 `ρ_i=+n0`，但这里用置零 k=0 一步到位。）

---

## 14. 对 `gpu_darwin_pic_plan.md` 的 Review 意见与修订点

> 本节是对总纲 `gpu_darwin_pic_plan.md` 的批判性 review。架构层面基本全盘认同，可直接照其数据结构开工；但下面几条要么和「纯 GPU」约束冲突、要么是 plan 写得不够而 GPU-only 路线下最容易卡住的数值正确性细节。

### 14.0 认同的部分（plan 抓对的关键点）

- 第一天就 2D3V（`uz` 必存，plan §2.5）——避免后面为 Darwin/whistler 推翻代码。
- `Sources2D`/`Fields2D`/`SpectralArrays2D` 预留 `J/B/E_k`，solver 可替换（plan §5.8–5.10, §14）。
- 位置存 cell 坐标、normalized units（plan §4 Step 0）。
- deposit/gather 共用同一套 CIC stencil；diagnostics 第一版就写（plan §10）。
- 先 flat SoA 再 chunk pool（plan §13）。

### 14.1 ⚠️ 最大冲突：plan 是 CPU-first，但本项目要纯 GPU、无 CPU 版

plan 通篇假设**先写 CPU reference 再移植 CUDA**（§4 Step 1、§15 Milestone 1–3 全是 CPU、§5.3 先 `HostArray` 再 `DeviceArray`）。这与「purely GPU, no CPU version」直接冲突。

**采取的折中方案**：

- 生产代码 **GPU-only**（符合约束）；
- 但**保留一个极小的 CPU reference，仅用于单元测试/对拍验证**（Poisson 解析模式、CIC 电荷守恒、单粒子轨迹），不进主循环、不参与性能、几十行即可。

理由：GPU PIC 最难的不是写 kernel，而是 debug 数值正确性。没有可信参照，很难判断 Langmuir 频率偏差到底是物理噪声、normalization 错、还是 FFT 索引错。若坚持连这个 CPU 对拍都不要，则必须靠**解析解对拍 + Parseval/能量守恒**作为替代验证手段（可行但更费劲）。

> 决策待定：GPU-only + 极小 CPU 对拍（推荐） vs 纯 GPU 仅靠解析解验证。

### 14.2 normalization 没闭环到 ωpe = 1（隐藏地雷）

plan 给了 `weight = n0*cell_volume/ppc`（§5.4）和 `phi_k = rho_k/(eps0 k²)`（§8），但**没把粒子权重、密度、场归一化串起来保证 ωpe=1**。UPIC 的 `affp = nx*ny/np` 正是干这个的（plan §2.8 提了 ffc，但 v1 直接令 g(k)=1/(eps0 k²)）。

后果：Langmuir 测试频率会系统性偏离 1，却容易误判为别处 bug。

**修订**：在 Step 0（S2 之前）就用纸笔推清楚一个 ωpe=1 的归一化算例，写进 `RunParams` 注释，别等 S9 失败再回头查。

### 14.3 leapfrog 初始化要把速度回退半步

plan §4 Step 4 用 leapfrog（u 在半步、x 在整步），但初始化没说要把 `u` 用初始 E 做 −dt/2 的 kick 回退半步。不做的话能量守恒曲线第一步就有系统性 offset，长程会让你误判格式不守恒。

**修订**：`Particles::initialize` 末尾加一次 half-step 回退；plan §16「最容易犯错列表」应补这一条。

### 14.4 验证算例加 quiet start

plan §17 初速度用 Maxwellian 或 cold。但 Langmuir（Test 5）和 two-stream（Test 6）用随机加载时，PIC 噪声可能淹没相干模式，尤其 ppc=16 这种小值。

**修订**：v1 就支持 **quiet start**（均匀/分层加载 + 单一 k 正弦扰动），物理验证信噪比会显著改善。

### 14.5 第一个测试应是 FFT 往返 + Parseval（先于 grid indexing）

plan §11 把 grid indexing 排第一。GPU-only 路线下，应把 **`c2r(r2c(f))≈f` 往返 + Parseval 能量一致性**排最前——它一次性暴露 cuFFT 归一化（plan §16 第 1 条最常见 bug）、R2C 半谱 layout、索引约定三件事。这对应本文 §11 实施表的 **S1 验证项**，应作为最高优先级。

### 14.6 显式锁定 R2C 的 reduce 维与内存 layout

本文 §4 `Grid`/`KGrid` 用 `nkx = nx/2+1`，意味着 R2C 在 x 维折叠（与 plan §5.2 的 `complex_ny=ny/2+1`、在 y 维折叠相反——两者各自自洽，但**约定必须二选一并全代码统一**）。这是后面 Darwin 加 batched FFT、transverse projection 时最容易踩的坑。

**修订**：在代码里**注释死**「哪个维是连续维 / 被 R2C 折叠」，并保证 `physical_kx` 跑全程、被折叠维只跑半程的约定与之绑定一致。

### 14.7 一句话总结

架构照搬 plan 即可；但要补两类东西：**(a) 与「纯 GPU」约束的路线冲突先决断（§14.1）；(b) 几个数值正确性细节（ωpe 归一化、leapfrog 半步、quiet start、FFT 往返优先测、layout 锁定）——这些恰是 GPU-only 下最容易卡很久的地方。**
