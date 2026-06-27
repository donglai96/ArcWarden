# 第一步编码计划 & Checklist —— 2D3V 电静电谱方法 GPU PIC

> 实现蓝图见 `es_pic_step1_class_design.md`；review 见 `es_pic_step1_gpt_review_round1.md`。
> 约束：纯 GPU、单卡、C++20 + CUDA。命名空间 `arc`，头文件在 `include/pic/`。
> 规则：**一个 subtitle = 一步**。每步先打勾完成项，再跑「验证」才算过；验证不过不进下一步。
> 关键约定（落代码时逐字写进对应文件顶部注释）：
> - layout：`real [ny][nx]`，`idx = j*nx+i`，`R2C → [ny][nx/2+1]`，`cufftPlan2d(&p, ny, nx)`。
> - 归一化：`weight=n0·dx·dy/ppc`；`rho += q·weight·CIC/(dx·dy)`；`phi_k=rho_k/(eps0·k²)`（k=0 与 Nyquist 置零）；`E_k=-i·k·phi_k`；C2R 后 ×`1/(nx·ny)`。
> - 粒子分量是动量 `u=γv`，v1 令 γ≡1。
> - `Grid` 是几何唯一真源；`RunParams` 不存几何。

---

## Step 0: 脚手架（建文件夹 + 空文件）✅

- [x] 建目录 `include/pic/`、`src/`、`tests/`
- [x] 建空头文件：`config.hpp` `cuda_utils.hpp` `device_array.hpp` `grid.hpp` `sources.hpp` `fields.hpp` `fft.hpp` `spectral.hpp` `particles.hpp` `depositor.hpp` `pusher.hpp` `solver_es.hpp` `diagnostics.hpp` `simulation.hpp`
- [x] 建空源文件：`src/main.cpp` `src/spectral.cu` `src/deposit.cu` `src/push.cu` `src/solver_es.cu` `src/particles.cu` `src/diagnostics.cu` `src/simulation.cu`
- [x] 建空测试：`tests/test_fft_roundtrip.cu` `tests/test_poisson_sine.cu` `tests/test_charge_conservation.cu` `tests/test_single_particle.cu`
- [x] 建空 `CMakeLists.txt`

**验证**：目录树存在、文件均为空、无多余文件。✅

---

## Step 1: 构建系统（CMake 能编译空工程）✅

- [x] `CMakeLists.txt`：C++20、`LANGUAGES CXX CUDA`、`CMAKE_CUDA_ARCHITECTURES native`（RTX 5090 = sm_120）
- [x] 链接 `cudart`、`cufft`（`find_package(CUDAToolkit)` → `CUDA::cudart` / `CUDA::cufft`）
- [x] 能跑的 `main.cpp`（打印 device 名/算力/显存 + 退出 0）
- [x] CTest 占位 `smoke_device`（跑 main 二进制）
- [x] 处理系统默认 `nvcc`(12.0) 太旧：CMakeLists 在 `project()` 前自动选 `/usr/local/cuda-13.3/bin/nvcc`

**验证** ✅：`cmake -B build && cmake --build build` 成功；`./build/arcwarden` 打印
`NVIDIA GeForce RTX 5090 / sm_120 / 31.4 GB`；`ctest` 1/1 通过。
环境：CUDA 13.3、cmake 3.28.3、RTX 5090(sm_120, 31.4GB)。

---

## Step 2: CUDA 基础设施 `cuda_utils.hpp` ✅

- [x] `CUDA_CHECK` / `CUFFT_CHECK` 宏（报错带 file:line，抛 `arc::CudaError`）
- [x] `CudaStream`（RAII，move-only，禁拷贝；`get()`/隐式转换/`synchronize()`）
- [x] `CudaEvent`（RAII，move-only；`record()`/`synchronize()`/`elapsed_ms_since()`）

**验证** ✅：sm_120 编译通过；create/move/destroy + 计时 + 错误路径 (`cudaSetDevice(99999)`
被捕获并打印 `file:line`)；`compute-sanitizer --leak-check full` 报 **0 bytes leaked /
0 allocations**（仅有的 2 个 error 是故意触发的 invalid-device 探针，非泄漏）。
注：`cufft 13.x` 删除了 `CUFFT_INCOMPLETE_PARAMETER_LIST/PARSE_ERROR/LICENSE_ERROR`，
error-string 表已去掉这三项。

**附带修复**：`CMakeLists.txt` 的 `CMAKE_CUDA_ARCHITECTURES` 之前被缓存成 75 —— 因为
`if(NOT DEFINED ...)` 守卫放在 `project(... CUDA)` 之后，而 `project()` 启用 CUDA 语言时
已把它初始化成编译器默认值(75)，守卫永不触发。已将 `set(... native)` 移到 `project()`
之前，现在 native 正确解析为 `120-real`（RTX 5090, sm_120）。

---

## Step 3: 内存封装 `device_array.hpp` ✅

- [x] `DeviceView<T>`（POD，`__device__ operator[]`；附 `size()/empty()`，POD 可值传 kernel）
- [x] `DeviceArray<T>`：普通 `cudaMalloc/cudaFree`、move-only、禁拷贝、`zero(stream)`、`view()`/`view() const`、`data()/size()/bytes()/empty()`；默认 ctor + move-assign 支持容器 resize 模式
- [x] 顶部注释：长存数组用普通 malloc，async scratch 以后再说

**验证** ✅：sm_120 编译；`zero()` 后拷回全 0；kernel 经 `view()` 写入后拷回校验
`a[i]==2i`；move 后源 `empty()`/`data()==nullptr`（无 double-free）；默认 ctor +
`c = DeviceArray<float>(N)` move-assign OK；`compute-sanitizer --leak-check full`
报 **0 bytes leaked / 0 errors**。

---

## Step 4: 配置 `config.hpp` ✅

- [x] `ShapeOrder`、`NormMode`
- [x] 沉积 policy tag：`AtomicGlobalDeposit` / `SharedTileDeposit` / `CellOwnedDeposit`
- [x] `SimConfig<Dim,VelDim,Shape,Real,HasB0,DepositPolicy>`
- [x] `RunParams`（dt、物理量、归一化字段；**不含几何**）
- [x] 顶部注释：写死归一化 contract

**验证** ✅：`static_assert(Cfg::dim==2, Cfg::vdim==3)` 通过；另加
`vdim>=dim` 守卫、policy 切换（`SharedTileDeposit` 得到不同类型）、
`RunParams` 为 `trivially_copyable`（kernel 可按值收）。两路编译通过：
nvcc 13.3 `-std=c++20 -arch=sm_120`（含一个按值收 `RunParams` 的 kernel）与
host `g++ -std=c++20`（头文件不依赖 CUDA 也能编译）。

---

## Step 5: 网格 `grid.hpp` ✅

- [x] `Grid`（nx,ny,dx,dy,Lx,Ly,`__host__ __device__ idx`，几何唯一真源）
      —— 加 `real_size()`、`wrap()`、`idx_periodic()`，ctor 从 (nx,ny,Lx,Ly) 算 dx,dy
- [x] `KGrid`（nkx=nx/2+1，`kx/ky/k2`，注意 ky 在 j>ny/2 取负）
      —— 加 `kidx()`、`complex_size()`；方法标 `__host__ __device__` 便于 host 单测
- [x] 顶部注释：写死 R2C layout 约定（real [ny][nx] / cufftPlan2d(ny,nx) / R2C [ny][nkx]）

**验证** ✅：host 单测全过 —— `idx`/`idx_periodic`（i=nx→0、i=-1→nx-1、j 同理、in-range 不变）；
`kx`（reduced 轴全非负 i·dkx）；`ky`（DC=0、+1、Nyquist j=ny/2 取正、j>ny/2 取负）；
`k2(0,0)==0`（solver 依赖）；`Grid/KGrid` 均 `trivially_copyable`。
两路编译通过：`g++ -std=c++20` 与 nvcc 13.3 `-arch=sm_120 -x cu`。

---

## Step 6: FFT + 谱引擎（`fft.hpp` / `spectral.hpp`）+ 往返测试 [S1] ✅

- [x] `CufftPlan2D`（按 `(ny,nx)` 建 plan，r2c/c2r，不重建）—— RAII move-only，
      持 R2C+C2R 一对 plan，`created_` 显式有效位（0 非安全 handle 哨兵），best-effort dtor
- [x] `SpectralWorkspace`（rho_k/phi_k/Ex_k/Ey_k）—— 一次按 `complex_size()` 分配
- [x] `SpectralFormFactor`（v1 `s(k)=1, g(k)=1/(eps0 k²)`）—— `__host__ __device__`，
      `green(k2)` 在 k2==0 返回 0（DC 不除零）
- [x] `SpectralEngine`（持有 plan + workspace + KGrid + formfactor；r2c/c2r 封装 + 归一化）
      —— `c2r` 后用模板 scale kernel 乘 `1/(nx·ny)`
- [x] `tests/test_fft_roundtrip.cu`（已接 CTest：`fft_roundtrip`）

**验证** ✅：`c2r(r2c(f))≈f` —— max|·|=4.8e-7；**Parseval**（R2C 半谱按非 DC/非 Nyquist
x 模计两次重建全谱）相对误差 4.5e-8；CTest 2/2 通过；
`compute-sanitizer --leak-check full`：**0 bytes leaked / 0 errors**。

---

## Step 7: 源/场容器（`sources.hpp` / `fields.hpp`）✅

- [x] `SourceViews` / `Sources`（rho、`zero()`、`views()`）—— ctor(Grid) + `allocate()`
- [x] `FieldViews` / `Fields`（Ex,Ey、`views()`）—— ctor(Grid) + `allocate()` + `zero()`
- [x] 预留（先不分配）Darwin 字段注释（jx/jy/jz/dcu/amu；Ez/Bx/By/Bz）

**验证** ✅：nvcc 编译通过；`zero()` 后拷回全 0；kernel 经 `views()` 写入
`rho=3i`、`Ex=i`、`Ey=-2i` 拷回校验；`allocate()`（build-empty-then-size）路径 OK；
`SourceViews/FieldViews` 均 `trivially_copyable`。

---

## Step 8: 静电谱求解器 `solver_es` + 正弦模式测试 [S5] ✅

- [x] k-space Poisson kernel：`phi_k=rho_k/(eps0 k²)`，**k=0 与 Nyquist 置零**
      —— green(k2) 在 DC 返回 0；Nyquist（偶 nx→i=nx/2 或偶 ny→j=ny/2）力置零
- [x] `Ex_k=-i kx phi_k`，`Ey_k=-i ky phi_k`（复数 -i k(a+ib)=(k b,-k a)）
- [x] `ElectrostaticSpectralSolver::solve(sources, fields, spectral, rp, stream)`：完整 rho→E
      —— r2c → k-space kernel（模板化 ODR-safe）→ c2r×2（归一化）；借用 engine 不拥有
- [x] `tests/test_poisson_sine.cu`（已接 CTest：`poisson_sine`）

**验证** ✅：`rho=sin(2πx/Lx)`（Lx=2π→k1=1）→ `Ex` vs 解析 `-cos(x)`：
max|·|=9.2e-8、peak|Ex|=1.0000；`Ey`=0（恒等）；CTest 3/3 通过；
`compute-sanitizer --leak-check full`：**0 bytes leaked / 0 errors**。

---

## Step 9: 粒子 `particles.hpp` + 加载

- [ ] `ParticleViews` / `Particles`（SoA：x,y,ux,uy,uz,cell；`views()`）
- [ ] `initialize`：Maxwellian + **quiet start**（分层加载 + 单 k 扰动）
- [ ] leapfrog 半步回退（B0=0 静电 kick；B0≠0 backward half Boris 占位）
- [ ] `migrate`：周期 wrap + 重算 cell

**验证**：速度直方图≈解析；位置在 `[0,nx)×[0,ny)`；总动量/能量初值合理。

---

## Step 10: 电荷沉积 `depositor` + 电荷守恒测试 [S4]

- [ ] `deposit_charge_kernel<Cfg>`（写 `SourceViews`，CIC，global atomicAdd）
- [ ] `Depositor<Cfg>::charge(particles, sources, grid, rp, stream)`
- [ ] 归一化按 contract（`q·weight·CIC/(dx·dy)`）
- [ ] `tests/test_charge_conservation.cu`

**验证**：`sum(rho)·dx·dy == N·q·weight`（单/多/边界粒子）；均匀分布 ρ≈常数。

---

## Step 11: gather + Boris 推进 `pusher` + 单粒子测试 [S6]

- [ ] gather（收 `Grid`，与 deposit **同一套 CIC 权重**）
- [ ] `boris_push_kernel<Cfg>`：half-E → B 旋转(`if constexpr Cfg::has_b0`) → half-E → 位置更新
- [ ] `u=γv` 语义（v1 γ≡1）；`Pusher<Cfg>::boris(particles, fields, grid, rp, stream)`
- [ ] `tests/test_single_particle.cu`

**验证**：已知均匀 E 下单粒子轨迹对解析；B0≠0 时回旋频率/半径对解析。

---

## Step 12: 诊断 `diagnostics`

- [ ] `Diagnostics::maybe_compute(step, pviews, sources, fields, rp, stream)`
- [ ] double 归约：动能、场能（`0.5 eps0 ΣE² dx dy`）、总电荷、max|E|
- [ ] CSV 输出 `step,time,ke,ee,total,charge,max_e`

**验证**：静止均匀等离子体能量≈0 漂移；ρ 谱可输出。

---

## Step 13: 主循环 `simulation` 接通 [S7]

- [ ] `Simulation` 持有 grid/particles/sources/fields/spectral/solver(concrete)/diag/stream
- [ ] `step()`：zero→deposit→solve→push→migrate→diag（按设计 §10）
- [ ] `init()` / `run()`

**验证**：**冷 Langmuir 振荡** ω≈ω_pe；KE↔EE 交换；总能量有界。

---

## Step 14: 物理验证 [S8/S9]

- [ ] 能量/电荷守恒长程曲线（S8）
- [ ] **two-stream 不稳定性**：早期增长率对解析；相空间涡旋（S9）

**验证**：增长率与理论一致；相空间形成 vortex。

---

## Step 15: 磁化静电 [S10 / Level 2]

- [ ] 启用 `B0`，Boris 旋转路径（`Cfg::has_b0=true`）
- [ ] B0≠0 的初始化 backward half Boris 落实

**验证**：E×B 漂移；upper-hybrid / 回旋运动对解析。

---

## 进度

- 当前：**Step 9**（Step 0、1、2、3、4、5、6、7、8 已完成）
- 完成即在对应 subtitle 勾选，并在此更新「当前」指针。
