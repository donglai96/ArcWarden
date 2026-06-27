# ArcWarden — Step 0–3 总结与核心知识点（中文版）

> 项目：一个 2D3V 静电谱方法 GPU PIC code，结构上为后续 Darwin / 全 EM solver 预留口子。
> 纯 GPU、单卡、C++20 + CUDA，命名空间 `arc`，头文件在 `include/pic/`。
> 本文总结 Step 0–3 做了什么，以及可迁移的知识：GPU 编程、C++ 数据结构设计、构建系统、CUDA。

---

## 第一部分 — 每一步做了什么

### Step 0：脚手架
- 建目录树：`include/pic/`、`src/`、`tests/`。
- 建**空的**头文件 / 源文件 / 测试文件，以及空的 `CMakeLists.txt`。
- 目的：在写代码前先把文件布局钉死。
- **验证**：目录树存在、文件均为空、无多余文件。

### Step 1：构建系统（CMake）
- 写 `CMakeLists.txt`：C++20 + CUDA，选对 `nvcc`（cuda-13.3），目标 RTX 5090
  （`sm_120`），链接 `CUDA::cudart` / `CUDA::cufft`。
- 一个能跑的 `main.cpp`（打印 GPU 信息并退出 0）；一个 CTest 冒烟测试。
- **关键修复（Step 2 时发现）**：`CMAKE_CUDA_ARCHITECTURES` 被缓存成了 `75`。
  原因：`if(NOT DEFINED ...)` 守卫写在 `project(... CUDA)` **之后**，而 `project()`
  启用 CUDA 语言时已把该变量初始化成编译器默认值。把 `set(... native)` 移到
  `project()` **之前**即修复 → `sm_120`。
- **验证**：`cmake -B build && cmake --build build` 成功；二进制打印
  `RTX 5090 / sm_120 / 31.4 GB`；`ctest` 通过。

### Step 2：CUDA 基础设施 — `cuda_utils.hpp`
- **`CUDA_CHECK` / `CUFFT_CHECK`** 宏：包裹任意 CUDA/cuFFT 调用，失败时抛
  `arc::CudaError`，带 `file:line`、出错表达式和原因。
- **`CudaStream`**：`cudaStream_t` 的 RAII、move-only 封装（单一 stream 驱动整条
  流水线，便于以后 CUDA Graph capture）。
- **`CudaEvent`**：`cudaEvent_t` 的 RAII、move-only 封装（计时 + 同步）。
- **验证**：创建/移动/销毁 + 计时 + 错误路径；`compute-sanitizer --leak-check full`
  → 0 字节泄漏。

### Step 3：GPU 内存所有权 — `device_array.hpp`
- **`DeviceArray<T>`**：**拥有** GPU 内存。host 端、RAII、move-only，普通
  `cudaMalloc`/`cudaFree`。有 `zero(stream)`、`data()`、`size()`、`bytes()`、
  `view()`。**绝不**进 kernel。
- **`DeviceView<T>`**：**非拥有**的 POD 句柄（`ptr` + `n`），可平凡拷贝，**按值**
  传进 kernel，带 device-only 的 `operator[]`。
- 默认构造 + move 赋值支持「先建空、后分配」的容器模式：
  `rho = DeviceArray<float>(grid.real_size())`。
- **验证**：置零 / kernel 写入 / 拷回 host 全部正确；move 后源对象为空；
  `compute-sanitizer` → 0 泄漏 / 0 错误。

---

## 第二部分 — 核心知识：GPU 编程

1. **两个处理器、两套内存。** Host（CPU+内存）和 Device（GPU+显存）地址空间**独立**。
   `cudaMalloc` 返回的是 GPU 内存指针——CPU 不能解引用，只有 kernel 能用；
   数据通过 `cudaMemcpy` 跨越。

2. **Kernel 与启动配置。** `__global__` 函数是 kernel，用
   `kernel<<<blocks, threads, sharedBytes, stream>>>(args)` 启动。每个线程算出
   自己的全局下标 `int i = blockIdx.x * blockDim.x + threadIdx.x;`，必须越界检查
   `if (i < n)`。

3. **Kernel 参数按值传递（按位拷贝到 GPU）。** 所以传的是小的 POD 句柄
   （`DeviceView`），绝不传拥有型对象。这正是「拥有 / 非拥有」拆分的原因。

4. **函数空间限定符。** `__host__` = CPU 可调用，`__device__` = GPU 可调用，
   `__host__ __device__` = 两者皆可。我们把 `DeviceView::operator[]` 标成
   `__device__`，让「在 host 上索引 GPU 内存」直接编译报错。

5. **Stream = 有序的 GPU 工作队列。** 同一 stream 上的操作顺序执行，CPU 可以继续
   往下走（异步）。用单一流水线 stream 让顺序简单，并便于以后 CUDA Graph capture。

6. **Event = stream 中的标记点**，用于计时（`cudaEventElapsedTime`）和同步。
   读取耗时前必须先同步较晚的那个 event。

7. **异步 + 同步的纪律。** `cudaMemsetAsync`、kernel 启动、`cudaMemcpyAsync` 都是
   非阻塞的；当 CPU 需要结果时再 `cudaStreamSynchronize` / `cudaDeviceSynchronize`。

---

## 第三部分 — 核心知识：C++ 数据结构与设计

1. **RAII（资源获取即初始化）。** 把资源寿命绑到对象上：构造时获取、析构时释放。
   清理自动发生——即使异常展开栈也一样。我们对 stream、event、GPU 内存都用它。

2. **Move-only 类型。** 对唯一资源（一个 stream、一块分配），**删除拷贝**、**允许
   移动**。拷贝会产生两个所有者 → 双重释放。移动是**转移**所有权；
   `std::exchange(x, nullptr)` 返回旧值并把源置空，使源的析构什么都不做。
   与 `std::unique_ptr` 同款模型。

3. **拥有 vs 非拥有 拆分。** 把「管理寿命」的类型（`DeviceArray`，host 端、RAII）
   和「在热代码里被使用」的类型（`DeviceView`，POD、可拷贝）分开。房东 vs 钥匙。

4. **POD / 可平凡拷贝。** 公开数据、无析构的朴素 struct 才能安全按位拷贝——这是
   kernel 参数的硬性要求。

5. **析构里的 `noexcept` 清理。** 析构吞掉错误（best-effort 的
   `cudaFree`/`cudaStreamDestroy`）——在栈展开时抛异常可能触发 `std::terminate`。

6. **抛异常，而非 abort。** `CUDA_CHECK` 抛异常，这样退出时 RAII 析构会运行、
   干净释放 GPU 资源，而不是硬 abort 导致泄漏。

7. **编译期 vs 运行期多态（设计方向）。** 每粒子的热路径用编译期 policy
   （模板 + `if constexpr`）→ 零开销；「用哪个 solver」是便宜的 host 端选择
   （`std::variant`/`std::visit`），绝不在 device 上做虚调用。

---

## 第四部分 — 核心知识：构建系统（CMake / make）

1. **CMake 是构建系统*生成器*；两个阶段。**
   `cmake -B build`（configure：读 `CMakeLists.txt`、探测编译器/GPU、生成 `build/`），
   然后 `cmake --build build`（build：跑编译器）。

2. **先编译后链接。** 每个 `.cpp`/`.cu` 独立编译成 `.o` 目标文件；链接器把目标
   文件 + 库拼成一个可执行文件。增量构建只重编改动过的文件。

3. **`CMakeLists.txt` 自上而下执行；顺序重要。** 凡是影响*语言如何被启用*的设置
   （编译器选择、GPU 架构）必须放在 `project(... LANGUAGES ...)` **之前**；target
   的配置放后面。

4. **缓存会持久化决策。** `build/CMakeCache.txt` 记住 configure 的一切。陈旧的缓存
   值会覆盖你新改的默认值——用 `-DVAR=...` 或 `rm -rf build` 解决（这正是 `sm_75`
   bug 的成因与隐藏原因）。

5. **以 target 为中心的作用域命令。** 用 `target_include_directories`、
   `target_link_libraries` 等。`PRIVATE` = 仅本 target；`PUBLIC` = 本 target +
   依赖者；`INTERFACE` = 仅依赖者。

6. **导入目标（imported targets）** 如 `CUDA::cudart` / `CUDA::cufft` 自带 include
   路径和链接选项，所以 `find_package(CUDAToolkit)` + 链接该 target 就够了。

7. **clean / 重建命令。**
   - 改了源文件 → `cmake --build build`
   - 改了 `CMakeLists.txt` → `rm -rf build && cmake -B build && cmake --build build`
   - 轻量 clean → `cmake --build build --target clean`

---

## 第五部分 — 核心知识：CUDA 专项

1. **错误处理靠手动。** CUDA 返回状态码（`cudaError_t`、`cufftResult`），不抛异常。
   每个调用都要检查——所以有 `CUDA_CHECK`/`CUFFT_CHECK`。

2. **cuFFT 没有内置 error-string 函数**（CUDA runtime 有：
   `cudaGetErrorName`/`cudaGetErrorString`）。我们自己写了查表。注意 cuFFT 13.x
   删掉了一些旧枚举值（`CUFFT_PARSE_ERROR` 等）。

3. **GPU 架构标志。** `sm_120` 是 RTX 5090 的算力（12.0）。`native` 自动探测；
   `120-real` = 为 sm_120 生成真实 SASS。

4. **toolkit 版本要紧。** 系统的 `nvcc`（12.0）对 sm_120 太旧；我们显式选 cuda-13.3。

5. **`compute-sanitizer`** 是 CUDA 正确性检查器：
   `--tool memcheck --leak-check full` 抓泄漏和越界。我们把它作为每个内存/资源步骤
   的验证关卡。

6. **按字节置零。** `cudaMemset` 写字节；对 `float`/`int`/`cufftComplex`，全零位
   模式等于数值零，所以可作为合法的 `zero()`。

7. **R2C FFT 布局（已钉死的约定）。** real `[ny][nx]`，`idx = j*nx + i`；R2C 输出
   `[ny][nx/2+1]`；`cufftPlan2d(&p, ny, nx)`。全代码锁死——Darwin/EM 的 batched FFT
   都依赖它。

---

## 第六部分 — 其它重要实践

1. **一个 subtitle = 一步；验证通过再前进。** 每步只有在验证（编译 + 运行 +
   sanitizer）通过后才算「完成」，不跳步。

2. **验证关卡 = `compute-sanitizer` 干净。** 对基础设施步骤，「能编译」不够；
   「0 字节泄漏 / 0 错误」才是标准。

3. **为未来设计，为当下分配。** 容器为后续物理*预留名字*（`J`、`B`、`E_k`、`B_k`），
   但 v1 不分配它们。这样从 ES → Darwin → 全 EM，API 保持稳定、不需重写。

4. **几何只放一处。** `Grid` 是 `nx, ny, dx, dy, Lx, Ly, idx` 的唯一真源；
   `RunParams` 只放物理量。避免同一个数在两处「漂移」。

5. **归一化 contract 先定。** `eps0=1, me=1, |e|=1, ωpe=1`；位置用 cell 单位；
   写代码前钉死，避免中途单位变动。

6. **每步配一份学习笔记。** `learning_step{1,2,3}_*.md` 讲清「为什么」，让知识能
   迁移到这个代码库之外。

---

## 进度
Step 0、1、2、3 完成并验证。下一步：**Step 4 — `config.hpp`**
（`SimConfig` 编译期 policy、`RunParams`、沉积 policy tag）。
