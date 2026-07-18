# ArcWarden-LShell 完整项目文档

**A GPU-First Fixed-L-Shell Electromagnetic Kinetic PIC Framework for Azimuthal Particle Injection and Whistler-Wave Chirping**

**Master Document v3** — 科研计划 · 模型方程 · 数值架构 · 里程碑 · 代码改造计划的统一完整版

> 文档沿革：v1 为原始设计愿景；第一轮评审补充了尺度、显存与运行时间的定量分析；第二轮评审校准了表示方法、权重精度与边界方案的过早结论；v2 roadmap 合并了三轮结论；代码计划 v2 基于实际仓库 donglai96/ArcWarden 重写。本 v3 将以上全部内容合并为单一完整文档，取代所有前版。
>
> 配套仓库：https://github.com/donglai96/ArcWarden （C++20 + CUDA，单 GPU，RTX 5090 / sm_120）

---

# 目录

**Part I — 科学背景与项目愿景**
1. 项目愿景
2. 科学动机
3. 模型定位与适用范围声明

**Part II — 目标物理模型**
4. 坐标系与几何
5. 背景场、扰动场与 Maxwell 求解
6. 电子模型：表示方法、δf 形式与沉积语义
7. Prescribed Azimuthal Drift
8. 边界条件与粒子注入
9. 尺度与相似性设计

**Part III — 数值架构与资源预算**
10. 现有代码基础（实际仓库盘点）
11. 三个关键架构决定
12. 时间步、融合 kernel、沉积、迁移与精度
13. 资源预算：网格、粒子数、显存与运行时间

**Part IV — 里程碑与代码改造计划（M0–M12）**

**Part V — 验证策略**

**Part VI — 风险登记**

**Part VII — 软件架构、工作量、执行纪律与成功定义**

---

# Part I — 科学背景与项目愿景

## 1. 项目愿景

ArcWarden 当前是一个面向单 GPU 的高性能 2D3V 谱方法 Particle-in-Cell code，支持静电谱求解与谱 Darwin（磁感应、无辐射）电磁模型。本项目的长期目标是将其发展为一个能够模拟固定 L-shell 曲面上波粒相互作用的大型 GPU kinetic code：

$$\boxed{\text{固定 }L\text{-shell 曲面上的 2D3V full-Maxwell electromagnetic kinetic PIC}}$$

固定：

$$L=L_0$$

解析两个空间方向：

$$(q^1,q^2)=(s,\phi)$$

其中：
- $s$：沿背景磁力线方向；
- $\phi$：沿固定 L-shell 的方位角，即 energetic electron 的 gradient-curvature drift 方向；
- $L$：法向方向，不解析。

粒子速度和电磁场保留全部三个分量。

最终研究目标包括：
- energetic electron population 沿方位方向漂移和注入；
- whistler-mode wave 的线性增长；
- nonlinear cyclotron resonance 与 phase trapping；
- rising-tone 和 falling-tone frequency chirping；
- chorus source region 在 magnetic local time（MLT）方向的空间结构；
- 注入电子结构经过固定 L-shell 区域时，chorus 的产生、传播和调制。

**核心科学问题（经尺度分析修订后的最终表述）**：

> 在一个局部 MLT 方位窗口内，当边界上出现时变的、能量-投掷角相关的高能电子漂移注入时，whistler/chorus 如何产生、chirp、传播和关闭？源区的方位宽度、element 间隔与 repetition period 由什么控制？

注意此表述与初版"模拟一整个电子云以真实速度穿过一小时 MLT"的区别：定量分析（§9）表明真实漂移时间尺度与可负担模拟时长相差 2–3 个数量级，因此大尺度方位输运通过**边界条件 + 相似性缩放参数**进入模型，而 $\phi$ 依然是被解析的真实空间维度。

## 2. 科学动机

### 2.1 一维 field-aligned model

现有 chorus PIC 模型的第一类几何只解析单条磁力线上的 $s$（如 Omura/Katoh 系 electron-hybrid code）。这种模型可以模拟：
- 非均匀背景磁场 $B_0(s)$；
- cyclotron resonance；
- nonlinear trapping；
- rising-tone chirping；
- 波包沿磁力线传播。

但无法解析：
- energetic electron cloud 的方位漂移；
- chorus source region 的有限 MLT 宽度；
- 不同方位位置之间的波动耦合；
- drifting injection structure 对 chorus onset 和 repetition 的调制。

### 2.2 二维 magnetic-meridian gcPIC

第二类几何解析 $(s,L)$，即沿磁力线方向与跨 L-shell 的径向方向。适合研究：
- chorus 的跨场结构；
- 多个 L-shell 之间的传播；
- wave-normal evolution；
- dipole meridional geometry。

但其第二维不是 energetic electron 的主要 gradient-curvature drift 方向。

### 2.3 ArcWarden-LShell 填补的空白

ArcWarden-LShell 解析 $(s,\phi)$，把以下两个过程放进同一个 kinetic simulation：

$$\boxed{\text{field-aligned chorus dynamics}+\text{azimuthal energetic-electron transport}}$$

| 几何 | 解析维度 | 能做 | 不能做 |
|---|---|---|---|
| 1D field-aligned | $s$ | 非均匀 $B_0$、trapping、chirping | 方位漂移、有限 MLT 源区、方位耦合 |
| 2D meridian gcPIC | $(s,L)$ | 跨壳结构、径向传播 | 第二维非漂移主方向 |
| **ArcWarden-LShell** | $(s,\phi)$ | chirping 动力学 + 方位注入/输运/耦合 | 跨 L 传播（见 §3） |

## 3. 模型定位与适用范围声明

**本节内容必须写入所有基于本代码的发表物。**

模型忽略 $\partial/\partial L$，因此不能自洽表现：
- 跨 L 折射与 radial group velocity；
- 波能量离开当前壳层；
- 多壳层耦合；
- 与径向密度梯度有关的 mode conversion。

真实 chorus 离开赤道后 wave normal 变斜、发生跨 L 折射，部分能量离开源区所在壳面；2D $(s,\phi)$ 模型强制波留在面内，可能高估方位耦合与波幅。因此：

> ArcWarden-LShell 是一个**固定壳面的局部约化模型**，目标是解析 field-aligned 与 azimuthal 动力学，而非完整三维波传播。

其结果最适用于：
- 赤道附近 source region；
- 有限时间内的 chirping 形成；
- 方位注入结构；
- 跨壳传播尚未主导之前的波粒相互作用阶段。

定义**适用性参数**：

$$\epsilon_L=\frac{v_{g,L}\,T_{\rm sim}}{\Delta L_{\rm physical}}$$

其中 $v_{g,L}$ 由理论色散或外部 ray tracing 估算（代码本身不解析）。当 $\epsilon_L\gtrsim1$ 时，固定壳面结果必须谨慎解释。$\epsilon_L$ 是每次生产运行的强制记录字段，诊断输出中自动标注。

---

# Part II — 目标物理模型

## 4. 坐标系与几何

### 4.1 模拟域

模拟域位于固定的 L-shell 曲面：

$$\mathbf r=\mathbf r(s,\phi;L_0)$$

解析坐标为：

$$s\in[s_{\min},s_{\max}],\qquad \phi\in[\phi_{\min},\phi_{\max}]$$

其中：
- $s=0$ 对应 magnetic equator；
- $s_{\min}$ 和 $s_{\max}$ 位于南北高纬区域；
- $\phi$ 表示局部 MLT 窗口（数百 $c/\omega_{pe}$ 量级，见 §13），或周期性方位方向。

### 4.2 三级几何近似

**第一级（Cartesian 展开，M1–M10 主线）**：

$$x=s,\qquad y=R_{\rm eq}\phi$$

**第二级（weak metric，M11 前半，必做）**：引入方位尺度因子

$$h_\phi(s)\approx r(s)\sin\theta(s)$$

（同样的 $\Delta\phi$ 在不同磁纬对应不同物理长度。）weak metric 只进入三处：漂移坐标换算、cell 面积/体积权重、能量诊断 Jacobian。若 $s$ 使用真实弧长，则 $h_s=1$。

**第三级（full metric-aware 曲面 Maxwell，M11 后半，条件触发）**：完整 curvilinear Yee + Jacobian-aware 沉积。仅当 weak-metric 对照显示对 M10 结果有显著（>10%）影响时才启动。在 $|\lambda_m|<30°$ 内 $h_\phi(s)$ 的变化只有几十个百分点，对 chirping 一阶物理预期 weak metric 已足够。

## 5. 背景场、扰动场与 Maxwell 求解

### 5.1 场分解

总磁场拆分为：

$$\mathbf B=\mathbf B_0(s;L_0)+\delta\mathbf B(s,\phi,t)$$

电场为纯扰动场：

$$\mathbf E=\delta\mathbf E$$

其中：
- $\mathbf B_0$ 来自解析 dipole model（第一阶段轴对称：$\partial_\phi\mathbf B_0=0$，$B_0=B_0(s;L_0)$）；
- $\mathbf B_0$ 不由 Maxwell solver 动态更新；
- solver 只推进扰动场：

$$\frac{\partial\delta\mathbf B}{\partial t}=-\nabla\times\delta\mathbf E$$

$$\frac{\partial\delta\mathbf E}{\partial t}=c^2\nabla\times\delta\mathbf B-\frac{\delta\mathbf J}{\epsilon_0}$$

**关键约定**：平衡电流（含背景漂移环电流）视为**外部支撑**，绝不进入扰动场方程。这一约定对不同粒子表示的沉积语义有直接后果（§6.4），是最容易出 bug 的地方之一。

### 5.2 为什么从 Darwin 转向 full-Maxwell

现有谱 Darwin solver（UPIC `mpdbeps2` 血统，丢弃辐射项 $\partial\mathbf E_T/\partial t$，无光波 CFL）适合周期、低频、均匀或弱非均匀问题，且已通过 Weibel 增长率与磁静态 gate 验证。但最终目标需要：
- outgoing wave packet 与 field-aligned absorbing boundary；
- finite MLT sector（非周期几何）；
- particle injection and outflow；
- potentially oblique waves。

因此长期目标转向 full-Maxwell（Cartesian Yee，后期可选 metric-aware curvilinear）。谱 Darwin 与静电求解器**永久保留编译与测试**，作为回归基线与交叉验证参照。

### 5.3 场求解器技术要求

- Cartesian Yee（或等价 FDTD）staggered E/B 推进，2D3V（$\partial_z=0$ 但保留全部 6 分量）；
- Esirkepov / EZ 类 charge-conserving current deposition；
- Gauss law 保持策略：守恒沉积为主，必要时 Marder/hyperbolic cleaning 兜底；
- $\nabla\cdot\mathbf B$ 逐 cell 监测。

### 5.4 吸收边界（重要修订：不预设 PML）

标准 PML 面向真空/简单介质；whistler 波进入含等离子体的 PML 常出现不稳定或强反射。因此 $s$ 端边界不预设 PML，改为"经过验证的吸收边界系统"，候选方案在 M2 中系统比较：
- field damping（masking / friction layer）；
- current damping；
- particle/moment damping region；
- 真空缓冲区之后的标准 PML；
- 混合方案。

第一版最可能的分层结构：

```
physical plasma region
      ↓
smooth particle/moment damping region
      ↓
low-density or vacuum buffer    ← 注意：密度渐变本身改变 whistler 色散，可能产生反射
      ↓
Maxwell PML
```

**边界 benchmark 必须在等离子体条件下进行**：测量

$$R(\omega,\theta)=\frac{P_{\rm reflected}}{P_{\rm incident}}$$

不能只测真空电磁波反射。密度渐变段需要独立的反射专项测试。

### 5.5 离子

第一阶段不推进离子粒子；离子作为静态中和背景，用于定义 equilibrium charge neutrality。

**明确记录的近似时效**：$\Omega_i^{-1}\approx1836\,\Omega_e^{-1}$，而生产运行 $>10^4\,\Omega_e^{-1}$，已跨越多个离子回旋周期。离子极化电流与低混杂响应被完全丢弃，对 whistler 频段一阶物理影响不大，但必须在能量核算文档中注明。若后期发现 EMIC/低混杂耦合重要，作为独立扩展立项。

## 6. 电子模型：表示方法、δf 形式与沉积语义

### 6.1 设计原则（三轮评审的核心共识）

**不预先锁死任何 population 的表示方法。** 软件目标是让每个 population 独立选择：

$$\boxed{\text{fluid}\quad/\quad\delta f\quad/\quad\text{full-}f}$$

最终生产配置由**同一个 1D chirping 基准（M4）和 2D injection 基准（M10 前期）**的数据裁决，而不是由设计阶段的直觉决定。

```cpp
enum class Representation { Fluid /*条件触发*/, DeltaF, FullF };
```

**初始默认与替代方案**：

| Population | 初始默认 | 必须保留的替代 | 物理理由 |
|---|---|---|---|
| core（冷，密度最高） | δf（wd≈0 常驻） | finite-T fluid（条件触发，见 6.6） | 只提供 whistler 色散的介质响应，几乎不进入共振；δf 下噪声 ∝ δf 而非 f，少量 marker 即可 |
| warm（keV 级） | δf | full-f 交叉验证 | 提供 Landau/cyclotron 阻尼，扰动幅度中等，**δf 优势最大的地方** |
| hot（各向异性驱动源） | **δf 与 full-f 都实现、都测试** | — | chirping 本质是 phase-space hole/hill，非线性阶段局部 $\vert\delta f\vert\sim f_0$，δf 权重方差爆炸后降噪优势消失；但失效速度取决于大权重 marker 占比、hole 相空间体积、remapping 策略等，须由基准数据裁决 |

**裁决指标**（M4/M8 使用）：

$$\sigma_w=\sqrt{\langle w^2\rangle},\qquad \frac{N(|w|>0.5)}{N},\qquad \max|w_p|$$

$$\mathrm{SNR}_{J_{\rm res}}=\frac{|J_{\rm coherent}|}{\sigma(J_{\rm noise})}\quad\text{（须区分噪声来源：δf 权重方差 vs full-f 平衡采样噪声）}$$

以及 chirping rate 与波幅对 marker 数的收敛性、显存与端到端运行时间。$\sigma_w>0.3$ 作为**经验预警值**（触发 remapping/复查），不作为硬性算法切换标准。

### 6.2 δf 双权重表示

$$f_e=f_0+\delta f,\qquad w_p=\frac{\delta f}{g},\qquad p_p=\frac{f_0}{g},\qquad f=(p+w)\,g$$

参考分布可包含多个成分：

$$f_0=f_{0,\rm core}+f_{0,\rm warm}+f_{0,\rm hot}$$

每个 population 拥有独立的 density、temperature、anisotropy、marker distribution $g_a$、marker count、importance-sampling strategy 与 injection boundary rule。总扰动电荷与电流：

$$\delta\rho=\sum_a\delta\rho_a=\sum_a q_a\sum_{p\in a}w_pS_p,\qquad \delta\mathbf J=\sum_a\delta\mathbf J_a=\sum_a q_a\sum_{p\in a}w_p\mathbf v_pS_p$$

若某 population 满足 $g=f_0$，则 $p\equiv1$，通过 policy 模板避免实际存储 p 数组。

**权重更新优化**：使用 $\nabla\ln f_0$ 的预计算系数形式而非每步重新计算完整指数分布。例如 bi-Maxwellian：

$$\frac{\partial\ln f_0}{\partial v_\parallel}=-\frac{2v_\parallel}{v_{t\parallel}^2},\qquad \nabla_{v_\perp}\ln f_0=-\frac{2\mathbf v_\perp}{v_{t\perp}^2}$$

运行时只需乘加，不需要每粒子调用多次 expf()。

### 6.3 平衡分布 f0(ℰ,μ)

参考分布必须尽量满足：

$$\frac{D_0f_0}{Dt}=0$$

在轴对称、静态背景下构造：

$$f_0=f_0(\mathcal E,\mu),\qquad \mathcal E=\frac12m_ev^2,\qquad \mu=\frac{m_ev_\perp^2}{2B_0}$$

在磁赤道定义 core、warm 和 hot distributions，再通过能量与磁矩守恒映射到其他 $s$。

**禁止**在每个 $s$ 位置直接装载同样的局部 bi-Maxwellian——背景将不是 Vlasov equilibrium，模拟开始时会发生非物理松弛。

**注入情形的说明**：一旦 $\phi$ 方向引入非周期注入云（$\partial f/\partial\phi\neq0$），"平衡"只是分离出的参考态，$w$ 随注入持续增长属预期物理行为，权重诊断必须与数值权重发散（Risk R1）区分开。

### 6.4 三种沉积语义（关键技术约定，防 bug 重点）

由于场方程只推进扰动场（§5.1），三种表示的电流沉积**语义不同**，必须作为三个独立 deposition policy 实现。**不允许**通过"设 w=1"之类技巧强行统一——在 two-weight 形式 $f=(p+w)g$ 下代入 $p=1,w=1$ 得 $f=2g$，自相矛盾；full-f 是独立 policy 而非 δf 的特例。

1. **DeltaF**：沉积

$$\delta\mathbf J_a=q_a\sum_p w_p\,\mathbf v_p\,S_p$$

平衡电流天然不出现。

2. **FullF**（准确名称：total-f markers + 扰动电流沉积）：markers 携带总分布，沉积总电流**减去解析平衡电流**：

$$\delta\mathbf J_a=q_a\sum_p W_p\,\mathbf v_p\,S_p-\mathbf J_{0,a}^{\rm analytic}(s)$$

否则平衡（含漂移环）电流会污染扰动场方程。两个重要推论：
- 减去解析 $\mathbf J_0$ **不能消除平衡部分的采样噪声**——那正是 δf 省掉的东西；
- full-f hot 之所以可接受，仅因 $n_{\rm hot}/n_e\lesssim$ 几个百分点，绝对噪声本来就由波驱动主导（Omura 系代码 20 年实践）。这一噪声来源区分必须写进 M8 评估指标。

3. **Fluid**（条件触发）：流体动量方程给出 $\delta\mathbf J_{\rm fluid}$，直接加入总扰动电流。

总扰动电流跨表示求和：$\delta\mathbf J=\sum_a\delta\mathbf J_a$。

### 6.5 δf 电流的 cancellation 问题

δf 电流含正负权重，比 full-f 更容易发生 cancellation。沉积管线要求：

```
cell-ordered particles
    → warp-level grouped reduction (__match_any_sync)
    → shared-memory tile accumulation
    → block reduction
    → few global writes/atomics
```

hot population 进入 $|w|\sim1$ 阶段后，tile 内部分和用 FP32-Kahan 或 FP64 shared-memory 累加（成本很低）。以下诊断量始终使用 FP64 归约：

$$\sum_pw_p,\qquad \sum_pw_p^2,\qquad \max|w_p|,\qquad \text{total field and particle energy}$$

### 6.6 Fluid core 的触发条件（范围控制）

finite-T fluid core 是一个全新模块（流体方程推进、Maxwell 耦合、有限温度闭合选型），保守估计 2–3 个月额外工作量。领域标准（Omura/Katoh electron-hybrid、Tao DAWN 等）用 cold fluid + kinetic hot 成功复现 rising tone，因此该选项必须保留；但为控制范围：

- 第一轮 representation study（M8）**只比较**：① 全 δf；② core/warm δf + hot full-f；
- fluid core **仅当**以下任一条件成立时才立项实现：
  - core δf markers 的显存占用被证明是 marker 预算瓶颈；
  - core 采样噪声被证明污染 whistler 色散或 chirping 诊断。

## 7. Prescribed Azimuthal Drift

### 7.1 为什么漂移必须外部给定

真实磁层电子沿 $\phi$ 方向的漂移主要来自 gradient-B drift、curvature drift（必要时加 $E\times B$）。但固定 L-shell 曲面模型不解析法向 $L$ 坐标，无法自洽产生 gradient-B drift 所需的径向磁场梯度。因此方位漂移必须作为 **reduced closure** 显式加入——不是任意的常数平移，而是从完整三维背景磁场计算并投影到 $\phi$ 方向。

### 7.2 漂移模型

背景 guiding-center drifts：

$$\mathbf v_{\nabla B}=\frac{m_ev_\perp^2}{2q_eB^3}\,\mathbf B\times\nabla B,\qquad \mathbf v_c=\frac{m_ev_\parallel^2}{q_eB}\,\mathbf b\times\boldsymbol\kappa,\qquad \boldsymbol\kappa=(\mathbf b\cdot\nabla)\mathbf b$$

需要的方位速度：

$$v_{d,\phi}=\left(\mathbf v_{\nabla B}+\mathbf v_c\right)\cdot\mathbf e_\phi$$

数值上从三维 dipole 预计算系数 $C_\perp(s;L_0)$、$C_\parallel(s;L_0)$，并引入**显式加速因子** $S_d$（尺度化设计的一部分，见 §9）：

$$v_{d,\phi}=S_d\left[C_\perp(s)\,v_\perp^2+C_\parallel(s)\,v_\parallel^2\right]$$

粒子方位坐标更新：

$$\phi^{n+1}=\phi^n+\frac{v_{d,\phi}}{h_\phi(s)}\,\Delta t$$

漂移自然依赖 energy、pitch angle、磁纬、电荷符号，以及波粒相互作用后的速度变化。**所有输出结果必须记录 $S_d$。**

### 7.3 与完整回旋动力学的耦合

Whistler/chorus 的 cyclotron resonance 依赖 gyrophase，hot electrons 不能简化为普通 guiding-center particles。粒子模型采用：

$$\boxed{\text{full gyrophase Lorentz dynamics}+\text{prescribed slow guiding-center drift}}$$

流程：
1. Boris（或等价 full-orbit pusher）推进 velocity；
2. 计算 $v_\parallel$ 与 $v_\perp$；
3. 由几何表查 $v_{d,\phi}$；
4. slow drift 加入 $\phi$ 方向位置推进；
5. **漂移位移进入 charge-conserving 沉积轨迹**（否则破坏连续性方程）。

**Double-counting 检查**：需要无波单粒子测试确认面内 pusher 已产生的解析漂移，并避免重复计入：

$$v_{d,\phi}^{\rm add}=v_{d,\phi}^{\rm 3D}-v_{d,\phi}^{\rm resolved}$$

（在 $B_0=B_0(s)\hat{\mathbf s}$ 且坐标无曲率的展开面内，$\nabla B\parallel\mathbf B$ 使 $\mathbf B\times\nabla B=0$，解析漂移预期 ≈0，但必须实测确认。）

**位移限制**：Esirkepov 类沉积假设单步位移 <1 cell；CFL 检查必须显式包含 $|\Delta\phi\cdot h_\phi|$ 项——$S_d$ 放大后尤其要查。

### 7.4 gyro 半径的面外投影

2D3V 标准处理：速度三分量、位置两分量，回旋轨道在法向（$L$）方向的偏移被投影忽略。这是所有 2D3V PIC 的共有近似，无特殊处理需求，但写入模型文档。

## 8. 边界条件与粒子注入

### 8.1 s 方向（field-aligned）

**场**：M2 选定的吸收边界系统（§5.4）。

**粒子**：
- absorbing precipitation boundary；
- loss-cone flux diagnostic；
- 必要时 reservoir injection；
- 不采用简单 periodic wrap。

### 8.2 φ 方向

**Option A — periodic MLT domain**：用于 homogeneous drift tests、repeated injection structures、full azimuthal ring approximation。

**Option B — finite MLT window（最终注入问题）**：
- upstream boundary：particle reservoir / 时变注入；
- downstream boundary：absorbing outflow；
- fields：吸收边界。

注入方向由 electron drift direction 决定。

### 8.3 通量一致的 marker 注入

边界注入必须按照**进入域内的相空间通量**采样：

$$g_{\rm flux}\propto\dot\phi\,g,\qquad \dot\phi=\frac{v_{d,\phi}}{h_\phi}$$

而不是简单从体积分布 $g$ 随机抽样——$\dot\phi$ 依赖 energy 和 pitch angle，漂移较快的粒子穿过边界的概率更高。

- equilibrium reservoir：$w_{\rm inject}=0$，$p_{\rm inject}=f_0/g_{\rm inject}$；
- 额外 anisotropic injection：$w_{\rm inject}\neq0$，或定义随漂移移动的参考 population；
- full-f population 按其总分布通量采样。

**注入与吸收边界隔离**：物理 reservoir 与场吸收区之间必须有 buffer zone；禁止在吸收层 cell 内直接注入；反射与注入分别独立 benchmark。

## 9. 尺度与相似性设计（science-validity gate）

### 9.1 定量问题陈述

模型包含四个差异巨大的时间尺度：

$$\tau_{\rm gyro}\ll\tau_{\rm trap}\sim\tau_{\rm growth}\ll\tau_{\rm drift}$$

以典型 chorus 参数定量（L≈4）：

| 量 | 值 |
|---|---|
| $B_{\rm eq}=0.311\,\mathrm{G}/L^3$ | ≈ 490 nT |
| $\Omega_e$ | ≈ 8.6×10⁴ rad/s |
| $n_e$（plasmapause 外） | ~1–10 cm⁻³ |
| $\omega_{pe}/\Omega_e$ | ≈ 2–4 |
| $c/\omega_{pe}$ | ≈ 1.7 km（n=10 cm⁻³） |
| whistler（0.2–0.4 $\Omega_e$）波长 | ≈ 8–15 $c/\omega_{pe}$ |
| 10–100 keV 电子漂移速度 $v_d$ | ≈ 5–15 km/s |
| 穿越 1 小时 MLT（L=4 弧长 ≈ 6700 km） | **~10 分钟 ≈ 5×10⁷ $\Omega_e^{-1}$** |
| 可负担模拟时长（§13） | **~2×10⁴–10⁵ $\Omega_e^{-1}$**（0.25–1 s 物理时间） |

相差 **2–3 个数量级**：在真实参数下，电子云在一次模拟内漂移位移仅 ~5–10 $c/\omega_{pe}$，基本不动。直接解析电子云跨越可观 MLT 区域的 full-Maxwell PIC 不可行。这是初版设计最大的遗漏，也是整个项目科学有效性的前提问题。

### 9.2 无量纲组

生产计算真正需要保持的是以下比值，而非同时保持真实 $L_\phi$、真实 $v_d$ 和真实时间：

$$\Pi_d=\frac{\tau_{\rm drift,domain}}{\tau_{\rm growth}},\qquad \Pi_t=\frac{\tau_{\rm drift,domain}}{\tau_{\rm trap}},\qquad \frac{L_\phi}{\lambda_{\rm whistler}},\qquad \frac{v_d}{v_g}$$

### 9.3 缩放策略

漂移加速因子作为显式运行参数：

$$v_d^{\rm sim}=S_d\,v_d^{\rm physical}$$

| 策略 | 适合回答的科学问题 |
|---|---|
| 真实漂移（$S_d=1$，小域） | 漂移对单个 element 的微扰级影响 |
| 加速漂移（$S_d\sim10^2$–$10^3$，保持 $\Pi_d,\Pi_t$） | 漂移穿越与增长/trapping 的竞争 |
| 缩小 domain | 源区宽度效应 |
| 时变边界注入 | 注入时序对 onset/repetition 的调制 |

### 9.4 产出与纪律

独立文档 `docs/SCALING_AND_SIMILARITY.md`（M5.5 deliverable）。M6 之后的任何生产运行，其参数必须能映射回该文档定义的无量纲空间；$S_d$ 与 $\epsilon_L$ 为强制元数据字段；$S_d$ 扫描是 M10 生产计划的固定组成部分，用于外推到真实磁层参数。

---

# Part III — 数值架构与资源预算

## 10. 现有代码基础（实际仓库盘点）

### 10.1 仓库概况

donglai96/ArcWarden（public，main，24 commits）：2D3V 谱方法 GPU PIC，C++20 + CUDA（CUDA 13.x，RTX 5090 / sm_120），namespace `arc`，header-only 库在 `include/pic/`。两个场模型共享一套粒子引擎与主循环，编译期选择：
- **Electrostatic**：谱 Poisson $E_L=-ik\rho_k/(\epsilon_0k^2)$；已验证冷 Langmuir 振荡、two-stream、bump-on-tail、长时能量/电荷守恒；
- **Spectral Darwin**：$B_k=i\mu_0(k\times J)/k^2$，横向 $E_T$ 由加速度密度（dcu）与动量流（amu）矩经 resummed transverse Green's function $1/(\epsilon_0c^2k^2+n_0)$ 求得；已验证磁静态 current sheet 与 **Weibel 不稳定性增长率**；并复现了 An et al. (2019, PRL) Simulation 1（外部 whistler pump Landau-trap 尾部电子激发 Langmuir 波）。

归一化：$\omega_{pe}=1$，$m_e=1$，$|e|=1$，$\epsilon_0=1$。

### 10.2 关键现有资产（决定改造起点）

| 资产 | 位置 | 对本项目的意义 |
|---|---|---|
| CMake ≥3.18 + **12 项 ctest**（FFT round-trip、谱 Poisson、装载、电荷守恒、单粒子 Boris、诊断、冷 Langmuir、two-stream 增长率、长时守恒、Darwin 双 gate） | `CMakeLists.txt`, `tests/` | 回归框架现成；改造期间 12 项永远保持全绿 |
| deck 输入系统（INI：`[grid]/[time]/[plasma]/[species]`，不重编译换 setup） | `include/pic/deck.hpp`, `decks/*.ini` | 配置系统现成，只需扩 schema |
| `SimConfig<Dim,VelDim,Shape,Real,HasB0,DepositPolicy>` 编译期 policy | `config.hpp` | 新 FieldModel/Representation policy 的挂载点 |
| Views 模式（owner 在 host，`*Views` POD 按值进 kernel） | `device_array.hpp` 等 | 全部新模块沿用 |
| `Particles` SoA **已含 per-particle `w`**，`Depositor` 已按 $q\cdot w[i]/\text{area}$ 沉积 | `particles.hpp`, `depositor.hpp` | 权重穿透管线有先例 |
| `Species`/`SpeciesList` 多成分装载（每 species 一个连续 slice + `species_init_kernel`） | `species.hpp` | core/warm/hot 装载框架现成；slice 约定天然避免 warp 内分支 |
| quiet start 采样原语（van der Corput `radical_inverse`） | `particles.hpp` (detail) | δf quiet-pair 直接复用 |
| float hot path + **double diagnostics/reductions** | 全库 | FP64 诊断归约已是现状 |
| 磁化 Boris 路径（`Cfg::has_b0`）"已存在只差 setup+验证"（ARCHITECTURE §7 原话） | `pusher.hpp` | 第一周补 gate 即解锁全部磁化物理 |
| 外部 pump field + B0 driver（`tools/whistler_pump.cu`） | `tools/` | M4/M9 验证链条"外部泵浦 trapping"这级已做过一次 |
| ω–k dispersion、k–t 谱、phase-space、movie 的 Python 管线 | `scripts/` | 色散验证与 chirping 谱诊断的基础 |
| leapfrog + `half_step_back()` 初始化 | `pusher.hpp` | 保留 |

**待开工第一天核对的四项**：① README 称 deposit 为"tiled + chunk-pool-sorted"而 ARCHITECTURE.md 称仅 `AtomicGlobalDeposit` 已实现（`SharedTileDeposit`/`CellOwnedDeposit` 为 tag）——以代码为准；② 两个 open PR 的内容；③ `Cfg::Real` 模板 vs float 硬编码点清单（ARCHITECTURE §7 已承认存在）；④ `migrate()` 目前 = periodic wrap + recell，确认有无排序迁移。

### 10.3 缺失清单（改造的净新增部分）

Yee full-Maxwell 求解器；charge-conserving（Esirkepov）电流沉积；吸收边界系统；非周期粒子边界；δf 权重体系；f0(ℰ,μ) 平衡装载；非均匀 B0(s)（当前 `RunParams::B0[3]` 为常向量）；prescribed drift；注入/outflow 与 chunk pool；增量迁移；HDF5 输出与 checkpoint/restart。

### 10.4 复用率结论

基础设施、配置、测试、装载、诊断框架几乎全部直接可用，重写集中在场求解与沉积两处。**净复用率约 50–60%**（早期基于假设的评估为 30–40%，读到实际代码后上调）。

## 11. 三个关键架构决定

### 11.1 FieldModel 三分支，而非仓库分家

初版计划的"旧代码搬进 legacy/ 冻结"方案**作废**。现有编译期 policy 体系 + 双场模型共享粒子引擎的设计，意味着正确做法是：

```cpp
struct SpectralES {};  struct SpectralDarwin {};  struct YeeMaxwell {};
// SimConfig<..., FieldModel> 编译期选择。
// ES/Darwin 走既有 Simulation<Cfg>（deposit→solve→push 步序）；
// YeeMaxwell 走新 MaxwellSimulation<Cfg>（B半步→push→deposit J→E整步→边界）；
// 三者共享 Particles / Depositor(charge) / Pusher / Diagnostics。
```

好处：每次 ctest 同时验证三条路径；谱法（无网格色散、精确导数）永远是 Yee 版数值色散的对照组；额外白送一个"均匀周期等离子体低频段 Yee vs 谱 Darwin 交叉验证"。代价：模板实例化编译时间上升，用显式实例化 .cu 文件控制。

### 11.2 权重语义拆分

现有 `w[i]` = macro-particle weight（一个 marker 代表的物理粒子数），装载与沉积已按它工作。**保持其语义不变**，δf 新增独立数组：

```cpp
struct Particles {                 // SoA, namespace arc
    float *x, *y;                  // 位置（cell 单位；x=s, y=R_eq·φ）
    float *ux, *uy, *uz;
    float *w;                      // 既有：marker/importance weight（装载时定）
    float *wd;                     // 新增：δf 演化权重 w_p = δf/g（DeltaF 专用）
    uint32_t *cell;                // 加位段：bit[0:23] cell, bit[24:27] pop id, bit[28:31] flags
};
// DeltaF 沉积：q · w[i] · wd[i] · v / area
// FullF  沉积：q · w[i] · v / area − J0_analytic(s)
// p 数组：g = f0 时编译期省略（policy），需要时再加
```

ES/Darwin/full-f 路径零改动，δf 是纯增量。权重精度模式（§12.4）只作用于 `wd`。

### 11.3 Representation 进 Species 而非 SimConfig

同一次运行里 core(δf)+hot(full-f) 必须共存，所以 Representation 是 **per-species 运行期属性 + kernel 按 population 分模板实例**，延续"每 species 一个连续 slice"的现有设计：

```cpp
for (auto& sp : species_list)
    switch (sp.rep) {
        case Representation::DeltaF: launch push_weight<DeltaFPolicy>(slice(sp)); break;
        case Representation::FullF:  launch push_weight<FullFPolicy >(slice(sp)); break;
    }
```

## 12. 时间步、融合 kernel、沉积、迁移与精度

### 12.1 主时间步（MaxwellSimulation）

```
 1. Maxwell B half step
 2. For each population (policy-instanced kernel, per-species slice):
        gather δE, δB, B0(s), geometry/drift 系数
        full-gyrophase Boris velocity push
        prescribed azimuthal drift（× S_d）
        weight update（DeltaF）/ none（FullF）/ fluid step（Fluid, 条件触发）
        advance s, φ
        classify destination / boundary
 3. Deposit currents（per-representation 语义，§6.4）
 4. Sum currents；FullF population 减去解析 J0
 5. Maxwell E update
 6. Absorbing boundary update
 7. Incremental migration; compact
 8. Remove outflow; inject reservoir（flux-weighted）
 9. Diagnostics（FP64 归约）
10. [every K steps] full sort   [every M steps] async checkpoint
```

### 12.2 融合粒子 kernel

最重要的 GPU kernel 为 `gather + push + drift + weight + position + classify` 的融合体。单粒子应尽量：从 global memory 读取一次 → registers 中完成全部更新 → 写回一次。

```cpp
template<class PopulationPolicy>   // 同时编码 Representation 与 f0 参数
__global__ void push_weight_classify(ParticleViews, FieldViews, GeometryViews, RunParams);
```

不同 population 单独实例化（`CorePolicy`/`WarmPolicy`/`HotPolicy`），避免 warp 内 species 分支。

### 12.3 加权沉积与增量迁移

沉积管线见 §6.5。排序 key 从 `tile_id` 升级为 `tile_id + local_cell_id` 并用 `__match_any_sync()` 分组归约——**先 profile 再决定**（Blackwell shared atomics 已很快，可能不需要复杂化）。

prescribed drift 通常慢且方向明确，大部分 marker 每步只会留在原 tile 或进入相邻 tile。因此不应每步完整执行 histogram→scan→scatter，长期方案为队列制增量迁移：

```
resident / outgoing ±s / outgoing ±φ / boundary outflow / overflow
```

每 K 步一次 full sort（现有排序路径保留为该 full-sort），中间用 incremental neighbor migration。

### 12.4 权重精度（M3 对照实验，不预先决定）

RTX 5090 FP64 算术吞吐极低（~FP32 的 1/64），且 FP64 权重使粒子状态、带宽、排序/迁移/checkpoint 数据量全面增大——数亿 marker 下差异可达数 GB。因此不预先决定，M3 中实现三种模式并用长时平衡测试裁决：

1. `float wd`；
2. `float wd + float compensation`（Kahan；注意也多 4 B，不一定比 FP64 存储省，但避开慢速 FP64 算术）；
3. `double wd`（参考实现）。

比较指标：$|\sum_pw_p|$、$\langle w^2\rangle$、$\delta J_{\rm noise}$、长期能量漂移。倾向的生产配置（待数据验证）：marker wd 与更新用 FP32；block/global 权重与能量诊断 FP64 归约；hot 必要时 compensated FP32；长期漂移靠 remapping 控制。

### 12.5 RTX 5090 优化路径

5090 优势：高 FP32 throughput、高显存带宽、32 GB GDDR7、Blackwell 异步内存特性。优先异步搬运规则网格数据（δE、δB、B0、h_φ、drift/equilibrium 系数）；粒子 segment 长度可变，不适合一开始强行使用规则 TMA tile。候选技术（全部以 Nsight Compute profiling 为前提，不提前假设有效）：

`cuda::memcpy_async`、double-buffered shared memory、persistent tile workers、dynamic work queue、CUDA Graph、Blackwell thread-block clusters、TMA、cluster-level work stealing。

### 12.6 I/O 与 checkpoint/restart

- checkpoint 格式在 M0 定义：版本号 + deck 快照 + 全部缩放参数（$S_d$、$\epsilon_L$）+ RNG 状态 + 数组 schema；
- 生产运行为多天级，**无 restart 不可接受**；
- checkpoint 走异步流式输出，与计算重叠；生产数据从 CSV 转 HDF5；
- M12 参数数据库 schema 从 M0 开始设计。

## 13. 资源预算：网格、粒子数、显存与运行时间

参考参数见 §9.1 表。

### 13.1 网格

- 分辨率 $\Delta\approx0.5\,c/\omega_{pe}$（whistler 15–25 cells/波长）；
- $s$：±1000 $c/\omega_{pe}$（±1700 km，覆盖 $|\lambda_m|\lesssim20°$ 源区）→ $N_s\approx4096$；
- $\phi$：局部窗口 256–1024 $c/\omega_{pe}$（物理 400–3500 km ≈ 0.05–0.5 h MLT）→ $N_\phi\approx512$–2048；
- 典型生产网格 **4096×1024 ≈ 4.2×10⁶ cells**，上限 8192×2048 ≈ 1.7×10⁷。

（对照：真实 1 h MLT 扇区需 $N_\phi\gtrsim8000$ 且漂移穿越时间不可负担——再次印证 §9 的缩放必要性。）

### 13.2 粒子数（32 GB 硬约束，保守口径）

每 marker 静态状态 28–36 B（7 float + cell）。**运行时摊销**须计入：sort/compaction scratch、destination key、tile offsets、迁移队列、注入池、outflow lists、Esirkepov 临时量、checkpoint staging、多 population 管理、solver workspace、吸收层辅助网格、安全余量：

$$\approx45\text{–}80\ \mathrm{B/marker}$$

（增量迁移 + 流式 checkpoint 做到位可压至 45–55 B。）

| 口径 | 总 markers |
|---|---|
| **舒适生产范围（规划基准）** | **2–4×10⁸** |
| 优化后可达 | 3–5×10⁸ |
| 极限容量实验 | 5×10⁸+ |

4096×1024 网格上初期总 markers/cell ≈ **50–100**。δf + importance sampling 下不必均匀分配：hot resonant 区域密集采样，远离源区的 core 稀疏；若 fluid core 触发，hot 预算可显著上调。启动前由预算器给出精确分配并强制检查（>90% 显存拒绝启动）。

### 13.3 时间步与总步数

- CFL：$\Delta t\le\Delta x/(c\sqrt2)\approx0.35\,\omega_{pe}^{-1}$；同时 $\Omega_e\Delta t\approx0.1$–0.2，Boris 相位精度足够；
- 一个 chirping element ~数千 $\Omega_e^{-1}$；含增长+饱和+多个 element 的目标时长 $2\times10^4$–$10^5\,\Omega_e^{-1}$；
- → **总步数 2×10⁵–10⁶**。

### 13.4 吞吐三情景（不锁定单一数字）

目标 kernel 含六分量 gather、几何/漂移插值、δf 源项、权重更新、完整轨迹分类、Esirkepov 沉积、正负权重归约——远重于裸 Boris push。公开的高性能 EM PIC 在同代硬件多在 2–6×10⁸ 区间：

| 模式 | 端到端吞吐（particle-step/s） |
|---|---|
| 保守（未深度优化） | 1–2×10⁸ |
| **合理优化（预算基准，M9 目标）** | **3–6×10⁸** |
| 进取目标（stretch goal） | 8×10⁸–10⁹ |

### 13.5 运行时间

以 3×10⁸ markers × 3×10⁵ steps = 9×10¹³ particle-updates 计，纯粒子阶段：10⁹/s → ~25 h；5×10⁸/s → ~50 h；2×10⁸/s → ~125 h。加上场更新、沉积、I/O、诊断：

| 场景 | 时长 |
|---|---|
| M3/M4 物理验证 run | 分钟 – 小时 |
| M10 原型 chirping | ~0.5–1.5 天 |
| **生产 chirping run** | **2–8 天/次** |
| M12 参数扫描（~20 runs） | 1–3 个月机时 |

**结论**：单 5090 产出可发表的自洽 chirping 可行；系统参数扫描要求 M9 至少达到"合理优化"档并严控 $\phi$ 域尺寸。若吞吐停在 2×10⁸，生产 run 拖到数周/次，项目节奏不可接受——这也是 fluid core 选项、域尺寸控制与 M9 优化必须认真对待的原因。

---

# Part IV — 里程碑与代码改造计划（M0–M12）

**使用方式**：每个里程碑一节，含物理目标、任务、文件级代码改动、验收 ctest、deliverables、exit criteria、工作量与完成 tag。验收通过即打 tag 进入下一节。两个硬 gate（M4、M5.5）用 ⛔ 标注：不通过不进入下层物理。

**Phase 划分**：A 基础正确性（M0–M3）｜B 可信 chorus 基线（M4–M5）｜C 尺度化固定 L-shell（M5.5–M7）｜D 表示裁决与优化（M8–M9）｜E 科学结果（M10–M12）。

---

## M0 — 现有代码基线与基建

**物理/工程目标**：可重复的性能与物理基线；显存/时间预算器；checkpoint 格式与 HDF5；磁化 Boris gate。

**任务与代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| 处理两个 open PR | — | 合并或关闭，锁定 main |
| 打基线 tag | — | `v0-spectral-baseline`；此后 12 项既有 ctest 永远保持全绿 |
| ctest 分档 | `CMakeLists.txt`, `tests/` | label：unit / physics / regression |
| run 元数据落盘 | `tools/run_deck.cu` | 输出目录写 deck 副本 + git hash + 时间戳（~20 行） |
| **磁化 Boris gate** | `tests/test_magnetized_boris.cu`（新） | `Cfg::has_b0` 路径补验证：uniform-B 回旋周期、E×B 漂移、μ 守恒三断言（半周工作，解锁一切磁化物理） |
| 预算器 | `scripts/budgeter.py`（新）+ `include/pic/budget.hpp`（新） | §13 公式内嵌；启动打印 markers/cell、预计吞吐与墙钟；>90% 显存拒绝启动；输出 Π_d/Π_t/ε_L（M5.5 后启用） |
| checkpoint schema + HDF5 | `include/pic/checkpoint.hpp`（新）, `include/pic/io_hdf5.hpp`（新） | §12.6 格式；先空实现+格式单测 |
| 归一化补页 | `ARCHITECTURE.md` §2 | 现有 ω_pe=1 contract 补 Ω_e/ω_pe、c、Yee 时空步约定 |
| Nsight 基线 | `docs/PROFILE_BASELINE.md`（新） | `deposit_bench` + `run_deck` 两入口存档；每 kernel 时间/带宽/particles-s |
| §10.2 四项核对 | — | deposit 完成度、open PR、float 硬编码清单、migrate 现状；结论回写本文档 |

**M0 状态回写（2026-07-13，§10.2 四项核对结论）**：
1. **Deposit 完成度**：以代码为准 — `AtomicGlobalDeposit` 与 `SharedTileDeposit`（全网格私有化，仅小网格；当前 `Cfg` 默认）均已实现，EM 路径另有 tiled+binned `deposit_rho_j`（4.9 G/s，为 EM 瓶颈）；`CellOwnedDeposit` 仍为 tag-only。README 与 ARCHITECTURE 的分歧已解决。
2. **两个 open PR**（#1 input-deck、#2 SharedTileDeposit）：内容均已在 main 上（deck 系统与 SharedTileDeposit 早经直接提交合入）→ 建议关闭为 superseded（待作者确认）。
3. **float 硬编码清单**：`SimConfig` 有 `Real` 模板参但 `Cfg` 固定 float；particles/pusher/fields/sources 直接用 float。符合"FP32 热路径 + FP64 诊断归约"的既定策略，暂不动。
4. **migrate() 现状**：v1 = periodic wrap + recompute cell（particles.hpp:443），无排序迁移；EM 路径另有 `sort_by_tile` chunk-pool 排序。与本文档假设一致。

M0 交付：`test_magnetized_boris`（E×B 漂移 1.5e-5、漂移系 μ 1e-5/50 周期、斜 B0 不变量）+ `test_checkpoint_format`（schema v1 往返）+ `scripts/budgeter.py`（§13 公式）+ run 元数据（RUN_META.txt + deck 快照，arcsim/chirp1d 已接）+ `docs/PROFILE_BASELINE.md` + ctest label（unit/physics/regression）。**注**：M4 的 1D chirping 物理已提前经独立 1D 载具 `chirp1d` 通过（Tao GRL17 复现，tag `v1d-chirping-tao2017`，见 docs/TAO2017_REPRODUCTION.md）——M1–M3 仍按序推进，M4 届时以 2D 代码退化复验。

**Deliverables**：benchmark report、tagged release、profile 基线、预算器。
**验收 ctest**：既有 12 项 + `magnetized_boris` 全绿。
**Exit criteria**：所有现有测试可重复通过；性能基线明确。
**工作量**：0.5 月　**Tag**：`v0-spectral-baseline`

---

## M1 — Cartesian full-Maxwell + charge conservation

> **状态：✅ 完成（2026-07-17，tag `v1-maxwell`）。** 四个验收 ctest 全过：
> `yee_vacuum`｜`esirkepov_continuity`｜`yee_langmuir`｜`yee_vs_darwin_lowfreq`
> （冷等离子体 whistler 本征模：Darwin 实测 0.45−0.05%、Yee 实测 0.449−0.03%，
> 两分支差 0.21% ≈ 位移电流理论差 0.23%）。∇·B rms 精确为 0（round-off ~1e-10），
> Gauss 残差漂移钉在原子舍入底 ~2e-7（`MaxwellSimulation::residuals()` +
> arcsim `<pref>maxwell.csv` 运行时输出）。`[field] model = yee` 已进 arcsim
> （带 CFL 守卫）。偏差说明：guard cells/StaggeredAccessor 推迟到 M2（v1 用模
> 索引）；诊断落在 simulation_maxwell.hpp 而非 diagnostics.hpp；dispersion_theory
> .py 未单列（色散求根内联在各测试里）。额外收获：修复两个 ny=1 越界 bug
> （Darwin tile-halo 装载、GlobalJSink 4 点 stencil），compute-sanitizer 全绿。
> 已提前完成的 M9 tiled 沉积也在本分支（case 7 生产验证 2.6× OSIRIS）。

**物理目标**：平直 2D Cartesian 网格上可靠的 2D3V full-Maxwell PIC 基础——Yee staggered E/B 推进、Esirkepov 守恒电流沉积、Gauss law 策略、periodic 场边界、场能量诊断；真空波、等离子体色散验证。

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| FieldModel 三分支 | `include/pic/config.hpp` | `SpectralES/SpectralDarwin/YeeMaxwell` 进 SimConfig；ES/Darwin 零改动 |
| guard cells + staggering | `include/pic/grid.hpp` | `Grid` 加 guard 宽度；新增 `StaggeredAccessor`（半格偏移封装，粒子侧不感知） |
| 六分量场 | `include/pic/fields.hpp` | 填满已预留 EM 槽位 δEx…δBz + `FieldViews` |
| 电流源 | `include/pic/sources.hpp` | 填满已预留 Jx,Jy,Jz + Esirkepov 中间量 |
| Yee 推进 | `include/pic/solver_yee.hpp`（新） | B 半步 / E 整步 kernel |
| 新主循环 | `include/pic/simulation_maxwell.hpp`（新） | `MaxwellSimulation<Cfg>`（§12.1 步序）；不动老 `Simulation<Cfg>` |
| Esirkepov | `include/pic/depositor_esirkepov.hpp`（新） | 与既有 `Depositor` 并列，共用 `CicStencil`/Views |
| 连续性/∇·B 诊断 | `include/pic/diagnostics.hpp` | 逐 cell FP64 归约 |
| 理论脚本 | `scripts/dispersion_theory.py`（新） | 冷等离子体 R/L 模求根；M1/M3/M4 复用 |

**验收 ctest（新增 4）**：`yee_vacuum_dispersion`（vs 理论）｜`esirkepov_continuity`（单粒子任意轨迹残差 ≈ FP32 舍入）｜`yee_langmuir`（与既有 ES 测试同物理、双路径对照）｜`yee_vs_darwin_lowfreq`（均匀周期等离子体低频段交叉验证——三分支架构的红利）。
**Exit criteria**：数值色散与理论一致；∇·B 受控；电荷连续性达到设计误差。
**工作量**：1.5–2 月　**Tag**：`v1-maxwell`

---

## M2 — 吸收边界研究

> **状态：✅ 完成（2026-07-17，tag `v2-boundary`）。** 方案：Umeda 乘性掩膜层
> （chirp1d 方案移植 2D），`[boundary] x = damping|hybrid nd numax`；hybrid 额外
> 阻尼层内粒子横向动量。**生产配置 hybrid nd=256 ν_max=0.1：准平行 R =
> 0.84/0.87/0.87% @ 0.4/0.25/0.1 ω_ce——整个 chorus 频段 < 1%（一套参数）。**
> benchmark：`tools/boundary_reflection.cu`（天线包 + 驱动频率窄带解调探针；
> 天线必须电子回旋旋向 (J_y+iJ_z)∝e^{+iω₀t}，反向在 ω<ω_ce 是 evanescent
> L 模什么都辐射不出）。ctest 门槛：`boundary_vacuum`（真空脉冲 R=0.14%）+
> `boundary_whistler`（生产配置 R=0.74% < 1%）。失效模式全记录在
> docs/BOUNDARY_STUDY.md：ν_max 过大→梯度反射；nd ≪ λ→透明；**密度向层内
> 递减反而使 R 变差 3–9×**（低密度→vg 加倍→积分阻尼减半——M5+ 偶极密度
> 剖面的设计规则：按层处局部低密度配 nd/ν_max）。偏差说明：未做独立
> boundary.hpp 抽象（deck 开关 + yee2d/simulation_maxwell 内实现足够）；
> 斜入射用 ny=1 斜 B0 代理测得 θ≤15° ≈2%（够 M10 准平行用例），θ≥30° 的
> ~13% 底座疑为慢准静电支污染，真 2D 斜包 benchmark 留作 refinement；
> PML 候选按计划条件未触发（掩膜达标）。

**物理目标**：不预设 PML；系统比较 field damping / current damping / particle damping / vacuum-gap PML / 混合方案；**等离子体条件**下的 $R(\omega,\theta)$ benchmark 与密度渐变反射专项。

**代码改动**：

| 动作 | 文件 |
|---|---|
| 边界抽象接口 | `include/pic/boundary.hpp`（新）：`FieldBoundary::apply(FieldViews,dt)` + `ParticleBoundary::classify_and_absorb()`，挂 `MaxwellSimulation` |
| 各候选实现 | `include/pic/boundary_damping.hpp`, `boundary_pml.hpp` 等（每方案一文件）；deck `[fields] boundary_s = damping\|pml\|hybrid` |
| 非周期粒子路径 | `include/pic/particles.hpp`：`migrate()` 扩展 s 端吸收分支（打 flag，压缩延后到 M7） |
| benchmark 工具 | `tools/boundary_reflection.cu`（新）：等离子体内发射 whistler 波包测反射功率比 |
| 研究报告 | `docs/BOUNDARY_STUDY.md`（新）：选型结论 + 失效模式 |

**验收 ctest**：`boundary_reflection_whistler`（0.1–0.5 Ω_e 准平行入射 R < 阈值，deck 可调，默认 1%）｜`boundary_vacuum_pml`。
**Exit criteria**：选定方案 whistler 频段反射低于阈值且有失效模式记录。
**工作量**：1–1.5 月　**Tag**：`v2-boundary`

---

## M3 — 单 population electromagnetic δf

> **状态：🔶 Phase 1 完成（2026-07-17）——Yee 分支 DeltaF 路径 + 两个门槛。**
> 已落地：Particles 加 `wd`（进 sort 双缓冲/views）；`[species] rep = deltaf`；
> 权重方程融合进 yee_advance_particle（chirp1d/Tao eq19 离散，波场 only——
> gather 已拆 δB 与 B0，∂ln f0 乘加系数，B0∥x̂ + 回旋对称约束）；DeltaF 沉积
> `q·w·wd·v`（flat + tiled 两路径，k_rho_nodes 同步）。门槛：
> `deltaf_equilibrium`（各向同性 + 种子，wd 有界 ×1.17，flat/tiled 逐位一致）
> `deltaf_growth`（单 bi-Max 平行哨声增长率 vs Z 函数理论：**γ=2.60e-3 vs
> 2.872e-3（9.4%，ppc=1600）；ω_r=0.18686 vs 0.1868 精确**；ppc 收敛
> 400/1600/6400 → 26%/9%/4% ≈ 1/√ppc——回旋共振带 marker 分辨率所限，
> 已写进测试头）。ctest 26/26。
> **待做（Phase 2）**：FullF policy + `deltaf_vs_fullf_consistency`；
> `deltaf_landau`（对照 solver_es）；精度三模式 + WEIGHT_PRECISION.md；
> 权重诊断模块（Σwd、⟨wd²⟩）；→ 齐后 tag `v3-deltaf`。

**物理目标**：w/wd 数据模型、非线性权重方程、加权守恒沉积、精度三模式对照、**full-f deposition policy 同步实现**；平衡保持、Landau damping、whistler 色散、各向异性增长率 benchmark。

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| 权重语义拆分 | `include/pic/particles.hpp` | 保留 `w`；新增 `wd`；cell 位段（§11.2） |
| Representation | `include/pic/species.hpp` | `Species` 加 `rep{DeltaF,FullF}`、uth∥/uth⊥、f0 模型 id、独立 markers-per-cell |
| f0 接口 | `include/pic/f0/f0_bimax.hpp`（新目录） | ∂ln f0 预计算系数（乘加形式，§6.2） |
| 权重更新 | `include/pic/weight_update.hpp`（新） | 融合进 pusher；`push_weight<DeltaFPolicy/FullFPolicy>` 按 slice 分实例 |
| 双沉积语义 | `include/pic/depositor_esirkepov.hpp` + `include/pic/j0_analytic.hpp`（新） | §6.4：DeltaF `q·w·wd·v`；FullF `q·w·v − J0`（J0 先返回 0，M6 后含漂移环） |
| 精度三模式 | `include/pic/config.hpp` | `WEIGHT_FP32 / FP32_KAHAN / FP64` 编译开关，仅作用 wd |
| deck schema | `include/pic/deck.hpp` | 每 species `rep=`、各向异性、ppc |
| δf 装载 | `include/pic/particles.hpp` | quiet-pair 复用既有 `radical_inverse` |
| 权重诊断 | `include/pic/diagnostics.hpp` | Σwd、⟨wd²⟩、max\|wd\|（FP64 归约） |

**验收 ctest（新增 6）**：`deltaf_equilibrium_hold`（无扰动长跑 wd 无系统增长）｜`deltaf_vs_fullf_consistency`（**同一 Maxwellian 双表示扰动场统计一致——防 bug 重点**）｜`deltaf_landau`（vs 理论，并与既有 solver_es 谱法对照）｜`deltaf_whistler_dispersion`｜`deltaf_anisotropy_growth`（vs 求根脚本，<10%）｜`weight_precision_longterm`（三模式 10⁵ 步 → `docs/WEIGHT_PRECISION.md`）。
**Exit criteria**：无扰动权重无系统性增长；线性响应与理论一致；δf 相较 full-f 显著降噪（顺带量化）；精度模式有数据结论。
**工作量**：1.5–2 月（w 管线/quiet start/FP64 归约现成）　**Tag**：`v3-deltaf`

---

## ⛔ M4 — 退化 1D 复现【PHYSICS GATE】

**物理目标**：$N_\phi=1$–4（periodic），复现文献中的 whistler 线性增长 → nonlinear trapping → rising-tone chirping（Katoh-Omura / Tao 类设置）；同步产出第一轮 hot 表示对比数据。**通过前禁止实现任何 MLT injection 物理。**

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| 1D 配置路径 | `decks/chirping_1d.ini`（新） | 纯 deck，无新物理代码 |
| 解析 B0(s) 先行 | `include/pic/background_b0.hpp`（新） | `RunParams::B0[3]` 常向量 → 1D 表接口；M4 用解析抛物近似，M5 换 dipole |
| pump 系工具升级 | `tools/whistler_pump.cu` → `tools/chirping_1d.cu`（新） | **起点优势**：An et al. 2019 外部泵浦 trapping 已复现过；验证链条"外部泵浦→自洽触发"从它长出 |
| chirping 诊断 | `scripts/plot_chirping.py`（新） | 动态谱/瞬时频率/trapped island，复用既有 k–t 谱管线 |
| 表示对比 | `docs/HOT_REP_ROUND1.md`（新） | σ_wd 演化、SNR_Jres（区分噪声来源）、chirping rate 收敛 |

**验收**：`chirping_1d_rising_tone`（人工评审 + 半自动断言）：频率漂移率与文献参数设置定量可比；对 marker 数收敛；关闭 pump 后可自洽触发。
**Exit criteria（硬性）**：复现成功。
**工作量**：1–2 月　**Tag**：`v4-gate-1d` ⛔

---

## M5 — 非均匀背景与平衡分布

**物理目标**：dipole $B_0(s)$、局部基、$h_\phi(s)$ 元数据；$f_0(\mathcal E,\mu)$ 装载；mirror/bounce 动力学；高纬粒子边界；长时零扰动平衡测试。

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| dipole 几何 | `include/pic/geometry_dipole.hpp`（新） | B0(s)/h_φ(s)/局部基 1D 预计算表（HDF5）；`scripts/gen_dipole_tables.py`（新）生成 |
| f0(ℰ,μ) | `include/pic/f0/f0_emu.hpp`（新） | 赤道定义 core/warm/hot，(ℰ,μ) 守恒映射（§6.3）；装载复用 quiet start |
| gather 扩展 | `include/pic/pusher.hpp` | 6 分量 δ 场 + B0(s) 表插值 |
| s 端粒子边界 | `include/pic/boundary.hpp` | 高纬吸收 + loss-cone flux 诊断（接 M2 接口） |
| deck | `decks/` | `[geometry] model=dipole L0=4.0 table=...` |

**Deliverables**：geometry module、单粒子 mirror benchmark、平衡装载工具、bounce-orbit 诊断。
**验收 ctest**：`mirror_bounce_orbit`（mirror point/bounce period vs 解析；**交叉验证用 taiparticle-uniform 独立轨道积分**）｜`mu_conservation`｜`f0_emu_hold`（无波 10⁴ Ω_e⁻¹ 密度剖面漂移 <1%）。
**Exit criteria**：mirror/bounce 正确；μ 在预期精度内守恒；equilibrium markers 无显著松弛。
**工作量**：1.5–2 月　**Tag**：`v5-dipole`

---

## ⛔ M5.5 — 尺度与相似性设计【SCIENCE-VALIDITY GATE】

**目标**：产出 `docs/SCALING_AND_SIMILARITY.md`（§9 全部内容的正式推导版）：$\Pi_d$、$\Pi_t$、$L_\phi/\lambda$、$v_d/v_g$、$\epsilon_L$ 的定义与测量方法；四种缩放策略各自能回答的科学问题；$S_d$ 扫描外推方法论。

**代码改动**（轻）：`scripts/budgeter.py` 扩展输出无量纲组；`include/pic/checkpoint.hpp` + run 输出把 $S_d$、$\epsilon_L$ 设为强制字段。

**Exit criteria**：文档评审通过；M6 之后所有运行参数可映射回无量纲空间。
**工作量**：2–3 周　**Tag**：`v5.5-gate-scaling` ⛔

---

## M6 — Prescribed gradient-curvature drift

**物理目标**：由三维 dipole 计算 $\nabla B$、curvature，预计算 $C_\perp(s)$、$C_\parallel(s)$；$S_d$ 显式参数；drift 纳入完整轨迹与守恒沉积；double-counting 测试；bounce-averaged drift 验证。

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| drift 表生成 | `scripts/gen_drift_tables.py`（新） | 3D dipole 投影 → C⊥(s), C∥(s) HDF5 |
| drift device 函数 | `include/pic/drift.hpp`（新） | $v_{d\phi}=S_d[C_\perp v_\perp^2+C_\parallel v_\parallel^2]$，融合进 push_weight |
| 位移入沉积 | `include/pic/depositor_esirkepov.hpp` | drift 位移并入轨迹分解 |
| CFL 扩展 | `include/pic/budget.hpp` | 加 $\vert\Delta\phi\cdot h_\phi\vert$ 项（预算器 + 运行时断言） |
| deck | `[drift] S_d=... table=...` | 输出元数据强制带 S_d |

**验收 ctest**：`drift_period_vs_theory`（vs taiparticle 独立积分与解析 dipole guiding-center 理论）｜`drift_no_double_counting`（面内 pusher 在 $B_0(s)$ 下解析方位漂移 ≈0 的实测确认，§7.3）｜`drift_continuity`（连续性残差不因 drift 变化）。
**Exit criteria**：drift period 与理论/独立积分一致；能量-投掷角依赖正确；守恒不破坏。
**工作量**：3 周　**Tag**：`v6-drift`

---

## M7 — 方位注入与 outflow

**物理目标**：GPU chunk pool 与自由槽管理；flux-weighted injection（§8.3）；quiet-pair 采样；零权重 equilibrium reservoir；非零权重 anisotropic injection；absorbing outflow；checkpoint/restart 投产；粒子/权重平衡诊断；长稳测试。

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| chunk pool | `include/pic/chunk_pool.hpp`（新）+ `device_array.hpp` 扩展 | 分 chunk + 自由槽；沿用 slice-per-species 约定预留注入段 |
| 通量采样 | `include/pic/injection.hpp`（新） | $g_{\rm flux}\propto\dot\phi g$ 反演表；wd=0 reservoir；wd≠0 各向异性注入；FullF 按总分布通量 |
| φ 边界 | `include/pic/boundary.hpp` | upstream 注入 / downstream 吸收；与场吸收区 buffer 隔离断言（§8.3） |
| restart 投产 | `include/pic/checkpoint.hpp` | 完整实现（RNG、chunk pool 状态）；异步流式写出 |
| 收支诊断 | `include/pic/diagnostics.hpp` | 粒子数/权重 balance 表 |

**验收 ctest**：`injection_spectrum`（KS 检验 vs 目标分布）｜`reservoir_quiet`（平衡注入长跑无虚假波、marker density 稳定）｜`outflow_no_reflection`｜`restart_consistency`（restart 后统计一致）。
**Exit criteria**：注入通量符合目标分布；equilibrium 注入不产生非物理波；长运行稳定；outflow 不反射。
**工作量**：1.5–2 月　**Tag**：`v7-injection`

---

## M8 — Population representation study

**目标**：数据裁决生产配置。用 M4 1D 基准 + 2D injection 前期基准比较：① core/warm/hot 全 δf；② core/warm δf + hot full-f。fluid core 仅当 §6.6 触发条件成立时加入第三方案。裁决指标见 §6.1（含噪声来源分解）。

**代码改动**（无新物理）：`decks/rep_study_*.ini` 对比组；`diagnostics.hpp` 加 SNR_Jres 分解（wd 方差项 vs FullF 平衡采样项）；`docs/REPRESENTATION_STUDY.md` 收口，锁定生产配置。

**Exit criteria**：生产配置由数据决定并文档化；chirping 收敛、噪声、权重方差、内存、运行时间五维对比完整。
**工作量**：1 月　**Tag**：`v8-rep`

---

## M9 — 5090 tile engine 优化

**目标**：端到端 ≥3×10⁸ particle-step/s（"合理优化"档）；大域 run 稳定填满 32 GB。每项优化独立 PR，**先 Nsight 后动手**。

**代码改动**：

| 动作 | 文件 | 说明 |
|---|---|---|
| classify 并入融合 kernel | `include/pic/pusher.hpp` | gather+push+drift+weight+classify 收口为单 kernel |
| 增量迁移 | `include/pic/migration.hpp`（新） | §12.3 队列制；full sort 降频每 K 步；**先核对既有 chunk-pool-sorted deposit 完成度，可能有半成品** |
| 排序 key 升级 | `depositor_esirkepov.hpp` | tile+local_cell + `__match_any_sync`——先 profile Blackwell shared atomics 再定 |
| async 双缓冲 | `include/pic/tile_engine.hpp`（新） | 场/几何 tile `cuda::memcpy_async` |
| CUDA Graph | `simulation_maxwell.hpp` | 主循环固化；checkpoint 异步输出 |
| 报告 | `docs/PROFILE_M9.md` | 每项 before/after Nsight 数据 |

**验收**：`deposit_bench` 扩展版端到端 ≥3×10⁸ particle-step/s；粒子状态每步接近单次读写；deposition atomic contention 显著下降。
**工作量**：1–2 月（与 M8/M10 前期并行）　**Tag**：`v9-perf`

---

## M10 — 局部固定 L-shell chirping（Cartesian 展开）

**物理目标**：首个 $(s,\phi)$ 固定 L-shell 自洽模拟——局部 MLT 窗口、Cartesian Maxwell、$B_0(s)$、prescribed drift、δf/full-f 电子、s 端吸收、φ 注入/流出；时变注入、有限方位源宽度、$S_d$ 扫描、rising/falling tone；weak-noise 初始化、trapped-particle 相空间分析、参数扫描（hot 密度、各向异性、漂移速度、注入宽度、$B_0$ 梯度、warm 参数、域尺寸）。

**代码改动**：无大块新 C++/CUDA。`decks/chirping_2d_*.ini` 系列；`scripts/` 下共振相空间、源区宽度/相干性、波法向、$\mathbf J\cdot\mathbf E$、pitch-angle scattering、precipitation flux 诊断；`scripts/campaign.py` 参数扫描驱动 + 参数数据库（M0 schema 落地）。

**Deliverables**：首个自洽 2D chirping、frequency-time spectrogram、共振相空间分析、收敛研究、源区可视化。
**Exit criteria（收敛四条）**：对 marker 数收敛；对 grid/Δt 收敛；不依赖边界反射；不由数值噪声触发；可由 nonlinear resonant current 解释。
**工作量**：2–3 月　**Tag**：`v10-chirping-2d`

### M10-SQ：科学问题——注入是否同时触发下带与上带 chorus？（2026-07-17 加入）

上带（0.5–0.8 f_ce）的激发机制无共识；候选：①双成分线性（低能各向异性成分独立驱动
上带，E_res(0.6 f_ce) ~ 数百 eV）；②非线性父子（下带上升调扫过 0.5 f_ce 被
v_R = −v_ph 简并阻尼掐断，越过部分成上带，Omura 系）；③下带谐波/波-波耦合
（Gao/Lu 系）；④高度斜传播上带由场向低能电子 Landau 共振驱动（Mourenas/
Artemyev 系）。0.5 f_ce 缺口可能"多机制过定"，观测指纹重叠，需受控模拟裁决。

**四个因果切断实验（全部只用已有/计划内基础设施）**：
1. **成分开关**（M3 δf + M7 注入）：只注高能 / 只注低能 / both，上带是否只随
   低能成分出现 → 裁决①。
2. **弑父实验**（M2 掩膜推广为频域/k 域选择性阻尼）：人为杀死下带，看上带
   是否仍然生长 → 裁决②③。观测上不可能做的实验。
3. **维度开关**：ny=1 杀死斜传播机制 vs 2D 放开 → 裁决④。
4. **共振追踪**（δf wd 加权相空间）：上带供能电子在回旋共振速度还是 Landau
   共振速度 → ①/④ 直接指纹。

前置：注入分布须支持能量依赖各向异性（M7 注入云参数化时预留 per-成分
uth⊥/uth∥(E)）；选择性阻尼 = k_damp_x 的谱域变体（小改动）。文献引用凭
记忆待核对（Fu 2014 双成分 PIC；Omura 2009 非线性缺口；Gao 谐波；
Li Jinxing 2019 缺口成因）。

---

## M11 — 曲面修正（weak metric 必做；full metric 条件触发）

| 级别 | 内容 | 代码改动 |
|---|---|---|
| **Weak metric（必做）** | $h_\phi(s)$ 进漂移换算（M6 已含）、cell 面积权重进沉积、能量诊断 Jacobian；对照 run 量化 vs Cartesian | `geometry_dipole.hpp` 扩展 + `depositor_esirkepov.hpp`/`diagnostics.hpp` 面积权重 |
| **Full metric（仅当 weak 影响 >10%）** | curvilinear Maxwell、metric-aware Yee staggering、Jacobian-aware 沉积、曲面吸收边界、局部基输运、几何 Poynting 验证 | `solver_yee_curvilinear.hpp`（新）——独立立项，不在本计划排期 |

**Exit criteria（若触发 full）**：charge conservation、$\nabla\cdot\mathbf B$、能量一致性、正确的几何传播与聚焦。
**工作量**：weak 2–3 周；full 另计 2–4 月。

---

## M12 — 生产科学 campaign

**核心科学问题**：
1. 有限方位宽度的时变 hot-electron injection 是否改变 chorus onset？
2. chirping 是否在整个注入结构内同步产生？
3. source region 的方位宽度由什么决定？
4. drift energy dispersion（经 $S_d$ 扫描外推）如何影响 chirping repetition？
5. warm/core electron kinetic response 如何影响 growth、saturation、damping、obliquity？
6. source-region 边缘与中心是否产生不同 chorus elements？
7. 波包是否可以沿 $\phi$ 方向耦合相邻 source regions？
8. injection 持续时间是否控制 element spacing 和 repetition period？

**Deliverables**：production simulations、可复现参数数据库（含全部 $S_d$、$\epsilon_L$ 记录）、methods paper、physics paper、开源发布与文档。

**代码侧**：campaign 驱动、数据库、发布清理（license、README 科学版、docs 汇编为 methods paper 附录）。

---

## 里程碑总表

| 里程碑 | 关键新文件 | 工作量 | Tag |
|---|---|---|---|
| M0 | budget.hpp, checkpoint.hpp, io_hdf5.hpp, test_magnetized_boris | 0.5 月 | v0-spectral-baseline |
| M1 | solver_yee.hpp, simulation_maxwell.hpp, depositor_esirkepov.hpp | 1.5–2 月 | v1-maxwell |
| M2 | boundary*.hpp, boundary_reflection.cu | 1–1.5 月 | v2-boundary |
| M3 | weight_update.hpp, f0/, j0_analytic.hpp | 1.5–2 月 | v3-deltaf |
| M4 ⛔ | chirping_1d.cu, plot_chirping.py, background_b0.hpp | 1–2 月 | v4-gate-1d |
| M5 | geometry_dipole.hpp, f0_emu.hpp | 1.5–2 月 | v5-dipole |
| M5.5 ⛔ | SCALING_AND_SIMILARITY.md | 0.5 月 | v5.5-gate-scaling |
| M6 | drift.hpp, gen_drift_tables.py | 0.75 月 | v6-drift |
| M7 | chunk_pool.hpp, injection.hpp | 1.5–2 月 | v7-injection |
| M8 | REPRESENTATION_STUDY.md | 1 月 | v8-rep |
| M9 | migration.hpp, tile_engine.hpp | 1–2 月（并行） | v9-perf |
| M10 | decks + campaign.py + chirping 诊断 | 2–3 月 | v10-chirping-2d |
| M11 weak | 面积权重 + Jacobian | 0.5–0.75 月 | v11-metric |
| **合计到 M10** | | **13–18 人月** | |

---

# Part V — 验证策略

项目采用逐层验证，不直接从最终 chirping simulation 开始。每层对应具体 ctest（Part IV 已绑定），本节给出完整物理清单。

## V.1 粒子轨道验证

无扰动背景中测试：
- uniform-B gyro orbit（回旋周期与半径）；
- magnetic moment conservation；
- mirror point 位置与 bounce period；
- 能量守恒；
- 局部与 bounce-averaged 方位漂移，及其 energy/pitch-angle 依赖；
- $v_{d,\phi}^{\rm code}$ vs 解析 dipole guiding-center theory 与独立轨道积分（taiparticle 系工具）。

## V.2 Maxwell solver 验证

- vacuum electromagnetic-wave propagation 与数值色散；
- Gauss-law preservation 与 $\nabla\cdot\mathbf B$；
- **等离子体条件下**吸收边界反射系数 $R(\omega,\theta)$（含密度渐变段专项）；
- 能量守恒；
- Cartesian Yee benchmark；
- Yee vs 谱 Darwin 低频交叉验证；
- （若触发 full metric）curvilinear divergence/curl identities、geometric Poynting flux、体积/面积权重。

## V.3 δf / 表示方法验证

- $w=0$ equilibrium 维持（$D_0f_0/Dt\approx0$；无波时 $w_p(t)\approx0$）；
- homogeneous Maxwellian response；
- electrostatic Landau damping（与谱法 ES solver 双路径对照）；
- electromagnetic whistler dispersion（热修正）；
- anisotropy-driven linear growth 与 multiple-population susceptibility；
- marker-number convergence 与 weight variance growth；
- **同分布 DeltaF vs FullF 一致性**（扰动场统计层面）。

## V.4 注入验证

- injected flux distribution 与 energy/pitch-angle spectrum（KS 检验）；
- particle-number/weight balance；
- upstream reservoir stability 与 downstream absorption；
- zero-weight equilibrium injection 不产生虚假场；
- 正确的 drift time across the domain；
- restart 一致性。

## V.5 Chirping 验证（逐级）

1. homogeneous whistler instability；
2. externally pumped whistler trapping（起点：An et al. 2019 复现经验）；
3. self-consistent nonlinear saturation；
4. inhomogeneous $B_0(s)$；
5. rising-tone chirping（M4 gate：1D 复现文献）；
6. repeated chirping；
7. 时变方位注入结构；
8. finite-width MLT source region。

诊断全集：dynamic spectrum、instantaneous frequency、wave amplitude envelope、wave-normal direction、resonant phase、trapped-particle island、$\delta f(v_\parallel,v_\perp)$、nonlinear resonant current、$\mathbf J\cdot\mathbf E$、pitch-angle scattering、precipitation flux、chirping rate、source-region width and coherence。

---

# Part VI — 风险登记

**R1 — δf 权重发散**。非线性 chirping 可能使局部 $|\delta f|\sim f_0$，导致 large weights、cancellation noise、δf 优势丧失。
缓解：监测 $\langle w^2\rangle$（$\sigma_w>0.3$ 为经验预警值，非硬切换标准）；importance sampling；remapping；adaptive marker replenishment；**hot 的 full-f 路径从 M3 起就存在**（不是事后补救）；hybrid δf/total-f 接口。

**R2 — 平衡非定常**。错误的 $f_0$ 自行松弛并产生虚假波动。
缓解：构造 $f_0(\mathcal E,\mu)$；orbit-equilibrium 测试；no-wave 长时运行；解析验证 $D_0f_0/Dt$。

**R3 — prescribed drift 破坏守恒**。drift displacement 未进入 current deposition 会破坏连续性方程。
缓解：drift 是完整轨迹的一部分；charge-conserving trajectory deposition；连续性残差诊断；$|\Delta\phi h_\phi|$ CFL 项（$S_d$ 放大后重点检查）。

**R4 — 背景漂移电流污染扰动场**。完整平衡 drift current 可能产生非目标背景场。
缓解：只推进扰动场；按 §6.4 语义沉积（FullF 扣除解析 $J_0$）；equilibrium ring current 视为外部支撑；reduced-model 能量核算文档化。注意扣除解析 $J_0$ 不消除平衡采样噪声（M8 指标区分）。

**R5 — 吸收边界与粒子 reservoir 相互作用**。边界注入可能在吸收层附近形成虚假电流或密度梯度。
缓解：物理 reservoir 与吸收区分离 + buffer zone；反射与注入独立 benchmark；禁止在吸收层 cell 内注入。

**R6 — metric-aware solver 过于复杂**。真曲面 Maxwell 与守恒沉积难度显著高于 Cartesian。
缓解：weak-metric 先行并量化影响；full metric 条件触发（>10%）；几何全部走接口；保留平直 metric 回归路径；metric 项增量加入。

**R7 — 32 GB 显存上限**。多 population、sorting buffers、吸收层与队列可能快速消耗显存。
缓解：按 45–80 B/marker 摊销口径规划（舒适区 2–4×10⁸）；policy 省略 p 数组；单缓冲增量迁移；memory pool；启动预算器强制检查；压缩元数据；后期可选 multi-GPU。

**R8 — 表示方法假设失败导致重写**。
缓解：per-species Representation 架构；三种沉积语义从 M3 起并存；M4/M8 数据裁决；fluid core 条件触发保底。

**R9 — 范围蔓延 / 单人执行风险**（这类项目最常见死因）。
缓解：两个硬 gate（M4、M5.5）不通过不进入下层物理；fluid core 与 full metric 均为条件触发而非兴趣触发；每个里程碑 exit criteria 量化；季度工作量记账（偏差 >30% 重排优先级）；既有 12 项 ctest 永远全绿。

**R10 — 离子物理缺失的时效**。长运行跨越 $\Omega_i^{-1}$，离子极化电流与低混杂响应被丢弃。
缓解：能量核算文档注明；对 whistler 频段一阶物理影响有限；若 EMIC/低混杂耦合被证明重要，独立立项扩展。

---

# Part VII — 软件架构、工作量、执行纪律与成功定义

## VII.1 目标软件架构（基于现有仓库增量演化）

现有结构（`include/pic/` header-only + `src/` + `tests/` + `tools/` + `decks/` + `scripts/`）保持不变，增量加入：

```
include/pic/
├── 既有（保留/扩展）
│   ├── cuda_utils.hpp  device_array.hpp        GPU 基建（原样）
│   ├── config.hpp                              + FieldModel 三分支、Representation、精度开关
│   ├── species.hpp                             + rep / 各向异性 / f0-id / 独立 ppc
│   ├── grid.hpp                                + guard cells、StaggeredAccessor
│   ├── fft.hpp  spectral.hpp  solver_es.hpp    冻结保留（回归基线 + ES 对照）
│   ├── (darwin solver)                         冻结保留（低频交叉验证）
│   ├── sources.hpp  fields.hpp                 + J 三分量 / δE δB 六分量
│   ├── particles.hpp                           + wd、cell 位段、吸收分支、quiet-pair
│   ├── depositor.hpp                           保留（charge，ES/Darwin 用）
│   ├── pusher.hpp                              核心复用 + 融合扩展
│   ├── diagnostics.hpp                         + 权重统计 / 连续性 / ∇·B / 收支 / SNR 分解
│   ├── simulation.hpp                          不动（ES/Darwin 主循环）
│   └── deck.hpp                                + [fields][geometry][drift][injection] schema
└── 新增
    ├── solver_yee.hpp  simulation_maxwell.hpp  depositor_esirkepov.hpp     (M1)
    ├── boundary.hpp  boundary_damping.hpp  boundary_pml.hpp                (M2)
    ├── weight_update.hpp  j0_analytic.hpp  f0/{f0_bimax,f0_emu}.hpp        (M3,M5)
    ├── background_b0.hpp  geometry_dipole.hpp                              (M4,M5)
    ├── drift.hpp                                                           (M6)
    ├── chunk_pool.hpp  injection.hpp                                       (M7)
    ├── migration.hpp  tile_engine.hpp                                      (M9)
    ├── budget.hpp  checkpoint.hpp  io_hdf5.hpp                             (M0)
    └── solver_yee_curvilinear.hpp                                          (M11 full，条件触发)

tools/    + boundary_reflection.cu  chirping_1d.cu
decks/    + chirping_1d.ini  rep_study_*.ini  chirping_2d_*.ini
scripts/  + budgeter.py  dispersion_theory.py  gen_dipole_tables.py
            gen_drift_tables.py  plot_chirping.py  campaign.py
docs/     + BOUNDARY_STUDY  WEIGHT_PRECISION  HOT_REP_ROUND1
            SCALING_AND_SIMILARITY  REPRESENTATION_STUDY  PROFILE_*
tests/    + ~20 项新 ctest（Part IV 各节列出）
```

## VII.2 工作量估计（1 名熟练 CUDA+PIC 开发者全职）

| 里程碑 | 难度 | 工作量 | 关键难点 |
|---|---|---|---|
| M0 | ★ | 0.5 月 | 纪律问题；CMake/deck/归一化已就位 |
| M1 | ★★★ | 1.5–2 月 | Esirkepov GPU 化；MaxwellSimulation 与既有体系并列 |
| M2 | ★★★ | 1–1.5 月 | 等离子体内吸收层非教科书问题 |
| M3 | ★★★★ | 1.5–2 月 | EM δf 权重方程；平衡保持测试极易暴露隐藏 bug；双表示一致性 |
| M4 ⛔ | ★★★★ | 1–2 月 | 复现是 gate；有 An et al. 复现经验垫底 |
| M5 | ★★★★ | 1.5–2 月 | **概念最难之一**：f0(ℰ,μ) 装载不对则之后一切测试被污染 |
| M5.5 ⛔ | ★★ | 0.5 月 | 推导 + 方法论 |
| M6 | ★★ | 0.75 月 | double-counting 与位移入沉积 |
| M7 | ★★★★ | 1.5–2 月 | 通量采样 + chunk pool + 长稳，工程繁琐 |
| M8 | ★★★ | 1 月 | 运行与分析 |
| M9 | ★★★ | 1–2 月 | 上限由 profiling 决定，禁止提前过度设计 |
| M10 | ★★★★ | 2–3 月 | 收敛性论证、排除数值伪像 |
| M11 weak | ★★ | 0.5–0.75 月 | full 版另计 2–4 月，条件触发 |

**到首个自洽 2D chirping（M10）：约 13–18 人月**；含 M11 weak 与 M12 生产 campaign：约 1.5–2.5 人年；full metric（若触发）另加 2–4 个月。若兼职（50% 时间）日历时间 ×2。

## VII.3 执行纪律

1. `v4-gate-1d` 之前不写任何 MLT injection 代码；
2. 任何生产 run 启动前必须通过显存/时间预算器，并记录 $S_d$、$\epsilon_L$；
3. fluid core 与 full metric 只由数据触发，不由兴趣触发；
4. 既有 12 项 ctest（ES/Darwin/守恒/Weibel gate）在整个改造过程中永远保持全绿；
5. 每个 PR 必须带测试；每项 M9 优化必须有 before/after Nsight 数据；
6. 每季度对照 VII.2 表记账一次，偏差 >30% 时重排优先级。

## VII.4 前两周行动清单

1. 核对 §10.2 四项待确认（deposit 完成度、两个 open PR、float 硬编码清单、migrate 现状），结论回写本文档；
2. 打 tag `v0-spectral-baseline`，ctest 加 label；
3. **补 `test_magnetized_boris`**（半周，解锁全部磁化物理）；
4. `scripts/budgeter.py` + checkpoint schema / HDF5 起步；
5. 写 `solver_yee.hpp` 与 `simulation_maxwell.hpp` 的空壳 header（接口先行）；
6. Nsight 基线 profile 存档（`deposit_bench` + `run_deck` 两入口）。

## VII.5 成功定义

**Physics**：正确的固定 L-shell 背景几何；kinetic warm/hot（core 表示由 M8 数据决定）电子响应；物理来源明确、$S_d$ 显式的 gradient-curvature drift closure；自洽 whistler 增长；nonlinear phase trapping；收敛的 chirping；可物理解释的注入与流出；**1D gate 复现文献结果**。

**Numerics**：charge-conserving current deposition（含漂移位移）；受控 $\nabla\cdot\mathbf B$；等离子体条件下验证过的吸收边界；稳定的 δf 权重（精度模式有数据支撑）；equilibrium preservation；可核算的能量；marker-number convergence。

**Performance**：RTX 5090 满载；tile-local field reuse；粒子状态每步接近单次读写；增量迁移；端到端 ≥3×10⁸ particle-step/s；2–4×10⁸ markers 舒适运行；生产 run 2–8 天/次；checkpoint/restart 可靠。

**Scientific contribution**：项目最终形成一个新的模拟能力：

$$\boxed{\text{fixed-}L\text{-shell }(s,\phi)\text{ electromagnetic kinetic PIC}}$$

它能够直接连接：

$$\boxed{\text{时变方位注入边界}\ (\text{经相似性缩放连接全局漂移})\ \rightarrow\ \text{whistler growth}\ \rightarrow\ \text{nonlinear chirping}}$$

这是 ArcWarden 从单 GPU 谱方法 PIC 原型发展为可信的磁层 wave-particle interaction research code 的核心路线——不是通过一次性押注宏大架构，而是通过两个物理 gate、数据裁决的表示方法、显式的尺度化框架，以及对现有工程体系（编译期 policy、Views 模式、deck-as-data、一测试一可执行）的增量演化，把每一个关键假设都变成有实验裁决点的问题。

---

*文档结束。下一步最值得单独撰写的配套文档：`docs/MODEL_EQUATIONS.md`（fixed-L 曲面、δf 权重方程、prescribed drift、Maxwell 方程与守恒电流沉积的统一严格推导，含 FullF 扣 J₀ 的完整公式）与 `docs/SCALING_AND_SIMILARITY.md`（M5.5 deliverable）。*
