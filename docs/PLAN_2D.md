# PLAN_2D v2 — Lu 尺度最大 full-2D mirror 旗舰(2026-08-08 用户修订版)

v1(c626f5f)的阶梯框架被用户修订取代:**第一主线 = 尽可能大的 full-2D
mirror simulation,t=0 起自洽 kinetic resonant electrons,不预置 plateau,
让 WNA、E∥、plateau、chirping、gap 在同一个长盒子里自然演化。**
1D/RSM 工作不再是本计划的参照系(仅 4 个 1D run 保留为可选对照数据)。

## 1. 物理核心(用户论证,预注册)

### 1.1 L=0.6 R_E 是 runway,不是普通尺寸参数
对流增长的 chorus 由积分增益决定,不是局域 γ:

    N_g(ω) = ∫ γ(ω,s)/v_g(ω,s) ds,   B_w(ω) ∝ exp(N_g)

Upper band 局域 γ 弱 → 需要更长传播距离才可见。短盒:N_g,LB > N_th 而
N_g,UB < N_th → 只见 lower band。0.6 R_E(→ Lx≈1331 c/ωpe,x10 映射)
可能翻转为 N_g,UB > N_th。TaRA 是沿传播方向的
trapping→release→new resonance→amplification **relay**,盒子太短则波包
在 relay 建立前进吸收层——这解释 Lu 长几何有跨带元素而短 2D 只有 LB。

### 1.2 电子模型(Annotation 1 修订)

    cold-fluid bulk + kinetic resonant population (t=0) + kinetic anisotropic chorus source

Resonant population 必须:参与电荷/电流沉积、对场反馈、在镜场真实
bounce、能形成 plateau/scar、不被 refresh 重置。**不是 test particles。**
覆盖速度区间:
- |v∥| ≈ 0.05–0.07c(upper-band cyclotron shell)
- |v∥| ≈ 0.087–0.10c(0.5Ωce Landau/相速区)
- |v∥| ≈ 0.15–0.26c(lower-band cyclotron shell)

**禁止**只在 0.09c 放窄 bump(人为制造 gap)。方案 = 平滑 product-kappa
+ 共振区 importance sampling,权重拆分:

    f0 = χ_r(v∥) f0 + [1−χ_r(v∥)] f0

两部分之和严格 = f0,增加的是共振区统计精度,不是新自由能。
实现:按速度空间分区——cold fluid 承载 |v∥|<v_cut 部分(含 kappa 核,
不参与共振,只出介电响应);kinetic 种群 = f0 限制在 |v∥|>v_cut
(平滑边沿 χ_r),权重精确归一。

### 1.3 因果序列(成功的机制指纹,预注册)

    E∥ → Δf(0.08–0.10c) → plateau → 0.5Ωce 功率下降

分布变化必须**先于** gap;gap 必须**随时间逐渐形成**(不是初始线性 γ
本来就在 0.5 最小);0.5 下降时 lower band 不能被整体压死。

## 2. 工程现状审计(2026-08-08,全部核对过代码)

**已有**:
- `decks/mirror2d_ke17.ini` = 被 G2.2 gate 验证的 2D mirror 生产配置:
  2048×512,b0_prof=3,**cold_model=full(2D 冷流体已在生产验证)**,
  4.2e8 hot markers **实测 ~21 GB**,dt=0.1,x=damping;
- y-wrap 已实证survivable:ke17 的 wrap 跳变(2aB0·x̃·Ly ≈ 2.4 B0)比
  旗舰盒(≈1.2 B0)更极端,straddler 层 ~0.6% 粒子、弹性(deck 注释);
- `dist=prodkappa`(dist=3,kappa_par,X4_PRODKAPPA_PLAN)+ 多 species
  加载(G1.3)+ per-marker weight `w`(Particles 结构)——χ_r 拆分的
  全部原料在位;
- G1.3 诊断包(fv|region、J·E Landau-vs-cyclotron ledger、WNA);
- Particles 内存模型:主数组+tile-sort 双缓冲+bin ≈ 60 B/marker →
  4.2e8 ≈ 25 GB 上限;fields 2.1e6 cells 可忽略。

**缺口(P0 工程件,~2–3 天)**:
1. **v∥-shell importance loader**:per-species 平滑速度窗(accept-reject
   + 权重归一 + raised-cosine 边沿),quiet-start 兼容;
2. **mirror2d 驱动升级到生产级诊断**:probe 行(y-mid)、bline 双份
   (y-mid 行 + y 平均)、**kyspec**(eq/±5°/±7.5° x-站的 B(x,ky,t)
   复谱)、checkpoint/resume;
3. **x=hybrid**(吸波+反射粒子,Lu 边界)接入 mirror2d 路径
   (ke17 用的是 damping;粒子边界行为要审计);
4. 旗舰 deck(§3)。

## 3. 旗舰 deck 候选(v0;P0 冒烟后定案)

| 项 | 值 | 备注 |
|---|---|---|
| 网格 | **4096 × 512**,dx=0.325,dy=0.699 | Lx=1331(=0.6 R_E 映射),Ly=358;dky=0.0176 连续 k_y 谱 |
| 镜场 | b0_prof=3,a=2.54e-6 | = x10 lre=1330.5 的抛物映射;端部镜比 ~2.1;边缘场线倾角 ~16–25° |
| dt | 0.15(CFL 上限 0.295) | t6000 = 200k steps |
| 冷体 | cold_model=full,cold_nc ≈ 0.978 | 含 kappa 核的 (1−χ_r) 部分 |
| chorus source | bimax,n_h=0.010,u∥=0.13,**A=4.0–4.5**,ppc 120–140 | 用户:2D 阈值更低,先别用 A=6/Lu 全强度防 flood;≈ke17 的 A=4.5 |
| resonant pool | **prodkappa**,n_r ≈ 0.012,shell |v∥|∈[0.04, 0.30]c 平滑边,ppc 60–80 | 覆盖三个共振区;能量递减各向异性;t=0 起全动理 |
| 总粒子 | ~4.2e8(总 ppc ≈ 200) | ke17 实测锚点 ~21–25 GB;**冒烟 gate 定案** |
| 边界 | x=hybrid,y periodic | Lu:吸波+反射粒子;吸收层按物理长度定 |
| seed | 20260720 | |
| 时长 | 阶段化(§4);完整阶段 **t≥12000/Ωe** | shell 电子 transit ~5300/Ωe(v∥=0.1c 过 Lx)→"多次 bounce"需 ≥2 个 transit |

速率估算:8.4e13 p-steps(t6000)@ 5e9–1e10 p-steps/s → 2.5–5 h;
t12000 ≈ 5–10 h = 一晚。数值 gate:ρ⊥/dy ≈ 2.0(A=4.5 时)= G1.2 规则
下限,记录;resonant shell ρ⊥ 更小 → 其 v⊥ 动力学主要靠 x 向解析,列入
风险登记。

## 4. 运行顺序(checkpoint gates,用户版)

1. **内存与静平衡 gate**(几百步):显存实测、∇·B、加载正确性
   (shell 权重和 = ∫χ_r f0)、初始噪声地板;
2. **线性阶段**(→ Ωce·t ~400):2D WNA、增长率、**upper-band linear
   fuel 存在性**(ke17 目标:onset ~400);
3. **第一元素阶段**(→ t ~1000–1500):清楚 rising element,不是
   broadband flood(W10 类度量,2D 重校准后判);flood → 停,降 A/n_h
   一档重来(唯一允许的参数动作);
4. **累积阶段**(→ t ~3000);
5. **完整 gap 阶段**(→ t ~6000–12000+):resonant electrons 完成多次
   bounce/处理循环。

每关出报告再续跑;ckpt 密度保证任何关卡可回溯。

## 5. 成功判据(全部同时满足,预注册,看谱前冻结)

1. lower band 有可追踪的**离散** rising elements;
2. upper band 功率不是噪声或谐波;
3. 连续 k_y 谱出现真实 finite-WNA power;
4. E∥ 对共振电子的**累计做功**清楚(ledger);
5. 0.08–0.10c 分布变化**先于** gap;
6. gap **随时间逐渐形成**(非初始线性 γ 谷);
7. 0.5 下降时 lower band 未被整体压死。

## 6. L=0.6 R_E 归因(discovery 成功后的论文义务)

长度扫描 L = 0.2 / 0.4 / 0.6 R_E,保持:Δx、Δy 不变;横向宽度不变;
初始 f0 不变;局域增长率不变;**吸收层按物理距离固定**。检验:

    ln B_UB ∝ ∫ γ_UB / v_g,UB ds

若只有 0.6 R_E 形成 upper band / 完整 gap →"既往 2D 失败 = runway 不足"
成为强新结果。

## 7. 禁做

不预置 plateau;不放 0.09c 窄 bump;resonant pool 不被 refresh 重置;
不跑 RSM;stage-3 flood 之外不做参数轮盘;第一版 A ≤ 4.5;
判据 §5 冻结前不看完整谱定标准。

## 8. 立即下一步

P0 工程 4 件(§2 缺口)→ 内存/速率冒烟实测报告(stage-1 gate)→
用户确认 deck 定案 → stage-2 起跑。
