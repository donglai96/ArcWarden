# PLAN_UB_1D v2 — 上带激发:文献批判 + 三步简化计划(2026-08-11,用户审查后重写)

v1(3b4c9e9)被用户裁定过于复杂;本版按裁决收敛:以 giant_x4_atmo40
(full-f,t20000,已在盘)为中心,先想清 "为什么我们做不出 UB / 为什么
现实的几百 eV 各向异性不够",文献过堂,然后三步。

---

## 1. 文献批判性对账(2026-08-11 检索)

### 1.1 候选机制与证据状态

| 机制 | 代表文献 | 状态 | 弱点 |
|---|---|---|---|
| A. plateau 切分 / 两成分 | Li 2019 NatComm(docs 有);Li 2022 GRL(gap 在源区形成,与平行加速关联);Chen 2022 JGR / Chen 2023 GRL(plateau-like 形状→gap) | 观测关联强:两带事件伴随 0.05–2 keV 与 >10 keV 双各向异性成分 + ~2 keV 平行 plateau | **关联≠因果充分**。① plateau 切分需要 E∥(斜传播)才能自生;② 20 事件双带统计里 **γ_UB 峰值 < 1e-5 Ωe** —— 从噪声长到观测幅度需 ~1e6/Ωe,本地线性激发实际不工作;③ 用户论点:统计上几百 eV 的 A 常在阈值下 3–5 倍 |
| B. 下带级联(2ω 谐波) | Gao 2016 GRL / 理论 AIP Adv 2018 / **Chen 2017 JGR 1D PIC**(平行 1D 里复现!) | **唯一在 1D 平行 PIC 里演示过的 UB 生成机制**;LB + 有质动力密度模 (ω,2k) → UB@2ω_LB | 硬预言 UB≈2×LB:只有一小部分 multiband 事件满足;一般双带的 UB 不在 2×LB → 只解释子类 |
| C. 远程/多源传播 | Tao 2023 GRL(LB 本地源 + UB 远程源,密度脊 ducting,无需高纬反射) | 与 Poynting 观测(赤道源、向两半球传播)相容;解释 UB 弱于 LB | 把"UB 哪来"推给远程源——远程处仍需生成机制;径向结构我们的 1D 沿场箱不可测 |
| D. 两族群注入 | Fu 2014;Zhou 2019 GRL(loss cone + 各向异性 → 斜 LB/UB) | 注入期低能各向异性可短暂超阈 | 统计弱(用户论点);Li 2019 事件实测有,但常态没有 |

### 1.2 观测硬约束(任何机制都要过)

- Gao 2019 GRL:2/3 chorus 事件有 gap;gap 频率峰 **0.49 fce**、宽 ~0.07 fce;
- Meredith 2009:**UB 局限于近赤道 |λ|≲15°** → 高纬生成类机制不利
  (与 Li 2022 "gap 在源区形成"、Tao 2023 "无需高纬反射"一致);
- 双带各自含离散升调元素(Li 2019 Fig 1c)→ 纯线性带状增长不够,UB
  也要能非线性组织成元素。

### 1.3 对我们旧结果的解释(为什么从来没出过 UB)

1. **bimax 的 A 与能量无关**:我们全部 1D 源是单 bimax/conecut → UB 共振
   电子(ωpe/Ωe=5 下 0.7–1.7 keV,即 v_R≈0.05–0.08c)携带与全体相同的
   A=0.5–0.8 → ω_m=0.33–0.44 < 0.55 → **UB γ<0 按构造**;从未有过独立
   低能各向异性成分;
2. **引擎种群是 UB 净吸收体**:A < ω/(Ωe−ω) 的种群在 UB 频段贡献负 γ
   → 低能成分要打赢"裸 KP 阈值 + keV 引擎回旋阻尼 + 冷成分阻尼"。
   **G2 失败(08-06)就是实测版**:pancake 的 UB 带被引擎吃掉 γ_net<0;
3. **1D 无 E∥**:机制 A 的 plateau 自生通道不存在 → 两成分结构只能手放;
4. **γ_UB 即便为正也小**(观测 ~1e-5–5e-4):从噪声长起需 1e4–1e6/Ωe
   或长 runway;full-f 噪声/时长都不够(δf 已解决可见性,备用)。

**"现实几百 eV A 不够"的定量含义**:裸 KP 阈值只是低能成分自身贡献的
零点;净点火阈值(含阻尼项)显著更高 → UB 事件应对应注入时刻的瞬时
超阈(Li 事件属此)或非线性通道(B)。这个净阈值曲线我们today可算。

---

## 2. 三步计划(简化版)

### Step 1 — 净阈值曲线(色散求解器,无 GPU,先行)

扩 `whistler_kinetic_dispersion.py` 到双热种群:引擎(giant_x4 参数
bimax 等效 u∥=0.198, u⊥=0.244, nh=0.0178)+ 冷 + 低能种群
(u∥₂≈0.045 ≈ 500 eV;扫 n₂ × A₂)→ 输出 **γ_UB(0.55–0.75)>0 的
净阈值曲线 (n₂, A₂)**,并标注:裸 KP 阈值线、Li 2019 Fig 2a 量级的
观测点、G2 失败点(应落在阈下,作为回溯验证)。
**产出即结果**:定量回答"观测的几百 eV A 为什么不够/什么时候够"。

### Step 2 — giant_x4_atmo40 挖掘(零成本,判据先冻结)

最强停滞元素列车(amp ~7e-3,t20000,ckpt 在盘)上:
1. **P(ω>0.5) 上限**(全程、分站);
2. **相位相干 2ω 搜索**(沿元素 ridge 的 bicoherence)——机制 B 在
   1D 可行(Chen 2017),我们的列车振幅与其 PIC 相当:**该出必须出**;
   不出 → 给级联效率定上限;预期谐波幅度 ~(δB/B0)×δB ≈ 2.5e-4,
   在 full-f 1D 地板边缘 → 用 ridge 锁相平均压噪声;
3. ckpt 提 f(v∥,v⊥) UB 壳层(0.05–0.08c)前后对比:LB 回旋处理(作用
   区 0.15–0.26c)是否触及低能段(预期否;把"1D 回旋版两级"正式关闭)。

### Step 3 — 一个新 run:净阈值上方的两族群(G2 重做)

giant_x4 base + 第二低能各向异性种群,**(n₂, A₂) 由 Step 1 曲线选在
净阈值刚上方**("注入时刻"情景)。问题:LB 元素列车与 UB 带共存?
0.5 分隔自然出现(1D 无 E∥ 的两源两带)?UB 内是否组织出离散元素
(观测约束 §1.2)?
成本:1D full-f ×4,ppc 28000,t10000 ≈ 1–2h。失败动作(预注册,一次):
γ_net 边缘 → n₂ 上调一档;flood → n₂ 降档。

## 3. 预注册判读

- Step 2 若见锁相 2ω → 机制 B 在我们数据成立,UB-无燃料路线有解;
- Step 2 全阴性 + Step 3 阳性 → 1D 里 UB 必须外源低能各向异性(注入
  情景),gap = 双源带间隔(H1 无记忆味);
- Step 3 阴性(γ_net>0 仍无 UB)→ 非线性/对流抑制,量化并转 2D 问题;
- 不在 flood/死寂基底上解读;不注入阈下假燃料后又声称"自然出 UB"。

## 4. 明确不做

不复现 Chen 2026 GRL;δf 臂降为备用(full-f giant_x4 线为主,按用户
裁决);机制 C 的径向结构 1D 不可测,如实出局声明。

## 5. 文献清单(本轮检索)

- Li et al. 2019 NatComm, Origin of two-band chorus(docs 在库)
- Li et al. 2022 GRL, Unraveling the Formation Region and Frequency of Chorus Spectral Gaps
- Li et al. 2024 JGR, Controlling Factors of Chorus Spectral Gaps
- Chen et al. 2022 JGR, Gap formation … plateau-like shape
- Chen (H.) et al. 2023 GRL, Unraveling the Role of Electron Plateau Distributions(obs)
- Gao et al. 2016 GRL, Generation of multiband chorus by lower band cascade
- Chen (L.) et al. 2017 JGR, Lower Band Cascade of Whistler Waves — 1D PIC
- Gao et al. 2018 AIP Adv, Theoretical analysis on lower band cascade
- Gao et al. 2019 GRL, Statistical Results of the Power Gap(2/3 事件,0.49/0.07 fce)
- Tao et al. 2023 GRL, Formation of Banded Chorus Waves by Propagation From Multiple Sources
- Zhou et al. 2019 GRL, Highly oblique LB/UB by loss cone + anisotropy
- Meredith et al. 2009 JGR, Survey of upper band chorus(UB 近赤道)
- Fu et al. 2014(两族群)
- Tao, Zonca & Chen 2025 PoP, What drives chorus frequency chirping(docs 在库,chirp 机制之争)
