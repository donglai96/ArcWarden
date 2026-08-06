# 双轨计划(2026-08-06)— 屏障证明与两带后果分离

前提 = 用户 08-05 深夜评审(全文要点收录于本文),核心裁决:**不再要求一个 G2 run
同时解决 source / barrier / visible gap 三个问题**。拆成:

- **Track A(主科学线)**:triggered 单组分 barrier proof —— control element 稳定爬过
  0.55,RSM 同款 element 停在 0.47–0.50;
- **Track B(后果线)**:G2 独立 upper-band feeder —— 屏障已证后,展示它如何把
  连续源整形成 lower band + gap + upper band。

我(Claude)对评审的核实与增量意见在 §5。

## 1. 共同的硬性标准(两轨通用)

- **源关卡**:单连通 ridge;Δω≥0.15;**ω_max ≥ 0.55**(0.53 弃用——离目标只有
  0.03,会被 STFT 分辨率和自然停止方差淹没);0.55–0.60 上肩 ≥10× 噪声(仅 Track B
  要求;Track A 不要求上肩——"拦住 vs 看见"矛盾由分轨化解);无 0.46–0.56 天然谷;
  Bw/B0 ∈ 3e-3–1e-2。
- **种子关卡**:**3 个 control 种子全过**才配 RSM(G1 教训:1 种子配对 → 不可解释)。
- **RSM 成功判据**:ω_max,ctrl ≥ 0.55 且 0.47 ≤ ω_stop,RSM ≤ 0.50 且
  **Δ = ω_max,ctrl − ω_stop,RSM ≥ 0.04**(≥2 个现 bin;判决脚本升级 nwin=2048 后
  = 4 bin);S ≤ 0.5;lower-band element 保留;**到达屏障前 RSM/ctrl 包络差 ≤2×**;
  全程记录 R_m(t) = 2B1_rms/B0_rms,理想 R_m ≤ 0.3–0.5。
- **R_m ≥ 1(平价)语义规则**:结果仍可发表,但只能称"强 oblique-mode competition
  终止 element",不得称"弱 E∥ 通道屏障"。不许人为限幅 m1;对策是物理拨盘:
  k⊥ = 0.24(D120 扫描的选择性最优点)、0.32。

## 2. Track A:triggered 单组分屏障实验

设计(用户评审 §第二阶段,经代码核实可零天线开发落地):

- 基底 = **D120**(nh=0.012 单组分、dipole+atmo、full-f)—— 不是 G 引擎。理由:
  G1 的 m1 疯长发生在 nh=0.0208 强引擎上;D120 的 B1 历史是 slaved(2e-3,无平价),
  竞争问题很可能本身就是强引擎伪影(§5.3)。
- 天线:chirp2d 现成 M2/M10 触发天线(旋转横向电流柱,R-螺旋 whistler 包)。
  参数起点:ant_w0 = 0.25 Ωe,amp 阶梯 {5e-4, 1e-3}·B0(≪ element 的 5e-3–1e-2,
  只定起点与相位,不造停止),ramp + toff ≈ 数个 trapping 周期,赤道注入。
- 两臂 deck 完全相同(天线只驱动 m0;m1 仍由噪声自发)。

阶梯与算力(每个 t10000 ≈ 4.5h):

| 步 | 内容 | 判据 |
|---|---|---|
| A0 | 诊断接线(写码,GPU 空闲期) | §4 |
| A1 | 天线 pilot:D120+trig 两个振幅,ctrl only | 任一振幅 ω_max≥0.55 且 trig 幅 ≤0.2× element 峰;两振幅都不行 → **停,回 Track B** |
| A2 | 中选振幅 ×3 种子 ctrl | 3/3 过关卡;trigger 应把 onset 钉死(种子间 ≲200/Ωe) |
| A3 | 配对 RSM(先 seed-1,过则补 2) | §1 RSM 判据 + §4 因果链 |
| A4 | 若 R_m 平价重演 | k⊥=0.24 重跑 A3(再不行 0.32;仍不行 → 语义规则收场) |

Trigger 的第三个红利(§5.4):钉死 t0 后,多种子时间序列可以对齐叠加,
因果链变成 ensemble 平均证据。

## 3. Track B:G2 两带后果实验

- **正在跑**:expG2_ctrl_r2(seed 20260720,t10000)。
- 关卡 = §1 全套(含上肩 10×);过 → seed 2、3 control;3/3 过 → G2 RSM 对。
- 角色定位:屏障已证后的整形演示。**G2-RSM 上带若消失不推翻 barrier**,只说明
  hot2 还不够独立。
- 风险预告(§5.5):pancake 把 0.5 处 γ 提高 69% 的同时也加强 oblique 驱动,
  G2-RSM 的 m1 平价可能比 G1 更糟 —— 所以屏障证明必须由 Track A 承担。

## 4. A0 诊断接线(最终生产 run 的必备输出)

chirp2d 增补(gap_diags.hpp 已有大部分实现,接入为主):

1. **逐步累积的 J1·E1 账本**:PJ1E1 / PJ1xE1x 每步累加、按 dump 区间平均输出
   (根除 2000-步采样对 ~30/Ωe 波周期的混叠 —— 现有 caveat);
2. **fvline 检测器**:赤道窗(|x|<150 cell)w-加权 f(v∥) 每 2000 步一帧
   (时间分辨,不再只有末态);粗节奏加 (v∥,v⊥) 2D 直方;
3. **R_m(t)**:m1line 已有,验证脚本直接算;
4. m=0 cyclotron work 按 v∥ 分箱:重内核改动,**A3 前评估**;若太贵,用
   f(v∥,t) 变平时刻 vs m0 包络衰减时刻的先后顺序替代(因果链仍可读)。

要读出的因果顺序(比任何一张 gap 图都重要):
E1∥↑ → P_L1>0 → f 在 0.087–0.10c 变平 → m0 cyclotron drive 熄灭/转阻尼 → chirp 停在 0.5。

## 5. 对评审的核实与增量(Claude)

1. **核实**:tao17_ctrl2 / tao17_ctrl_s2 数据在 build/(0.583 双种子跨越属实,
   另有 tao17_rsm1/rsm_s2、tao17_ctrl_nan_t11000 遗留);gap_diags.hpp 存在未接线;
   chirp2d 天线完整(config.hpp M2/M10,deck [antenna],`--amp=` CLI)。
2. **ω_stop 精度**:现判决链 bin = 0.0204 Ωe(nwin=1024,rec 1.5/wpe)。
   升级 nwin=2048(bin 0.0102)+ ridge 末端中值;Δ≥0.04 判据即 4 bin。
3. **m1 平价可能是强引擎伪影**:G1(nh 总 0.0208)m1 到平价 2.9e-2;D120(0.012)
   B1 只到 2.1e-3 且 slaved。Track A 用 D120 基底,R_m 天然可控的先验不低 ——
   这是 A 优于"在 G 引擎上修竞争"的核心理由。
4. **trigger 修的不只是 seed 方差**:G1-RSM burst 提前 ~2000/Ωe(m1 噪声提前点火),
   固定窗比较随之失效;trigger 钉死 t0 同时修复窗口可比性 + 允许 ensemble 对齐。
5. **G2 二号风险**:pancake 各向异性同样喂 oblique 增长,G2-RSM 竞争可能更糟。
6. **论文**:同意 paper/main.tex(GPU 方法文)不装 gap;物理文五主图按评审;
   RSM_MODEL_DEFINITION 的 novelty 表述正式写作前收紧为
   "首次在镜场 chirping 系统中受控证明自洽 oblique-mode feedback 终止 lower-band element"。

## 6. 执行顺序(即时)

1. ✅ expG2_ctrl_r2 已启动(Track B 关卡 s1,~4.5h);
2. GPU 忙期:A0 写码(账本+fvline+脚本 nwin=2048)+ A1 deck 草拟(d120+trig ×2 振幅);
3. G2 s1 出结果:过 → 排 B 的 s2/s3 与 A1 交替占 GPU;不过 → GPU 全给 Track A;
4. 每一步判决入 docs,种子关卡不许跳。
