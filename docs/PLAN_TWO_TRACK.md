# 双轨计划 v2(2026-08-06)— 屏障证明与两带后果分离

v1 经用户第二轮评审修正(两处会改实验设计的硬伤 + 六处执行修正),本版为可执行版。
v1 的两处错误已由数据/代码复核坐实:

- **D120 k⊥=0.16 的 m=1 并非 slaved**:element 窗 [500,2500] rms 复算
  R_m = 2B_{m=1,rms}/B_{m=0,rms} = **0.97(k16 平价)/ 0.44(k24)/ 0.19(k32)**。
  v1 "平价是强引擎伪影" 论断作废(当时引用的 2.1e-3 是 burst 起始瞬时值)。
- **天线 `amp` 是外加电流幅度,不是 δB/B0**(config.hpp M2/M10 注释;tao17 deck
  已有校准:`amp=1e-4 → δB_trig/B0≈1e-3`,2026-07-18)。

## 0. 源关卡(两轨通用,替代旧"§1 通用"引用)

单连通 rising ridge(非 broadband);Δω ≥ 0.15;**ω_max ≥ 0.55**;
Bw/B0 ∈ 3e-3–1e-2;0.46–0.56 无天然谷。**Track B 额外**:上肩/噪声 ≥ 10
(element-top 窗测)。3 种子 control 全过才配 RSM。

## 轨道结构

- **Track 0(discovery case,保留不废弃)**:G1 —— ctrl 0.213→0.511,RSM stop
  0.491,选择性抑制成立;受限于 1 个频率 bin、种子方差、m1 平价
  (docs/G1_PAIR_VERDICT.md)。论文叙事 = G1 发现候选屏障 → A 受控证明 → B 两带后果。
- **Track A(主科学线)**:triggered 单组分 barrier proof(D120 基底)。
- **Track B(后果线)**:G2 独立 upper-band feeder → lower + gap + upper。

## Track A 设计(修正版)

### k⊥ 角色分配(由 R_m 数据决定,不再从 0.16 起步等失败)

| k⊥ | R_m(D120) | 角色 |
|---|---|---|
| 0.24 | 0.44 | **主 barrier case** |
| 0.32 | 0.19 | weak-channel confirmation |
| 0.16 | 0.97 | strong-competition comparison(对照,非主证) |

### 天线(全部 knob 显式,不许只改 --amp)

```ini
[antenna]
amp   = <由校准定>   # 电流幅度;先验 Tao 标度: 5e-5 / 1e-4 -> δB/B0 ~ 5e-4 / 1e-3
x0    = 9696         # 赤道 (nx/2)
sigma = 4            # cells
w0    = 0.05         # 代码单位 = 0.25 Ωe (wce=0.2)
trmp  = <数百/wpe>
toff  = <element 到 0.4Ωe 之前必须关断>
```

- **A-cal(先行短跑)**:D120 几何线性校准,以**测得的 δB_trig/B0** 为准选 amp,
  目标 δB/B0 = {5e-4, 1e-3}(≪ element 5e-3–1e-2,只定起点/相位)。
  **测量位置纪律**:不在 antenna cell 上测(近场污染);天线关断后,在赤道两侧
  近源探针测向外传播包的峰值;验证线性 δB_trig ∝ ant_amp;
  packet 离开赤道源区之后才允许开始 nonlinear element 分析。
- 约束:element 过 0.4Ωe 前天线已关且 packet 已离开赤道源区(否则持续改写
  cyclotron 分布,破坏"自洽终止"归因)。

### 阶梯(算力修正版;t10000≈4.5h,pilot 短跑)

| 步 | 内容 | 长度/判据 |
|---|---|---|
| A0 | 诊断接线(见下) | 写码 |
| A-cal | 天线线性校准 | 极短;定 amp |
| A1 | pilot ×2 幅度,ctrl only | **t3500–4000 即可**(D120 element 窗 800–2500);任一幅度 ridge 清楚跨 0.55 → 停 pilot;两幅度都不行 → 停,回 Track B |
| A2 | 中选幅度 ×3 种子 ctrl 全长 | 3/3 过源关卡;trigger 应钉死 onset(种子间 ≲200/Ωe) |
| A3 | **3 个 paired RSM 全部预承诺运行**(k⊥=0.24) | 见预承诺规则 |
| A4 | k⊥=0.32 确认 + k⊥=0.16 对照 | 至少各 1 种子 |

**预承诺规则(反 optional-stopping)**:3/3 ctrl 过关后,3 个 RSM 无条件全跑;
第一组只查数值健康(NaN/坏输出),不决定其余两组是否运行。

## 指标(A/B 拆分,不再共用 S)

- **Track A:S_A = R_barrier / R_LB**(R_x = P_x^rsm/P_x^ctrl;barrier=[0.46,0.56],
  LB=[0.25,0.45])。判据:ctrl ridge 连续跨 0.55;RSM ridge 停在 0.47–0.50 且
  Δ = ω_max,ctrl − ω_stop,RSM ≥ 0.04;屏障以下 element 振幅保留(到达屏障前
  RSM/ctrl 包络 ≤2×);**S_A < 0.5;不引用 P_up**(G1 教训:上带=噪声时
  R_up 无意义,S_B 会误判)。**窗口纪律**:trigger 钉住 onset,但 RSM 仍可能改
  chirp rate,故 S_A 同时按三种窗报告——(a) onset 对齐固定时间窗;
  (b) **ridge-phase 对齐窗**(各臂各自从 ω=0.25 爬到 0.45 的区间);
  (c) 完整 element 窗。三者一致才算干净。
- **Track B:S_B = R_gap/√(R_lo·R_up)** + 上肩 up/hi-noise ≥ 10(两肩真实存在
  才有资格用双肩指标)。
- **R_m 语义规则**(记号固定 R_m = 2B_{m=1,rms}/B_{m=0,rms},与背景 B0 无关):
  element 期间 R_m ≳ 1 → 只能主张"强 oblique-mode competition 终止 element";
  R_m ≤ 0.3–0.5 才可主张"弱 E∥ 通道屏障"。不许人为限幅 m1。

### ω_stop 测量纪律(nwin=2048 不是免费升级)

2048 点窗 = Δt_win·Ωe ≈ 614,会把快扫频 ridge 涂宽并偏移终止时刻。规程:
同一结果用 **nwin = {1024, 1536, 2048} 收敛检查 + 峰值频率插值**;
仅当 ω_stop 窗间波动 < 0.01Ωe 才引用;硬门槛 Δ≥0.04 保留。

## A0 诊断接线(生产 run 必备)

1. **hot/cold J1·E1 分离 + 时间居中(必须,剩余最重要的技术门槛)**:现 rsm 路径
   step 后 j1 = hot deposit + cold twin 之和,总 PJ1xE1x 混入冷流体反应性振荡。
   输出三条:P_L1,hot、P_L1,cold、P_L1,total(deposit 后、cold twin 加入前
   快照 j1);离散功必须时间居中:**P^{n+1/2} = J^{n+1/2}·(E^n+E^{n+1})/2**
   (否则反应性功与净 Landau 功因时间错位混合);全部**逐步累积、按 dump
   区间平均**。**验收 = 能量闭合测试**:ΔW1 + ∫P_J1E1 dt + boundary flux ≈ 0,
   且 P_hot + P_cold = P_total(逐 dump 区间检查)。
2. **fvline**:赤道窗 **|i − i_eq| < 150 cells**(i_eq = nx/2;不是代码坐标 x<150),
   w-加权 f(v∥) 每 2000 步;粗节奏 (v∥,v⊥) 2D;后处理输出 **plateau 斜率及其
   统计误差**,不只做直方图目视。
3. **R_m(t)**:m1line 已有,脚本直算。
4. **m=0 cyclotron work 按 v∥ 分箱:不可选**。至少在每臂 1 个代表性机制 run 开启。
   论文核心链条 P_L1,hot → f(v∥) plateau → P_C0,hot 由驱动转弱/阻尼 → chirp stop
   缺了它就只剩时间相关性,审稿人可解释成一般模式竞争。
5. **A0 non-regression gate(接线后、生产前)**:诊断关闭 → 旧 simulation path
   不变(statistical-envelope 复核,同 regress_case2 方法);诊断开启 → 场演化
   不被改变、hot/cold/total 账本闭合、开销可接受;paired production 全部使用
   同一 clean build、同一 executable hash(BUILD_PROVENANCE 记录)。

## Track B:G2 现状与规程

- expG2_ctrl_r2 于 t≈153(51 blines,无 ckpt)被停,数据保留;**明日用户重启**,
  用全新目录:
  `cd build && ./chirp2d ../decks/giant_x4_expG2_ctrl.ini expG2_ctrl_r3 --ckpt=100000 --noeline`
- 关卡:§0 源关卡全套(含 Track B 附加的上肩 ≥10×);过 → s2/s3 ctrl;
  3/3 → 是否完成全部双频带配对**最后再决定**(执行顺序第 10 条)。
- 风险预告不变:pancake 也喂 oblique 增长,G2-RSM 的 m1 平价可能比 G1 更糟。

## 修正后的执行顺序

1. G1 保留为 Track 0 discovery case;
2. 明日重启 G2 s1(expG2_ctrl_r3);
3. A0:先做 hot/cold J1E1 分离,再 fvline/账本;
4. A-cal:D120 天线短校准,按测得 δB_trig 选幅度;
5. A1 pilot 只跑 t3500–4000;
6. 中选源 3 个 control 全长;
7. Track A 主打 k⊥=0.24,0.32 弱通道验证,0.16 强竞争对照;
8. 3 个 paired RSM 预承诺全跑;
9. A 用 S_A,B 用 S_B;
10. 最后再决定 G2 是否值得完成全部双频带配对。

---

## 执行状态(2026-08-06)

- **A0 全部完成**:
  - ①hot/cold J1·E1 时间居中账本 = 97f1530(闭合 resid 中位 9e-4,
    off-path WB ratio 1.000);
  - ②fvline + ④m=0 回旋功 = f9ed880:chirp2d `--fvdiag`,赤道窗
    |i−i_eq|<150 cells 的 f(v∥)(600 bins/2000 步)+ 粗 (v∥,v⊥)
    (10000 步);m=0 J·E Landau/cyclotron 按 (16 x-区, 240 v∥-bin),
    每 20 步采样(≥21 采样/周期到 0.5 wce)、2000 步 dump+reset;
    诊断级(非闭合账本,闭合归 m1ledger);
  - ⑤非回归:开/关同 seed WE/WB 包络比 1.000000/1.000005(rerun
    噪声级);k_workledger 实测 4× push(double-atomic 瓶颈)→
    wl_acc=20 把机制 run 开销压到 ~13–19%,仅机制 run 打开。
- **A-cal 完成(PASS)**:decks/giant_x4_acal_{a05,a10}.ini,
  scripts/acal_verdict.py,data build/acal_{a05,a10}。测量纪律:±145.2
  探针 post-off 波列通过窗 [x/vg+trmp+100, x/vg+toff](均 > toff,
  断言强制),exp(−iw0t) 解调 + 2 周期 boxcar(±0.025 带通);近场
  (赤道探针)只作 flagged 参考。结果:**amp 5e-5 → δB_trig/B0 =
  7.4e-4;amp 1e-4 → 1.46e-3**(探针几何均值);均值线性度偏差 0.8%
  PASS(逐探针 ±17% = 同 seed 热背景相干干涉系统误差,几何平均一阶
  消除);带内噪声底 ~4e-5。附带:波包对流放大 2e-4→6.8e-4(t2100/wpe)
  —— D120 介质在 0.25 We0 有增益。教训入档:解调旋向要用 ±w0 谱线
  实测定(z=By+iBz 载波在 +w0),通过窗要按发射区间 [trmp,toff]+x/vg
  推,不能按 toff+x/vg。
- **A1 pilot decks 就绪(未启动)**:decks/giant_x4_a1_{a05,a10}.ini
  (= acal + nsteps→133333 = t4000/Oe;toff=1000/wpe 早于一切 ridge
  活动,"off before 0.4" 由构造满足)。Runner:
  `./chirp2d ../decks/giant_x4_a1_a05.ini a1_a05 --ckpt=100000 --noeline --fvdiag`
  (a10 同)。两幅度预承诺都跑;GPU 让给用户先跑 G2 s1(expG2_ctrl_r3)。

## Track B 判定(2026-08-06):G2 s1 源关卡 FAIL

expG2_ctrl_r3(t10000/Ωe 完整,seed 1,data build/expG2_ctrl_r3):

- **ω_max = 0.491**(ridge 逐帧最大,@t5549;分位 50/90/99% =
  0.327/0.456/0.491)——从未过 0.5,bar ≥0.55 FAIL;与 G1 ctrl 系
  (0.511/0.470)同在种子方差包络内,pancake 没有帮 riser。
- **上肩不存在**:0.55–0.60 在所有窗([3000,4500]/[4500,6000]/
  [5664,8000]/[8000,10000]/burst 全窗)、所有探针(eq/±145/±291)
  ≤1.4× 噪声(bar ≥10×);0.60–0.95 扩展扫描同样为噪声底 —— pancake
  独立上带完全没有出现,element-top 窗检查排除时间稀释。
- 判定脚本的 "natural valley V=0.398" 为公式误报:up(9.4e-7)<
  noise(2.1e-6),V 的双肩假设不成立,真实形状是单调下降坡,不是谷
  (与 G1 "P_gap<P_up" 同类的噪声底陷阱,再次确认 S_B 需 up/hi≥10)。
- **失败机制(定量)**:引擎 A=0.52 → 边际频率 0.342,其上全部被引擎
  回旋阻尼;线性净增长率符号估算 @ω=0.6:pancake 项 +n2·η·[A(1−ω̃)−ω̃]
  = +0.0035·0.55·1.4 ≈ +0.0027,引擎项 −0.0178·0.95·0.39 ≈ −0.0066 →
  净负;ω=0.7 更糟(−0.0094 vs +0.0023)。结构性问题:5× 密度的引擎在
  pancake 全设计频段占主导阻尼,n2 需 ≳3× 或引擎各向异性需提高
  (ArmU 之所以有双带:单种群 A=1.5 → 边际 0.6,无自阻尼冲突)。
- **执行后果(按预承诺规则)**:s1 FAIL → 不跑 s2/s3、不配对;Track B
  搁置,重新设计权在用户(选项:n2≥0.01 / 引擎 A 提升即回到 ArmU 型
  单种群 / 放弃双带 consequence)。Track A 即刻转主线:A1 pilot 对
  (a05/a10)已于同日启动。
