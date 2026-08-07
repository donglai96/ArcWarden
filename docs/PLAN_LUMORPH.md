# Lu-morph 源计划(2026-08-06 用户口述,源阶段 source of truth)

背景:docs/WHY_NO_060_ELEMENTS.md 走廊分析。**停止 A20/A25 与 G2,不再堆
各向异性;唯一目标 = 先拿到稳定跨过 0.55Ωe 的 control source。**
本计划在源阶段取代 PLAN_TWO_TRACK.md v2.1 的执行顺序;RSM 判据(§四)沿用
两轨计划的 S_A/R_m/三窗纪律原文。

## 第一阶段:Lu-morph 源筛选

x4 镜场、密度、loss cone、数值参数全部不变,只改热电子速度分布。
不开 RSM、不开天线。**关键 = 降 u∥(关低频走廊),不是升 u⊥。**
目标:0.2–0.3 不易点火,元素出生推到 0.3–0.4,保留 0.55+ 增长带。

| Case | u∥ | A | u⊥ | deck |
|---|---|---|---|---|
| L12A5 | 0.12 | 5 | 0.294 | decks/giant_x4_lumorph_L12A5.ini |
| L10A6 | 0.10 | 6 | 0.265 | decks/giant_x4_lumorph_L10A6.ini |

共同:density 0.012 / cold_nc 0.988 / ppc 36000 / conecut / amp 0 /
seed 20260720 / t4000Ωe。

## 第二阶段:pilot 判据(全部同时满足)

1. 无 A20 式早期宽带爆发;
2. 一条清楚、连通的 rising element;
3. 起点 0.28–0.40;
4. 扫频宽度 ≥ 0.15;
5. 三种 STFT 窗(1024/1536/2048)下稳定跨 0.55,窗间差 < 0.01;
6. 赤道 Bw/B0 ∈ [3e-3, 1e-2];
7. 0.46–0.56 无天然谷;
8. 非孤立高频点、非宽带背景假 ridge。

停止规则:t2000 已明显宽带爆发 → 提前判失败;t4000 无清楚元素 → 失败;
单个短窗碰到 0.6 不算成功。

## 第三阶段:源的三种子验证

两候选取较干净的一个,seeds 20260720/21/22,全部 ≥ t6000Ωe。
**3/3 稳定跨 0.55 才算拿到 source**(A20 式 1/3 晚期 riser 不得进 RSM)。

若两个 x4 Lu-morph 都失败:不再随机扫 x4 参数,直接转 x10/Lu 几何
(u∥≈0.141, A=5, density≈0.01, lre≈1330)= "Lu source reproduction",
先确认代码能产生 0.55–0.65 元素,再讨论移回 x4。

## 第四阶段:RSM 配对(source 3/3 通过后才开始)

- control ×3 + RSM k⊥=0.24 ×3,同一代码、同一 executable hash,逐 seed
  配对;三个 RSM 全部预承诺完成。
- 成功标准:ctrl 清楚跨 0.55;RSM 停在 ~0.47–0.50;终止频率差 ≥ 0.04;
  0.25–0.45 下带元素仍在(不能整条压没);S_A < 0.5;R_m ≲ 0.3–0.5
  (否则只能解释为强模式竞争)。
- 随后:k⊥=0.32 弱 E∥ 通道确认;k⊥=0.16 强模式竞争对照。

## 第五阶段:机制证明

≥1 个 ctrl/RSM 配对开完整诊断(--fvdiag + m1ledger),检查时间顺序:
1. m=1 hot J∥E∥ 首先出现;
2. 赤道 f(v∥) 在 ~0.09–0.11c 变平;
3. m=0 cyclotron drive 随后减弱/转阻尼;
4. rising element 最后在 0.5 附近停止。
顺序成立才支持 E∥,m=1 → Landau plateau → m=0 drive 下降 → chirp stop。

## 当前不要做的事

不跑 A20-RSM;不继续 A25/更高 A;不升天线幅度;不重启 G2;
**不把 control 自己停在 0.5 解释为 RSM gap。**

---

# 2026-08-07 修订(用户三项决定;lurepro_x10 改判"单种子定性 PASS")

lurepro_x10 判定修正:旧追踪器早期锁低频带错过 t2300–2900 真 riser →
FAIL 撤销。第三阶段物理问题已回答:ArcWarden full-f 在 Lu 参数 + 强梯度下
能产生越过 0.5、达 0.65+ 的 element。距"稳定 source"还差三种子。

## 决定 1:双位置判据(不整体搬到离赤道)

- **赤道探针**:出生时间、初始频率、低频背景、洪泛判定;幅度只记录不设下限。
- **|λ|=5°(|s|=116.4)= 主判定位**:扫频 Δω、终止/最高频率、成熟波幅。
- **|λ|=7.5°(|s|=175.1)= 确认位**:传播放大与连续性。
- **传播方向预注册**:每 seed 先用低频波包 outward Poynting flux 定主方向;
  ctrl 固定后 paired RSM 用同方向同探针;禁止看完 RSM 再挑半球。
- **归一化修正(必须双报)**:ω/Ωe,eq 与 ω/Ωe(s);Bw/Beq 与 Bw/B0(s)。
  B0(s)/Beq:2.5°=1.0086 / 5°=1.0347 / 7.5°=1.0794(局地半频 0.5043/
  0.5174/0.5397)。固定全局 0.46–0.56 带在不同纬度会把局地 fce 移动混入 gap。
- **判据⑥改**:主方向 5° 探针成熟 element Bw/B0(s) ∈ [3e-3, 1e-2]。

## 决定 2:三种子立即执行(同一 x10/Lu source)

seeds 20260720/21/22,除 seed 与 nsteps 外零改动,全部 t≥6000,同一
executable hash;seed20 t4000 = pilot,正式记录延到 t6000(restart 允许)。
**成功条件(3/3)**:①t<2000 无 x4 式 flood;②至少一条空间-时间连续
outward element;③5° 处 Δω≥0.15 Ωe,eq 且 ω_max≥0.55 Ωe,eq;④三 STFT 窗
均确认过 0.55;⑤5°→7.5° 到达时间与放大顺序符合 outward;⑥高频结构 =
同一 convective packet(不许两个不连通 dots 拼谱)。
预注册硬门槛 3/3 ≥0.55 不得事后提高;"达 0.60/0.65 的种子数"作第二指标。

## 决定 3:天然浅谷 = paired baseline,不是 gap 机制证据

V_ctrl=0.87 浅谷 + 真实上带(up/hi=75)→ gap 带 control 分母非噪声,
S_A 增量阻断可靠。但 m=0 control 无 E∥ 也能自然产生浅谷 →
**gap 外观本身不证明 RSM/Landau 机制**。论文主张 = "RSM 在已有浅谷基线上
产生额外抑制,使原本穿过的 control element 停止"。主证据顺序:
①paired ctrl 连续穿过;②paired RSM 停在局地半频附近;③Δω_stop≥0.04;
④S_A<0.5;⑤下带 element 保留;⑥m1 Landau→plateau→m0 drive 降→stop 时序。
命名:"control baseline valley";不得称 control 已证 Omura gap,不得把
与 Ke2022 外观相似等同为机制相同。

## 修订执行顺序

1. 修判据/追踪器(双位置 + 方向预注册 + 双归一化);
2. x10/Lu 三个 t6000 controls;
3. 3/3 过 → x10 上 RSM 短校准;
4. 不照搬 x4 的 k⊥=0.24 —— 先测 x10 source 下的 R_m,选实际 R_m≈0.2–0.4
   的弱通道作主 case;
5. 三个逐 seed 配对 RSM。
