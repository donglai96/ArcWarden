# G1 成对实验判决(2026-08-05)

分支 `rsm-test`,提交 `6a13468`。上游文档:`docs/RSM_TEST_PLAN.md`(§3 波源标准、§7 G 方案)、
`docs/RSM_MODEL_DEFINITION.md`(模型宪法)。本文档记录 G 路线第一对 RSM-on/off 实验的完整判决。

## 一句话结论

G1 波源七项标准全过(第一次拿到"清晰单元 + 覆盖上频带"的源);开 RSM 后出现**选择性抑制**
且上扫单元**停在 0.491**(对照穿过 0.5 到 0.511),方向支持门控假设;但**没有可见的两带空隙**
——RSM 臂的上肩从未形成,0.5 以上只剩噪声底,结果是"频谱在 0.5 被砍断"而非"两带夹一缝"。

## 1. 波源筛选(RSM-off,expG1_ctrl_r2)

Deck:`decks/giant_x4_expG1_ctrl.ini`(expG 引擎 nh=0.0178 + 注入煎饼 n2=0.003,
upar2=0.06,uperp2=0.147;seed 20260720,t=10000/Ωe)。
判决脚本:`scripts/g1_source_verdict.py`(图 `build/expG1_ctrl_r2_verdict.png`)。

§3 七项标准:

| 条件 | 结果 |
|---|---|
| 1. 单条连通上扫 ridge | ✓ t2300–6700 连续一条 |
| 2. 扫频 Δω/Ωe ≥ 0.15 | ✓ 0.213→0.511,Δω=0.30 |
| 3. ridge 越过 0.50 | ✓ wmax=0.511 |
| 4. 0.55–0.60 上肩可检测 | ✓ 单元顶部窗 [5500,7500] 达噪声 20×(全窗平均只有 2×——时间稀释陷阱,与 box-avg 同类) |
| 5. 无天然空隙 | ✓ V=gap/√(lo·up)=1.6–7.5,全程 ≥0.7 |
| 6. 测点时延一致 | ✓ onset 单调:+291(3072)→+145(3303)→eq(3533)→−145(3936)→−291(4109) |
| 7. 振幅量级 | ✓ Bw/B0=7.2e-3,单元尺度,非 ArmU 式过驱 |

## 2. 成对 RSM 判决(expG1_rsm)

Deck:`decks/giant_x4_expG1_rsm.ini` = ctrl + `[rsm] enable=true, k1=0.16`(唯一差异,
Ly=2π/k1 相位契约已在 ctrl 中设好)。同 seed、同长度。

### 2.1 支持假设的证据

- **选择性抑制**(`scripts/armu_gap_verdict.py`,图 `build/g1_pair_burst.png`):
  固定 burst 窗 [3500,8500] 全五测点 S<1(0.23–0.64);单元相位对齐窗(各臂用自己的
  单元窗,ctrl [2324,6740] / rsm [1364,5914])S = 0.11(eq) / 0.22(+145) / 0.50(−145)。
  gap 频段被压得比上下肩更狠——不触发"整体压低不算 gap"的 stop rule。
- **ω_stop 门控方向正确**:RSM 单元停在 **0.491**,对照 **0.511**(`scripts/x4_omega_stop.py`)。
  正是 gap-as-statistics 的预言方向;但每臂 N=1,不足以排除巧合。
- **空隙频段绝对变暗**:RSM 臂单元窗内 gap/噪声带 = 0.32–0.47,对照同窗 2.9–7.3。

### 2.2 推翻"可见 gap"的证据(噪声底检验)

- 初算 "P_gap < P_up 三测点全真、V=0.15" 貌似可见空隙,但 up/hi(0.65–0.75 噪声带)
  = 0.76–0.97:**所谓上带就是噪声底**。真实情况 = gap 被压到噪声以下,上带不存在。
- RSM 臂上肩全程 sh/噪声 ≤ 1.0(对照 20×):**上肩从未形成**。

### 2.3 机制发现:m1 疯长 + 单元削弱

图 `build/g1_m1_history.png`:

- **m1 长到与 m0 rms 平价**:t4000 时 2B1rms/B0 = 8.3e-3 ≈ m0 rms 8.4e-3;峰值
  2|B1|max/B0 = 2.9e-2(t5000)。与 uniform-B 分水岭一致(2W1 到达 WB 平价)。
- 副作用:burst 提前约 2000/Ωe(m1 噪声提前点火)、WB 峰低 2.9×(0.28 vs 0.81)、
  **相干单元包络弱 3.4×**(Bw/B0 2.1e-3 vs 7.2e-3)。注意 m0 总波量并不弱
  (t4000 rsm m0 rms 反而 > ctrl)——被削弱的是"组织成相干单元"的那部分。
- f(v∥) 末态(t9000 ckpt,`scripts/ckpt_fvpar.py`,图 `build/g1_fvpar.png`):
  rsm/ctrl 在共振带 [0.06,0.14] 仅 +2%,弛豫后的弱证据。

### 2.4 深层困境(与 D120 同构,但更深一层)

对照组的上肩是单元**穿过 0.5 之后**继续上扫喂出来的。若 RSM 真在 0.5 门控 chirp,
它拦住单元的同时也断了上肩的粮——"拦住了"与"看见两带"在此波源上互相矛盾,
**除非上频带有独立于单元穿越的增长源**。这正是 G2(n2=0.0035,更强的煎饼独立增长)
的意义所在。

## 3. 进行中与待决

- **进行中**:seed-2 成对(seed 20260721,`decks/giant_x4_expG1_{ctrl,rsm}_s2.ini`,
  顺序 ~10h)。目标:ω_stop 集合统计——若对照系统性越过 0.50 而 RSM 系统性停在
  ≤0.49,门控假设成立。
- **用户裁决点**(seed-2 出结果后):
  1. 走 G2:给上带独立增长源,追求真正可见的两带空隙;
  2. 接受"截断即机制":空隙的本质是 chirp 门控,可见两带需要双源结构,
     以 ω_stop 统计为主要证据链;
  3. 若 seed-2 统计不支持门控(RSM 停点不稳定/对照也停),回到源/架构讨论。

## 3.5 追记(2026-08-05 晚):seed-2 control 不合格 → G1 波源判定为"种子脆弱"

seed-2(20260721)control 自身**未通过 §3**:wmax=0.470(未越过 0.50);上肩在所有窗口
(含单元顶部窗)≤1.6× 噪声(seed-1 是 20×);burst 窗 V=0.575 出现天然谷。
按 plan 明文 stop rule("control 本身不能跨过目标频段就不跑对应 RSM 长实验"),
rsm_s2 于 37% 处停止(数据保留,`build/expG1_rsm_s2` 1228 blines)。

**对 seed-1 门控证据的冲击**:ctrl 自身的 ω_stop 种子间波动(s1 0.511 vs s2 0.470,
Δ=0.04)**大于** seed-1 的 rsm-ctrl 差(0.491 vs 0.511,Δ=0.02)——门控信号目前
淹没在波源自身方差里。要么波源必须先做到"跨 0.5 种子稳健",要么需要 O(5–10) 对
种子做统计。已按 plan §7 顺序转 **G2 RSM-off 筛源**(n2=0.0035,
`decks/giant_x4_expG2_ctrl.ini`,run `build/expG2_ctrl`),新标准:除七项外,
额外要求跨 0.5 与上肩有足够裕度,并在配 RSM 前先验种子稳健性。

## 4. 数据与产物

- 运行:`build/expG1_ctrl_r2`(13G)、`build/expG1_rsm`(13G),各 t10000,
  binary sha `f9bae814`(chirp2d,BUILD_PROVENANCE 记录在各 run 目录)。
  旧的半程 `build/expG1_ctrl`(t1340)按 keep-data 规则保留。
- 图:`build/expG1_ctrl_r2_verdict.png`、`build/expG1_rsm_verdict.png`、
  `build/g1_pair_burst.png`(成对谱图)、`build/g1_m1_history.png`、`build/g1_fvpar.png`。
- 脚本:`scripts/g1_source_verdict.py`(新)、`scripts/armu_gap_verdict.py`、
  `scripts/x4_omega_stop.py`、`scripts/ckpt_fvpar.py`。
