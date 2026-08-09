# PLAN_LRE_SCAN — 1D LRE 分岔扫描(2026-08-08 用户口述,与 PLAN_2D v2 并行)

**主张边界(用户原文,预注册)**:LRE 扫描能证明磁场非均匀性控制
rising element 能否穿过 0.5Ωe,并区分"驱动停止"与"波被吸收";但它本身
不能证明 v_g=v_ph 是根本原因,也不能单独解释 upper band 来源。
论文最强主张 = "磁场非均匀性控制 0.5Ωe 附近的 nonlinear termination
或跨越;plateau damping 是否为次级锐化机制,由 frequency-resolved J·E
与 2D kinetic 对照决定。"

## 0. 现有 x4/x10 对比不可用(混杂清单)

x10(u∥=0.141, A=5, nh=0.01, bimax, hybrid)vs x4 Lu-morph(u∥=0.10/
0.12, A=5/6, nh=0.012, conecut, atmo):f0、loss cone、边界、纬度范围、
盒长、ppc 全不同 → 必须重做 one-parameter scan。

## 1. 物理预期:LRE 分岔链

B_th ∝ LRE⁻⁴(nonlinear triggering threshold):
- **大 LRE(弱梯度)**:B_th(0.2–0.3) 低 → 低频早点火,元素生在
  0.2–0.3,到 0.5 时 Bw > B_opt(ω) → 失去 trapping → 停;
- **小 LRE(强梯度)**:低频走廊关死 → 元素生在 0.3–0.4,到 0.5 时
  尚未过度放大 → 保有 phase-locking headroom → 走到 0.6–0.7。

**要点:LRE 改变的是元素出生频率,以及它到达 0.5Ωe 时处于
under-optimal / optimal / over-optimal 状态**,不是"梯度大 chirp 快"。
预期状态图:LRE → {low-frequency flood | element stops ~0.5 |
element crosses 0.5}。

## 2. 阻尼量级(用户估算,校准解读)

|γc|=1e-3 Ωe、dω/dt=1e-3 Ωe² 穿越 0.45–0.55 用时 Δt≈100/Ωe →
B_out/B_in = e^−0.1 ≈ 0.90,功率 e^−0.2 ≈ 0.82:**单次快速穿越只降
~18% PSD,不足以成 gap**。PSD 降 10×/100× 需 ∫|γc|dt ≥ 1.15/2.30。
但存在非线性可能:弱阻尼 → chirp 变慢 → 0.5 附近停留变长 → 累计阻尼
增强(反馈锁死)。线性量级排除"快穿越+弱阻尼成 gap",不排除
"阻尼—停留时间"锁死。

## 3. 关键修正:1D 扫描不能证明 Landau barrier

严格平行 m=0 1D 中 E∥=0。若纯 1D 已出现 stop↔cross 分岔,该分岔
**不可能**由 Landau E∥ 造成 → 证明存在至少一个不需要 plateau 的机制
(回旋共振电流失相干 / trapping 条件失效 / Omura 走廊改变 / TaRA relay
中断 / 波包离开源区)。**这是强结果**:没有 E∥ 和 plateau,磁场非均匀性
本身也能决定是否穿过 0.5。2D kinetic 随后回答 plateau 是否加强/锐化。

## 4. v_g=v_ph:候选临界点,需组合参数

冷平行 whistler:v_g/v_ph = 2(1−ω/Ωe) → ω=0.5Ωe 处 v_g=v_ph,carrier
对 envelope 滑移率 k(v_g−v_ph)=0。但该条件对所有 LRE 成立 → 控制量必为
组合参数,如 C = (|v_g−v_ph|/L_env)/ω_tr,或直接用共振相位方程的
nonlinear inhomogeneity ratio:

    S = [dω/dt − v_R ∂sΩe + …] / ω_tr²,   稳定 trapping 要求 |S| < 1

**有力证据 = stop case 在 0.5 附近 |S|→1 或超 1;cross case 同频率
|S|<1;且随 LRE 系统变化。**

## 5. 扫描设计(冻结)

固定(全臂零差异):f0 = Lu-x10(nh=0.01, u∥=0.141, u⊥=0.3454, A=5,
bimax 无 loss cone);ωpe/Ωe=5;dx=0.26;dt=0.15;**ppc=72000/cell**
(统一;×4 臂显存约束定值;归档 110k x10 runs = ppc 不敏感性交叉验证,
不入扫描);x=hybrid,nd=300;**wall MLAT = ±26.563° 固定**(x10 现几何
定义);t6000(200k steps);seed 20260720;RSM off;eline ON(J·E 需要)。

| 臂 | lre (c/ωpe) | 压缩 | nx | Lx | markers | 显存 |
|---|---|---|---|---|---|---|
| l1330 | 1330.50 | ×10 | 5000 | 1300.00 | ~197M | ~11GB |
| l1663 | 1663.13 | ×8 | 6250 | 1625.00 | ~246M | ~14GB |
| l2218 | 2217.68 | ×6 | 8334 | 2166.84 | ~328M | ~18GB |
| l2661 | 2661.01 | ×5 | 10000 | 2600.00 | ~393M | ~22GB |
| l3326 | 3326.26 | ×4 | 12500 | 3250.00 | ~492M | ~27GB |

**Lx = 2·g(26.563°)·lre**(g = 偶极弧长函数):不能固定 Lx 只改 lre,
否则同时改 wall MLAT/镜比/loss cone/bounce 时间/边界-源区距离。
probe 位置按固定 MLAT 缩放:s(λ) = g(λ)·lre,λ = ±{2.5,5,7.5,10,12.5,
15,20,25}° + eq(B0(λ)/Beq 与 lre 无关 → 双归一化因子全臂相同,
b5=1.0344 不变)。跑序 ×10→×8→×6→×5→×4(小臂先出数据,×4 显存最险
放最后)。预计总时长 ~10–12h。第一轮 1 seed;跨越-终止过渡臂随后加
seeds 20260721/22(预注册)。

## 6. 每元素测量(预注册)

ω_birth、ω_end(冻结 ridge audit,MLAT-5° 主判位原判据不动)、dω/dt、
Bw(ω) 沿 ridge、ω_tr、S(ω)。能量转移**不用全带总 J·E**,沿 ridge 做
复解析信号:

    P_ω(s,t) = ½ Re[ J̃_ω(s,t) · Ẽ*_ω(s,t) ]

P_ω>0 = 电子吸收波能(damping);P_ω<0 = growth;→0 = 驱动电流失相干。
完整判别 = 波作用量平衡:∂t W_ω + ∂s(v_g W_ω) = −P_ω。
(bline/jline/eline 3/Ωe 采样,载波 ≤0.7Ωe=0.14 ωpe < Nyquist 0.21 ✓)

**Spectrum bite 预期**:0.5 附近先有进入的 wave-action flux → P_ω>0 →
穿过后 action 明显降 → 共振电子同步增能。
**Generation barrier 预期**:0.5 以下 P_ω<0 相干源 → 接近 0.5 负功率
趋零 → 电流相干消失 → 无明显 P_ω>0 吸收层 → 高频延伸从未形成
(而非形成后被吃)。
**混合机制(最可能)**:drive 减弱 → chirp 变慢 → 0.5 停留变长 →
plateau/回旋阻尼吸掉残余 = generation barrier → long residence →
spectrum bite sharpens gap。

## 7. 结果解读表(预注册)

| 扫描结果 | 结论 |
|---|---|
| endpoint 随 LRE 清楚跃迁,stop 时 J·E ≯ 0 | nonlinear generation/phase-locking bifurcation |
| stop 时强局域 J·E>0 + 粒子加热 | spectrum bite/damping 主导 |
| 驱动先消失,随后正 J·E | barrier 触发,damping 加深 |
| endpoint 与 LRE 无系统关系 | x4/x10 差异来自 f0/边界/tracker |
| 某臂 broadband flood | **第三种 source state**,不得强行追踪 dominant component 比 endpoint |

## 8. 与 PLAN_2D 的组合

1. full-2D discovery run(PLAN_2D v2)= 主线,P0 工程继续;
2. 1D LRE 扫描并行(便宜);
3. 找到临界 LRE 后,2D 只跑三点:LRE_stop / LRE_critical / LRE_cross。
1D 回答"磁场非均匀性是否决定 nonlinear source branch";2D 回答
"kinetic E∥ 和 plateau 如何修改这个 branch"。

## 9. 执行

decks/lrescan_l{1330,1663,2218,2661,3326}.ini;runner
`./chirp2d <deck> <outdir> --ckpt=100000`(eline 保留);判定:冻结
lurepro_ridge_audit.py(probe 自动按 meta lre 取 MLAT-5°)+ 新
S/P_ω 工具(跑扫描期间开发,应用于已有 dump,不改判据)。
