# PLAN_2D — 回到全 2D 路线(2026-08-08,RSM 单 k⊥ 判定之后;用户指令)

## 0. 背景:为什么回 2D,以及 2D 继承什么

RSM phase-4 判定(docs/PAIR_VERDICT_2026-08-08.md,0/3):单固定 k⊥=0.32
弱通道 + m0/m1 约化在 x10 Lu 源上**不产生可见 0.5 gap**,只产生元素削顶
(+0.03~0.07)与活动期缩短;S_A 失败对频带口径稳健。Constitution §6.1
风险 #4(单固定 k⊥ 可能不足)兑现。

但两个既有结果划定了问题的形状:
- **分水岭(rsm_band_big)**:均匀 B 大盒 + Li 源,m0+m1 把 gap 挖到
  30–1000×——约化模型**在线性均匀源上够用**;
- **Li 2019 in-house 复现**(15° tilt Yee 全 2D):two-band + 0.5 gap ✓。

即:失败不在"m1 通道不存在",在于 **chirping 镜场源 + 单 k⊥** 这个组合。
2D = 未约化假设的直接检验:自洽 k⊥ 谱、WNA 随传播演化、m1×m1→m2、
局地条件全都自动在场。

**继承的资产**(全部现成):
1. x10 Lu 源 4/4 seeds 跨 0.55(冻结审计)——**四个 1D 归档 run 就是
   天然的 k⊥-off 对照臂**,逐 seed 配对不用重跑;
2. G1.2 2D slab mirror(b0_prof=3,By=−2aB0x̃ỹ,∇·B=0 精确,y-resolved
   完整镜力)已在共享引擎 yee2d.hpp,mirror2d.cu 为验证参照;
3. G1.3 诊断包:fv|region、J·E Landau-vs-cyclotron ledger、WNA;
4. cold_model=full(3 分量冷流体)在 config 中已标注"ny>1 必需";
5. 冻结判定工具族(ridge audit / W10 / 连通域)与协议纪律。

## 1. 物理问题(预注册)

**同一个盒子里:镜场 chirping 源(1D 已证跨 0.55)+ 自洽 k⊥ 谱,是否
自生成 0.5 fce,eq 的可见 gap?**

- "可见 gap" 标准沿用用户 08-04 判语(D120 教训):必须是**可见谷**
  (P_gap < P_up),削顶不算;单位 = **Ωe,eq**(文献口径,ke2022jgr/
  Li2019/Ke2017 原文已核:报告层 = 赤道 fce,机制层条件是局地的)。
- 主对照 = 1D 四种子归档 run(同 f0、同 lre、同边界,唯一差 = k⊥ 自由度)。

## 2. 参数与成本(提案,P0 冒烟实测修正)

| 项 | 值 | 依据 |
|---|---|---|
| 几何 | x10:Lx=1300, nx=5000, dx=0.26, lre=1330.5 | 与 1D 臂严格同 |
| 横向 | Ly=39.27, ny=160, dy=0.245 | k⊥ 格 = 0.16n → {0.16,0.32,0.48,…} 覆盖 Li 机制带 k⊥/k∥~0.2–0.6 |
| dt | 0.15(CFL 上限 0.178)| 与 1D 同,t6000 = 200k steps |
| ppc | 690(1D 等效:690×160≈110k/列)| 噪声与 1D 臂同级;总 552M markers |
| 显存 | 估 ~30GB(1D 300M 用 16.5GB 外推)| **边缘,P0 冒烟 gate;备选 ppc=500 → ~22GB** |
| 速率 | 5–7e9 p-steps/s → t6000 ≈ 4.5–6.5 h | 过夜可行;3 seeds = 3 晚 |
| ρ⊥/dy | 1.73/0.245 ≈ 7 | G1.2 规则 ≥2 ✓ |
| y-wrap 踢 | 2a·x̃·Ly/B0 ≈ 11% @ |s|=650 | ⚠️ 超 G1.2 薄层假设,P1 实测 gate |

## 3. 阶梯(每级 gate + stop rule)

### P0 工程(~1–2 天)
chirp2d 扩 ny>1 生产路径(引擎已 2D,改的是驱动):
- b0_prof=3 + cold_full 接入 chirp2d 主循环;
- 诊断:probes 取 (x_p, y=mid) 行;bline 双份(y-mid 行 + y 平均);
  新增 **kyspec**:少数 x 站(eq/±5°/±7.5°)的 By(x, ky, t) 复谱
  (= RSM m1line 的推广,直接对接现有分析);ckpt 兼容;
- mirror2d 保留为交叉验证参照,不做生产。
**Gates**:① ny=1 回归 = 统计包络 ≤3×(regress_case2 类,off-path
不许动);② t100 2D 冒烟:显存/速率实测 → 定 ppc;③ 均匀盒短对拍
chirp2d-2D vs mirror2d(场能曲线一致)。

### P1 数值 gates(~0.5 天,短 run)
- y-wrap 边界层:实测受 >2% 场跳变的 marker 份额与其散射特征;超标
  → Ly 减半(k⊥ 格 0.32n)或 y-taper;
- 能量守恒 ≤ 1D 同级;2D quiet-start 噪声地板 vs 1D(WB(t=0) 比)。

### P2 2D control(1 晚)
Lu f0、x10、seed 20260720、t6000。判源 = 冻结 ridge audit(y-mid 探针,
判据原文不动)。**已知风险即发现**:2D 斜向/Landau 阻尼可能压低源——
- 若仍跨 0.55:进 P3;
- 若停在 ~0.5:**这本身可能就是答案**(自洽 k⊥ 的内禀 stop = gap 机制
  的 2D 表现),转 P3 但主指标换成 1D-vs-2D 差分;
- 若源整体死(无元素):停,短盒诊断,**不做参数轮盘**。

### P3 gap 判定(预注册,写死后才看谱)
逐 seed 与 1D 臂配对,三条证据线:
1. **可见谷**:eq/5° 谱 P_gap(0.46–0.56) 与 P_up(0.56–0.66)、gap/LB;
   可见 gap 标准 = P_gap < P_up 且 gap/LB ≤ 0.3;
2. **差分形态**:P_2D/P_1D 比值谱谷底位置(0.5 带 vs riser 顶)——
   区分挖谷/削顶的判决性测量;
3. **机制链**(G1.3):上/下带 WNA 分离、k⊥ 谱时间演化、J·E Landau-vs-
   cyclotron ledger 空间分布、f(v∥) plateau 0.09–0.11c、E∥/E_tot。

### P4 种子 ×3 + 因果实验(M10-SQ 清单兑现)
1D-vs-2D 已内置;备选:kill-the-parent(上带谱阻尼)、component switch。

### P5 收束
- 2D 挖谷 + 1D 不挖 → **k⊥ = 因,flagship 主结果**;回头定位 RSM 缺失
  成分(k⊥ 谱?WNA 演化?m2?)→ bridge/方法论文;
- 2D 也只削顶 → **削顶即物理**:RSM 与 2D 一致,叙事转 ke2022
  "gap+dots"(RSM 判定书选项 D 升主线);
- 2D 源死于斜向阻尼 → 内禀 stop 机制研究(短盒)。

## 4. 风险登记
显存 30GB 边缘 / y-wrap 11% / 2D 源变弱或死 / cold_full 未在 2D 生产
验证过 / 速率估计 ±2× / 1D 冻结度量(W10 带宽等)在 2D 需重校核但
**判据文字不改**。

## 5. 禁做
不再跑任何 RSM 臂(除非用户点名);不把削顶叫 gap;P3 标准冻结前不看
2D 谱;P2 源不过不加密度/ppc 轮盘;ny=1 全家族路径按构造不动。

## 6. 立即下一步
P0 工程:chirp2d ny>1 分支 + 三个 gate。完成后向用户报冒烟实测
(显存/速率/ppc 定案)再进 P2。
