# RSM-TEST 分支验证计划(2026-08-03)

> 分支 `rsm-test`,HEAD = `6f63683`。本文件 = 本轮工作的 source of truth。
> 结论:继续用 `rsm-test`,但不要马上扩大 k1 扫描,也不要继续直接强化初始各向异性。正确顺序:
> 1. 在修复后的提交上复现旧的强驱动 gap;
> 2. 用 RSM-off 调出"既清晰、又能覆盖上频带"的源;
> 3. 再做唯一变量为 RSM 开关的成对实验;
> 4. 最后做多随机种子、k1 和 G 双组分验证。
>
> 目前的首选仍然是"保持高各向异性、降低热电子密度",G 作为后备方案。

## 1. 当前代码状态

当前分支:

```text
rsm-test
HEAD = 6f63683b51c4f5f541b5d251c682387bf6303377
```

关键修复已经分成两部分:

- `f575a97`:修复 `ny=1 + tile_sort` 下 RSM apron flush 的越界问题。这会直接影响旧的 full-f Arm U/G 结果。
- `6f63683`:补上 flat RSM δf 沉积中的 `wd` 权重。这个修复主要影响 `tile_sort=0` 的 δf 路径,对现在的 full-f tiled 生产运行没有直接物理影响,但使 flat/tiled 验证可信。

相关位置:

- `include/pic/rsm_oblique.hpp` (~421)
- `include/pic/yee2d.hpp` (~505)
- `include/pic/simulation_maxwell.hpp` (~94)

已有测试记录是 38/38 通过,RSM 色散、偏振、连续性、∇·B1 和能量账本都在合理范围。

但是,目前没有一组修复后重新完成的 full-f gap 生产运行。旧 Arm U 和 G 都是在 apron 越界修复之前得到的,因此只能作为物理线索,不能作为最终证据。

还有一个工程风险:`RUN_META` 是运行时读取当前 git hash,因此即使可执行文件较旧,它也可能写入新的 HEAD。正式运行前必须强制重新编译,并记录 executable hash,不能只相信输出中的 git hash。

---

## 2. 第一阶段:修复后复现旧 Arm U gap

先不要改物理参数。直接重跑旧 Arm U 成对实验:

- `decks/giant_x4_u2band_ctrl.ini`
- `decks/giant_x4_u2band_rsm.ini`

保持:

```text
nh       = 0.0178
upar     = 0.19783565
uperp    = 0.31283
Tperp/Tpar = 2.5
A        = 1.5
k1       = 0.16
full-f
conecut
tile_sort = 25
same particle seed
```

第一轮只跑到:

```text
tΩe = 3000
nsteps = 100000
```

目的不是生成好看的 element,而是回答一个基础问题:

> 修复越界之后,旧 Arm U 中的选择性 gap 是否仍然存在?

需要同时比较:

- equator;
- 源区附近 z=±145;
- z=±291;
- RSM-off 和 RSM-on 的 m=0 波谱;
- RSM-on 的 m=1 场强和频带。

接受条件:

- control 在 0.45–0.55 没有同样的深谷;
- RSM-on 在源区对该频带的抑制明显强于上下两侧频带;
- 不是所有频率统一下降;
- gap 在 equator 和至少一对近源测点可见。

旧结果里 equator 的 gap-band power 大约降到 control 的 0.13,valley 指标约为 0.25;这是很强的信号。但如果修复后不能重复,就必须判定旧 gap 主要受到数值问题污染,不能继续沿用。

这一阶段不要长跑到 tΩe=10000。强 Arm U 本来就会形成 broadband burst,短跑足够验证 gap 机制是否还存在。

---

## 3. 第二阶段:只用 RSM-off 调整 chirping source

现在真正的问题不是"RSM 没有 gap",而是:

> 初始源太强时有 gap、但没有清晰 element;源变弱后有 element,却不能覆盖 upper band。

这里不要降低各向异性 A,因为降低 A 会直接降低线性不稳定频率上限。更合理的是:

- 保持 A=1.5;
- 保持 upar=0.19783565、uperp=0.31283;
- 只降低热电子密度 nh。

这样线性正增长上限仍约为 ω/Ωe=0.60,但总驱动下降。

建议候选:

| 实验 | nh | 预计作用 |
|---|---:|---|
| D65 | 0.0065 | 最容易得到清晰 element,但可能偏弱 |
| D75 | 0.0075 | 首选平衡点 |
| D90 | 0.0090 | upper band 更强,但更可能再次变 broadband |

对应设置:

```text
cold_nc = 1 - nh
ppc      = 36000
RSM      = off
antenna  = off
refresh  = off
```

先跑 D75。根据结果决定下一步:

- D75 仍然是 broadband burst:转 D65;
- D75 很清晰但到不了 0.50–0.55:转 D90;
- D75 同时有清晰 ridge 且能越过 0.50:直接成为候选,不必把三个点全部跑完。

每个 screening run:

```text
tΩe = 6000
nsteps = 200000
```

判定一个合格 source,不能只看自动 element 数量。旧的 `x4_omega_stop.py` 会把一个宽带爆发切成许多"element",因此应同时检查二维谱图和 ridge 连通性。脚本位置:

- `scripts/x4_omega_stop.py`
- `scripts/whistler_growth.py`

合格 source 应满足:

1. 出现单条连通的 rising-tone ridge,而不是整块频带同时增强;
2. 单个 element 的扫频至少约 Δω/Ωe≥0.15;
3. ridge 越过 0.50;
4. 0.55–0.60 仍有可检测的 upper shoulder;
5. control 自身在 0.46–0.56 没有自然 gap;
6. equator 和传播方向测点显示一致的时间延迟;
7. 最好振幅回到基准 x4 element 的量级,而不是 Arm U 的 4–6 倍过驱动状态。

---

## 4. 第三阶段:将候选 source 延长成长时 control

找到 D65/D75/D90 中最合适的点后,先仍然只跑 RSM-off,延长到:

```text
首选:tΩe = 20000, nsteps ≈ 666667
最低:tΩe = 15000, nsteps = 500000
```

原因是现有 base x4 中,比较稳定的 element train 主要出现在 tΩe≳10000。只看到早期第一个 element 不足以判断后续能否持续覆盖 upper band。

进入 RSM 成对实验前,control 必须满足:

- 至少有两个清晰的后期 element;
- 不是只有初始化瞬态;
- 至少一个 element 穿过 0.50;
- 在 0.55 左右有连续功率;
- 0.46–0.56 没有预先存在的 valley。

如果 control 本身已经有 gap,就不能用它验证 RSM gap formation。

---

## 5. 第四阶段:严格的 RSM-on/off 成对实验

选定 source 后,再生成 RSM-on 版本。两组必须做到:

- 同一个 commit;
- 同一个可执行文件;
- 同一个粒子 seed;
- 相同 Ly,即使 control 不启用 RSM 也保留同样的 `Ly`;
- 相同 `tile_sort=25`;
- 相同粒子数、输出间隔、边界和运行长度;
- 唯一物理差异是 RSM enable。

首轮继续固定:

```text
k1 = 0.16
```

暂时不要扫 k1。如果源本身没有调好,扫 k1 只会把"源问题"和"传播角问题"混在一起。

完整 gap 判据:

### 波谱条件

- control 的 ridge 连续跨过 0.46–0.56;
- RSM-on 的 ridge 在这个频带出现明显凹陷或停止;
- equator 的 gap power 相对 control 至少降低 50%;
- gap 的下降强于 lower/upper shoulder 的下降;
- 上下两侧仍保留可辨识功率,不能只是整条 chirp 一起变弱。

### 空间条件

- gap 首先在 equator/近源区形成;
- 在 z=±145 或 ±291 至少一对测点可以追踪;
- 如果只有远端出现,而源区没有,需要怀疑传播、窗口或驻波效应。

### 粒子条件

重点检查理论共振区:

```text
v_parallel ≈ 0.087–0.100
```

比较 control 和 RSM:

- RSM-on 是否形成更明显的 plateau;
- plateau 出现时间是否和 element 穿越 gap 频率一致;
- 正负传播方向是否具有合理对称性。

### 模式条件

必须把 m=0 和 m=1 分开:

- gap 应体现在物理 chirping 的 m=0 谱中;
- E∥^{m=1} 应在 gap 形成前或同时增强;
- 如果只是 m=1 能量增加、m=0 全频段统一衰减,不能称为选择性 gap formation。

目前生产输出没有完整的 m=1 能量时间序列和 J1·E1 功率账本。因此前期可以用 `m1line` 中的 E1x,E1y 做时序关联;等 source 成功后,再决定是否补最小诊断。现在不应为了诊断先改代码。

---

## 6. 第五阶段:随机种子与 k1 扫描

只有第一组长时间 paired run 成功后,再做统计验证。

### 随机种子

至少三个种子(seed1/seed2/seed3)。要求不是每个 gap 深度完全一样,而是:

- control 多数都能跨过目标频段;
- RSM-on 多数都在相近频率区间出现停止或 valley;
- gap 频率变化小于 element-to-element 的自然带宽;
- 不能只有单个 seed 成功。

### k1 扫描

```text
k1 = 0.16 / 0.24 / 0.32
```

顺序是先固定 source 和 seed,只改变 k1。观察:

- gap 中心频率是否移动;
- gap 深度是否随 E∥ 增强;
- upper element 是否被完全摧毁;
- plateau 位置是否随共振条件合理移动。

这个扫描才真正回答"RSM 斜传播产生的 E∥ 是否控制 gap"。

---

## 7. 如果 density rescue 失败:转向 G

如果 D65–D90 出现以下矛盾:

- 低密度:element 清晰,但永远不到 upper band;
- 高密度:能到 upper band,但总是 broadband;

那么再转 G 双组分方案。当前 G deck:`decks/giant_x4_expG_inj_rsm.ini`。

旧 G 的问题是第二组分:

```text
n2 = 0.003, upar2 = 0.05, uperp2 = 0.1225
```

它在约 0.45–0.525 仍提供净阻尼,因此旧 element 停在约 0.45 并不意外。

更合理的两个候选:

```text
G1: n2 = 0.003,  upar2 = 0.06, uperp2 ≈ 0.147
G2: n2 = 0.0035, upar2 = 0.06, uperp2 ≈ 0.147
```

它们基本不改变 lower-band 主驱动,但能把中上频带从阻尼改成弱正增长。

执行顺序仍然是:

1. G1 RSM-off;
2. 如 upper band 不够,再 G2 RSM-off;
3. 选出清晰且跨频带的 source;
4. 最后才打开 RSM。

不要现在加入 refresh。当前 G 只是初始第二组分,并不是真正持续 injection;refresh 会增加新的自由能来源,使 gap 的因果归因变得困难。

---

## 推荐的实际运行顺序

```text
A. 固定 rsm-test@6f63683,强制重编译并跑测试
B. ArmU-fixed-control,tΩe=3000
C. ArmU-fixed-RSM,tΩe=3000
D. D75-control,tΩe=6000
E. 根据 D75 结果选择 D65 或 D90
F. 最佳 density-control 延长到 tΩe=15000–20000
G. 最佳 density-RSM 跑相同长度
H. 成功后增加另外两个 seeds
I. 最后扫 k1=0.24、0.32
J. density 路线失败才进入改良 G
```

最重要的 stop rule:如果 RSM-on 只造成全频段振幅下降,而没有相对于上下肩更强的局部抑制,就停止把它解释成 gap;如果 control 本身不能连续跨过目标频段,也不要运行对应的 RSM 长实验。
