# SESSION 2026-08-19 — 稀疏 tile 池落地 + V4R5-A5 验证三项

## 一、完成的稀疏存储一期(PLAN_SPARSE_GRID §5 详细记录)

结构:池 + int32 tslot 表;`idx()` → `(slot<<8)|((k&15)<<4)|(i&15)`
或 −1;`ld()` 保护读;沉积丢弃计数(runaway[1],healthy() 门);
checkpoint v2;deck 预检精确扫描 tile;BAND_MARGIN=130(修正旧
margin=40 只盖住 1/3 吸收 ramp 的缺陷)。

验证阶梯(tests/test_sparse2d.cu):
- S1:全 tile 活动池 vs 稠密,场管线 500 步 + 掩模 100 步 +
  两种 filter,**逐位一致**(纯寻址改动的唯一诚实标准);
- S2:mini linedipole Sim2D 三臂(稠密/全活动/带状),healthy、
  沉积零丢弃、W_EM 包络、带状池确实小于稠密。

## 二、用户的三项验证(本次 run 的唯一目标)

V4R5-A5 deck(decks/warden2d_v4r5_a5.ini):V4R4 紧凑几何
(x_min=2500,band [3300,4700],dL=60 ≥ 5.5 λ∥)+ **A=5**
(uthperp=0.345,rel=1 强制)短跑 60000 步(t=9000/ωpe,
Ωe·t=1800)。

1. **初始密度图**:warden2d 在 t=0 输出 dens_<species>.bin
   (k_dens CIC → jtmp 暂存 → unpack 稠密),plot_density2d.py
   叠加设计 shell 等值线(L0±dL/2 实线、±3σ 边缘虚线、活动带
   白虚线),并打印 ⟨L⟩ 与 3σ 包络内权重占比——粒子初始位置必须
   与设计 L-shell 完全一致。
2. **whistler 激发与 chirping**:A=5 驱动强,增长时间在数百
   Ωe⁻¹;quicklook2d.py(通用化,全部参数读 meta.txt)每 10000
   步出谱图;判据:能量指数增长后饱和,±5°/±10° 探针谱图可见
   分立上升调(chirping)。
3. **波的传播图**:plot_wavemap2d.py 从 f2d 快照(每 5000 步)
   画 By 波场图,叠加 shell 与 band 边界——波沿磁力线传播、在
   band 边缘与吸收框内衰减,不得有 band 硬切割处的反射条纹。

## 三、代码可维护性(用户要求"清晰简单,deck 通用易操作")

- 输出节奏全部进 deck:[diag] probe_every / fv_every / dens_init
  (关闭了审查遗留项 D4 的大半);
- meta.txt 自描述:box 几何、profile、每种群 shell 设计、active
  band、探针表——所有画图脚本零硬编码,`脚本 <outdir>` 即用;
- 诊断/画图三脚本共用 meta 解析;runner 不再为实验改动。

## 四、执行结果(2026-08-19 完成)

**Gate 结果:sparse 10/10 全绿**(S1 逐位、S2 三臂),快速回归全绿
(units 5/5、ckpt v2 3/3 逐位、absorber 5/5、orbit 23/23)。

**S2 抓到一个潜伏 loader bug(重要)**:rejection sampling 64 次
耗尽后保留最后一个被拒候选——大 loader box(接受率 ~5–9%)下
百分量级 markers 被静默放在 shell 外(测得 2 个落在 prof=e⁻⁷²²
的 L≈400)。修复:尝试 1024 次(前 64 次槽位逐位保留,V1/V3 统计
不受影响;新槽位 4096+ 避开 kappa 流 210+),真耗尽放 shell 中心
并计数(runaway[2],build 时报告)。**V2R2/V4R3 的加载带此缺陷**
(当时无密度图诊断);V4R5 起为修复版。

**S4 规模基准**(V4R5, 3.72e8 markers, 8.77e7 cells):
sparse 53.6 ms/step / 17.78 GB vs dense 63.6 ms/step / 18.85 GB
(快 16%,省 1.1 GB;紧凑盒下 pool=80% dense——大头显存节省来自
V4R3→V4R4 的盒缩减,sparse 提供剩余部分与多 GPU 地基)。对比
V4R3 的 186 ms/step:吞吐 3.6×。剩余 ~14 GB = production ppc
提升空间(100 → ~150)。

**V4R5-A5 运行**(60000 步 = t 9000/ωpe,58 ms/step 稳定,
runaway 0 全程,58 分钟,commit 4f7bc75 后启动):

1. **密度验证 ✅**:⟨L⟩=4000.2(设计 4000),99.92% 权重在 3σ
   包络内,无游离晕(loader 修复的直接可视确认)。
2. **激发与 chirping ✅(定性)**:t≈650 波包出现,t≈2000 起指数
   增长,t≈5000 饱和 W_EM≈2.1,δBy 峰值 1.5e-2 = 7.5% B0eq。
   谱功率集中 lower band 0.15–0.5;+5° 探针可见多个分立上升
   元素(riser,持续 ~300–500/ωpe,斜率量级 ~1e-3 Ωe²_local);
   **0.5 Ωe 处功率清晰截止**,0.55–0.75 零星 upper-band 内容——
   gap 形态定性复现。定量 df/dt 受 A=5 短元素 × STFT 分辨率积
   限制,留给 A 阶梯 production 长跑。
3. **传播验证 ✅**:波包沿磁力线束伸展至 |z|≈1100(λ≈17°),
   中纬斜波前条纹清晰;band 边缘(L=3300/4700)无堆积、无反射
   条纹;吸收框工作正常。

图件:build/v4r5/{density_init, quicklook_*, wavemap_*_zoom,
chirp_fine}.png;fv/ledger dumps 于 20000/40000/60000。
plot_wavemap2d.py 新增 --zoom=x0,x1[,z0,z1] 全分辨率大图
(用户要求 x∈[3500,4200] 常备)。

## 五、下一步

- production 路线不变:A 阶梯(5→3→…)+ ppc 150 的 V4R4 系
  长跑,大量测试先行;
- deferred:D1(V3 修复版随机数重跑)、D2(Nsight 定位场管线
  ~50ms)、精细 riser 追踪脚本(ridge tracking)。
