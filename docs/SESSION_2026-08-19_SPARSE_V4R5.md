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

## 四、待办(本 session 内顺序)

1. Bash 权限恢复后:重建 → test_sparse2d(S1/S2)→ 快速回归
   (units/ckpt/absorber/orbit)→ bench 稠密 vs 稀疏(V4R5 规模,
   显存 + ms/step)→ preflight 报告;
2. 跑 V4R5-A5:先出密度图(验证项 1)送用户,再全程前台跑,
   每 10000 步谱图 + 波图(验证项 2、3);
3. 通过后:commit,更新 MEMORY/计划文档;ppc 提升决策留给
   production deck(V4R4 系)。
