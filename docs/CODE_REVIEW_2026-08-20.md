# pic2d 代码审查与清理(2026-08-20,响应用户两轮审计)

## 本轮已修(全部重编译 + GPU 测试绿:units 5/5, ckpt 3/3, δf-hold 4/4, sparse 10/10)

1. tests/test_deltaf2d_hold.cu:acc 1→4 doubles(k_wd_stats 写 acc[0..3],原先越界)
2. tests/test_sparse2d.cu:flat 路径 ok_f 纳入最终判定(B1 healthy ×4)
3. decks/warden2d_p2_real.ini:wdrms_max=0.25(healthy() 真执行)+ ppc 60→50
4. warden2d --contcheck:不再 `continue` 吞掉当步诊断(probe 替代 step,能量/探针/ckpt/health 照常);数值落盘 outdir/contcheck.csv
5. 发射前显存硬拒绝:need ≥ free → exit 2(原先只打印 DOES NOT FIT 仍继续)
6. 显存估算 36→44 B/marker(sorter 键/索引 + CSR 折算)——立即抓到 P2 ppc=60 实际装不下
7. W_ant 列入 energy.csv(t,W_EM,W_cold,W_ant)
8. 天线截断散度:cutoff 4σ→6σ(实测截断残差 4.3e-5 → ~1e-7 量级;不再宣称 "exact")
9. H(z) 表改中点采样(消 k∥Δz/2≈0.07 rad 载波相位偏)

## 仍然阻断 P2-real 的事项(用户判定,未动)

- continuity 升级为带冻结阈值的独立 gate(full-f floor / δf frozen floor / live δf
  / jfilter 0 与 matched / dt/2 / ppc 收敛)→ 若 live 显著高于 floor,须实现
  权重源修正或 Gauss projection
- 天线:J∥/J⊥ 比值与 source mask 输出;W_ant 用 ½(Eⁿ+Eⁿ⁺¹) 居中;
  finite-toff 远场 ringdown 标定(A-cal v2);vacuum/isotropic-null 臂
- checkpoint:绑定 deck/species/antenna/git-hash;保存 ant_wacc 与诊断累加器;
  resume 自动截断 probes/energy;非 resume 拒绝复用脏 outdir
- 二进制嵌入 git hash(当前 ckpt 里还是旧 58fd191)
- 10k/25k P2 pilot 全门确认后才考虑 300k

## 与成熟 PIC(WarpX/OSIRIS)对照的结构性差距(优先级排序)

1. **边界**:Umeda 掩模阻尼 vs WarpX 的 PML——掩模反射率从未定量测过;
   建议一次"入射包-反射包"标定实验给出 dB 数
2. **Gauss/电荷守恒监控**:WarpX 提供 divE 清理与告警;我们现在有 contcheck
   (hot δf 部分),缺 ∇·E−ρ 全量监控(cold 电荷未追踪是结构性原因)
3. **Provenance**:WarpX 每输出带完整输入+git hash;我们 meta.txt 部分覆盖,
   checkpoint 不绑定(上表阻断项)
4. **确定性**:float atomics 非确定(同 seed 不同 bit);WarpX 有 deterministic
   选项;我们以统计包络 regression 替代(已有裁定,记录在案)
5. **IO 健壮性**:fopen 多处未检查(磁盘满 = 静默段错误);低危,建议下轮加守卫
6. **可配置性**:探针纬度 {0,±5,±10,±20} 硬编码(P2 的 lam_w=10 已经踩到);
   ledger every 8、sline=snap/4 等魔数应进 deck
7. **代码布局**:header-only ~3.6k 行尚可读;sim2d.hpp 已兼任天线+探针+编排,
   下轮拆 antenna2d.hpp;kinetic2d.hpp 的 push/deposit/loader 三合一可拆
8. 好于常规研究代码的部分(保持):deck 硬门体系、未知键告警、健康巡检
   + 应急 checkpoint、双探针线 + 局地 fce 记录、shell 语义化几何
