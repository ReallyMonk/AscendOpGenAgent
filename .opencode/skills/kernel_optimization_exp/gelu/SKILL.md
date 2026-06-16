# GELU 算子优化经验

## 算子信息

- **算子名称**: gelu
- **生成时间**: 2026-06-13T18:55:00
- **Session ID**: sess_gelu_20260613_01
- **kope 版本**: ascend-kernel-kope v2

## 优化历史

### Baseline (litebench)
- DSL: litebench/baseline_implementation/gelu/kernel/gelu_kernel.h
- 策略: 单行逐行处理，blockM=128，无N-tiling
- 正确性: PASS
- 限制: N 必须 ≤ UB 容量

### Iteration 0 (当前)
- 实现: qoder_output/gelu/kernel/gelu_kernel.h
- 策略: Shape-adaptive 3-path (N-tiling + double-row batching + dynamic UB)
- 正确性: PASS
- 性能: 3.018ms mean (vs PyTorch 1.738ms)，speedup 0.576x
- 决策: 正确性通过但性能劣于 PyTorch 内置实现

## 优化策略

### 有效策略 (正确性层面)

1. **shape-adaptive-elementwise-tiling**
   - 类别: tiling
   - 描述: 根据N维度自动选择执行路径——N≤4096双行批处理、4096<N≤8192标准单行、N>8192沿N轴分块
   - 代码变更: 新增 needTiling_ 标志、TILE_N_MAX=8192、batchSize 动态计算、ProcessRowBatch/ProcessRowTiled 双路径

2. **DataCopyParams 跨步拷贝**
   - 类别: memory
   - 描述: tiled-N 路径使用 DataCopyParams(blockCount=2, srcStride/dstStride) 实现两行跨步 GM↔UB 拷贝
   - 代码变更: CopyInTile/CopyOutTile 函数

### 已知局限

- Element-wise 自定义 kernel 在 Ascend 910B 上难以超越 PyTorch 内置 GELU（PyTorch 在框架层做了算子融合）
- 当前实现正确性通过但性能约为 PyTorch 的 57%

## 最佳配置

### 关键参数
- TILE_N_MAX: 8192
- DEFAULT_BLOCK_M: 128
- batchSize: N≤4096→2, else→1
- BlockM (动态): M≤1024→32, M≤4096→64, else→128

### UB 缓冲
- bufN = min(N, TILE_N_MAX)
- xInQ_/yOutQ_: bufN * batchSize * sizeof(T)
- castBuf_/cBuf_: bufN * batchSize * sizeof(float)

## 使用条件

### 可直接复用
- 其他 element-wise 激活函数 (swish/silu, tanh, relu 等) 的 shape-adaptive tiling
- 相同硬件 (Ascend 910B/910B4)
- N 在 1024-32768 范围内

### 需要重新探索
- 需要超越 PyTorch 内置性能时（考虑算子融合策略）
- 输入 shape 差异极大 (M 或 N < 256)
- 不同 NPU 型号

## 产出清单

| 产物 | 路径 |
|------|------|
| trace.md | /home/hrl/AscendOpGenAgent/qoder_output/gelu/trace.md |
| 性能报告 | /home/hrl/AscendOpGenAgent/qoder_output/gelu/gelu_performance_report.html |
| kope-mem Case | case_gelu_sess_20260613_01_0 |
| knowledge-cards Card | case_gelu_sess_20260613_01_0 |
| Pattern | shape-adaptive-elementwise-tiling |
