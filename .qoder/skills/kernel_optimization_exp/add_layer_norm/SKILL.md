# add_layer_norm 算子优化经验

## 算子信息
- **算子名称**: add_layer_norm
- **算子语义**: out = LayerNorm(x + y, gamma, beta)
- **生成时间**: 2025-06-13T17:42:00Z
- **kope 版本**: ascend-kernel-kope-lite

## 优化历史

### Session: sess_add_layer_norm_20250613_1742

#### Iteration 0: TileLang Design
- DSL: design/tile_level/add_layer_norm.py
- Strategy: merge_n (N≤1024, row_factor=8) + single_row (N>1024) + ReduceSum + dynamic_blockM
- Status: PASS
- Insight: 纯 Vector 算子, ReduceSum 替代标量循环是核心

#### Iteration 1: AscendC Translation (Current)
- Kernel: kernel/add_layer_norm_merge_n_kernel.h + kernel/add_layer_norm_kernel_opt.h
- Strategy: ReduceSum 逐行归约 + merge_n 多行合并(row_factor=8) + dynamic_blockM(64/128)
- Effect: step_improvement=25x (对于 N≤8192 的标量循环场景)
- Decision: ACCEPT

## 已验证的优化策略

### 有效策略 (按优先级)

1. **reduce-sum-replace-scalar-loop**
   - 类别: compute (instruction)
   - 描述: 将逐元素标量累加循环替换为 AscendC::ReduceSum 硬件矢量归约
   - 适用: N ≤ 8192 (UB 容量限制)
   - Pattern 文件: opt_memory/patterns/reduce-sum-replace-scalar-loop.md

2. **merge-n-multi-row-normalization**
   - 类别: compute (tiling)
   - 描述: N ≤ 1024 时 row_factor=8 多行合并处理
   - 适用: N ≤ 1024
   - Pattern 文件: opt_memory/patterns/merge-n-multi-row-normalization.md

3. **dynamic-blockM**
   - 类别: parallelism (core_partition)
   - 描述: M ≤ 4096 → blockM=64, M > 4096 → blockM=128
   - 适用: 所有 normalization 类算子

## 最佳配置

### 关键参数
- blockM: dynamic (64 for M≤4096, 128 for M>4096)
- row_factor: 8 (merge_n path, N≤1024)
- vec_num: 2
- double_buffer: depth=2 (merge_n path)
- reduce_sum: AscendC::ReduceSum (tensor前n个数据计算)

## 使用条件

### 可直接复用
- 算子类型: add_layer_norm, layer_norm, rms_norm, add_rms_norm
- Shape: M ∈ [128, 4096], N ∈ [256, 8192]
- 硬件: Ascend 910B
- Dtype: float16, float32, bfloat16

### 需要重新探索
- N > 8192: 需要 N-tiling splitd 路径
- M 极大或极小
- 不同硬件型号
