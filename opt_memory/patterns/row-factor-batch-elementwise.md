# row-factor-batch-elementwise

## 适用场景
- **算子类型**：activation, elementwise
- **Shape 范围**：M 任意, N 推荐 <=2048 时生效。大 N 自动降级。
- **硬件约束**：UB 空间需容纳 rowFactor * N 个元素。Ascend910B UB=256KB，fp16 下 rowFactor=8 时 N_max=16384 安全。

## 核心思路
对 memory-bound elementwise 算子，当 N 较小时（如 N<=2048），使用 rowFactor 一次处理多行（如 8 行），减少 GM 访问次数。大 N 时降级为逐行处理以避免 UB 溢出。此模式适用于 relu、gelu、sigmoid 等纯 elementwise activation。

## 代码模板
```python
// rowFactor = (N <= 2048) ? 8 : 1;
// subBlockRows = blockM / vec_num;
// rowLoops = (subBlockRows + rowFactor - 1) / rowFactor;
// pipe->InitBuffer(xInQueue_, 1, rowFactor * N * sizeof(T));
// for loop < rowLoops: ProcessRows(rowStart, min(rowFactor, M-rowStart));
```

## 已知限制
- 仅适用于纯 elementwise 无跨行依赖的算子。bf16 在 dav-m200 架构下 bfloat16_t 类型不可用。

## 来源
- 首次发现于：relu 优化 2026-06-13
- 相关 Case：case_relu_sess_20260613_4721_0

## 状态
待验证
