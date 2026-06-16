# merge-n-multi-row-normalization

## 适用场景
- **算子类型**：normalization,layernorm,rmsnorm
- **Shape 范围**：N<=1024, M 任意
- **硬件约束**：Ascend910B, UB 容量需满足 row_factor*N*sizeof(float)*3 + 预广播 buffer

## 核心思路
对于逐行归一化算子（LayerNorm/RMSNorm/AddLayerNorm），当 N ≤ 1024 时，将 row_factor=8 行合并处理。关键步骤：1) Init 时预广播 gamma/beta 到 [8,N]；2) 使用 ReduceSum 向量化归约替代标量 for 循环；3) mean/inv_rms 动态广播复用单一 broadcast buffer；4) 8 行同时 normalize+affine。此模式将逐行参数加载开销从 O(rows) 降至 O(rows/8)，并将标量归约加速为向量归约

## 代码模板
```python

```

## 已知限制
- 

## 来源
- 首次发现于：addlayernorm 优化 2026-06-06
- 相关 Case：case_addlayernorm_sess_20260606_01_0

## 状态
待验证
