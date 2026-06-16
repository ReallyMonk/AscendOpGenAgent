# vectorized-scalar-reduction

## 适用场景
- **算子类型**：pooling,reduction,elementwise
- **Shape 范围**：Any shape with regular access pattern; benefits increase with larger spatial dimensions
- **硬件约束**：Ascend 910B Vector unit; 128-bit vector width for float32

## 核心思路
Vectorized SIMD implementation for pooling/reduction ops: replace scalar GetValue/SetValue with vector Load/Store, use UB buffers for accumulation, apply vector Add/Muls operations. Achieves ~8x speedup for float32 on Vector unit.

## 代码模板
```python
// For memory-bound reduction/pooling operations:
// 1. Replace scalar GetValue/SetValue with vector Load/Store
// 2. Use UB buffers for accumulation staging
// 3. Apply vector Add/Muls for SIMD processing
// Example: 8-element SIMD for float32 = 128-bit vectors
```

## 已知限制
- Only works when data access pattern is regular and aligned; may need scalar fallback for tail elements

## 来源
- 首次发现于：avgpool2d 优化 2026-06-13
- 相关 Case：case_avgpool2d_sess_20260613_ab12_0

## 状态
待验证
