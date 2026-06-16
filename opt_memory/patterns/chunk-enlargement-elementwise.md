# chunk-enlargement-elementwise

## 适用场景
- **算子类型**：activation, elementwise
- **Shape 范围**：Any shape with large inner dimension (N >= 4096)
- **硬件约束**：Ascend 910B series, UB size ~192KB. Chunk size * sizeof(float) * num_tbufs must fit in UB alongside VecIn/VecOut queues.

## 核心思路
For memory-bound element-wise operators on Ascend 910B, enlarge the compute chunk size (e.g., 64→512) to reduce loop iteration count and improve vector unit utilization. Fewer iterations means less loop overhead and more data per vector instruction batch, reducing the impact of instruction dispatch latency.

## 代码模板
```python

```

## 已知限制
- Small shapes dominated by kernel launch overhead show minimal benefit. Chunk too large may exceed UB budget when combined with N-tiling.

## 来源
- 首次发现于：clipped_swiglu 优化 2026-06-15
- 相关 Case：case_clipped_swiglu_sess_20260615_2801_0, case_clipped_swiglu_sess_20260614_4827_0

## 状态
待验证
