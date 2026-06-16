# fp16-direct-compute-elementwise

## 适用场景
- **算子类型**：activation, elementwise
- **Shape 范围**：
- **硬件约束**：Ascend 910B series, half (fp16) precision sufficient for element-wise activation functions

## 核心思路
For memory-bound element-wise operators on Ascend NPU, compute directly in fp16 instead of casting to fp32 and back. This eliminates 2 vector Cast operations per chunk, reducing instruction count significantly. Use Adds(dst, src, 0) for same-type vector copy instead of Cast (which doesn't support same-type). Prefer Exp-based activation functions (sigmoid/silu) over tanh-based approximations for fewer operations. Use larger chunk sizes (256 vs 64/128) to improve vector unit utilization.

## 代码模板
```python

```

## 已知限制
- fp16 precision may not be sufficient for all operators (e.g., those requiring high dynamic range). Small shapes still dominated by kernel launch overhead (~0.13ms).

## 来源
- 首次发现于：clipped_swiglu 优化 2026-06-12
- 相关 Case：case_clipped_swiglu_sess_20260612_fp16direct_0

## 状态
待验证
