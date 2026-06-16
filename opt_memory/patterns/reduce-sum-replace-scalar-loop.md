# reduce-sum-replace-scalar-loop

## 适用场景
- **算子类型**：rms_norm,add_rms_norm,layer_norm,add_layer_norm,normalization
- **Shape 范围**：N≤8192 for simple norms, N≤4096 for complex norms with extra parameters
- **硬件约束**：

## 核心思路
将 normalization 类算子中逐元素标量累加循环替换为 AscendC::ReduceSum AR 模式。标量循环 O(N) → ReduceSum O(log N)。适用于 add_rms_norm, rms_norm, add_layer_norm, layer_norm 等算子。

## 代码模板
```python

```

## 已知限制
- 

## 来源
- 首次发现于：add_rms_norm 优化 2026-06-13
- 相关 Case：case_addlayernorm_sess_20250613_1742_0,case_addrmsnorm_sess_20250121_8742_0

## 状态
待验证
