# reduce-sum-max-replace-dual-scalar-loops

## 适用场景
- **算子类型**：add_rms_norm_quant,add_rms_norm_dynamic_quant,quantization,normalization
- **Shape 范围**：N≤4096 for quant norms (UB constraint from dual Reduce buffers)
- **硬件约束**：

## 核心思路
将 quantized norm 算子中的双标量循环(sumSq + rowMax)替换为 ReduceSum AR + ReduceMax AR + Abs 组合。先用 Abs 取绝对值，再用 ReduceMax AR 沿N轴归约最大值。两个 Reduce 操作共享同一个 reduceTmp buffer，显著减少 UB 占用

## 代码模板
```python

```

## 已知限制
- 

## 来源
- 首次发现于：add_rms_norm_quant 优化 2026-06-12
- 相关 Case：case_addlayernorm_sess_20260611_8k3f_0

## 状态
待验证
