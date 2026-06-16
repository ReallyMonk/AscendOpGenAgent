# fp16-inplace-on-tque-for-precision

## 适用场景
- **算子类型**：add_rms_norm_quant,add_rms_norm,rms_norm,add_layer_norm,layer_norm,normalization
- **Shape 范围**：N≤8192
- **硬件约束**：Ascend910B; CANN 8.5.0; KERNEL_TYPE_MIX_AIC_1_1 (AIV-only)

## 核心思路
在AscendC TQue LocalTensor上直接做in-place fp16算术运算（Add/Mul），替代先Cast到fp32再运算的模式，以匹配PyTorch reference的fp16精度路径。配合sqrt+1/x替代Rsqrt来进一步消除rsqrt硬件实现的精度差异。适用于normalization类算子的add+square步骤。

## 代码模板
```python
// fp16 in-place on TQue LocalTensor
xInQ_.DeQue<T>(xL_);
rInQ_.DeQue<T>(rL_);
AscendC::Add(xL_, xL_, rL_, N); // xL_ = x+r (fp16)
AscendC::Mul(rL_, xL_, xL_, N); // rL_ = (x+r)^2 (fp16)
// Then cast to fp32 for reduction
AscendC::Cast(bufFp32, rL_, CAST_NONE, N);
// sqrt + reciprocal for rms computation
float rms = sqrt(sumSq * invN + eps);
float invRms = 1.0f / rms;
```

## 已知限制
- TQue buffer must be depth-1 (single-buffered) for in-place modification to work safely. 99.61% precision match achievable; remaining 0.39% from fp32 div vs mul-by-reciprocal difference is fundamental.

## 来源
- 首次发现于：add_rms_norm_quant 优化 2026-06-16
- 相关 Case：case_add_rms_norm_quant_sess_20260616_4821_0

## 状态
待验证
