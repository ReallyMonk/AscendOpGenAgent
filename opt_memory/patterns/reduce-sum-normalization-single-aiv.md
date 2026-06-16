# reduce-sum-normalization-single-aiv

## 适用场景
- **算子类型**：rms_norm,layer_norm,add_layer_norm,add_rms_norm,normalization
- **Shape 范围**：N=[256..8192], M任意
- **硬件约束**：Ascend910B, MIX_AIC_1_1模式, UB需容纳2*N*sizeof(fp32)+N*sizeof(fp16)+2*N*sizeof(fp16)+256*sizeof(fp32)字节

## 核心思路
在Vector-only normalization类算子中，使用ReduceSum硬件矢量归约替代标量for循环进行逐行sum/sumSq计算。配合KERNEL_TYPE_MIX_AIC_1_1避免双AIV导致UB翻倍溢出。精简TBuf至2个fp32计算缓冲区+1个fp16 I/O缓冲区。Gamma/beta per-row cast替代预加载节省UB。

## 代码模板
```python

```

## 已知限制
- 小M场景(<1024)因kernel launch开销可能略慢于PyTorch。N>16384需splitd路径额外N-tiling支持。

## 来源
- 首次发现于：add_layer_norm 优化 2025-06-15
- 相关 Case：case_add_layer_norm_sess_20260615_a3f8_0,case_addlayernorm_sess_20250613_1742_0,case_add_layer_norm_sess_20260613_0001_0

## 状态
待验证
