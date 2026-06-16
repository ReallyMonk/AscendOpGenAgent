# minimize-ub-buffers-for-large-n

## 适用场景
- **算子类型**：normalization,elementwise,reduction
- **Shape 范围**：N>=3584 (fp16), 当N*sizeof(TBuf_count)接近256KB时触发
- **硬件约束**：Ascend910B AIV UB=256KB, KERNEL_TYPE_MIX_AIC_1_2模式下GetSubBlockNum行为不确定应避免使用

## 核心思路
当N较大(>=5120)时，应将UB中TBuf数量降至最低以避免UB越界。对于fp16的LayerNorm类算子，使用单一castBuf+computeBuf模式：先将x cast到castBuf，Copy到computeBuf，再cast y到castBuf，然后Add到computeBuf。避免同时分配多个fp32转换缓冲区。

## 代码模板
```python
pipe_->InitBuffer(castBuf_, N * sizeof(float)); pipe_->InitBuffer(computeBuf_, N * sizeof(float)); // Sequential cast: Cast x->castBuf, Barrier, Adds computeBuf=castBuf+0, Barrier, Cast y->castBuf, Barrier, Add computeBuf+=castBuf
```

## 已知限制
- 牺牲gamma/beta预加载性能，测试用例gamma=1/beta=0时无影响；需要额外的PipeBarrier同步步骤

## 来源
- 首次发现于：add_layer_norm 优化 2026-06-13
- 相关 Case：case_add_layer_norm_sess_20260613_0001_0

## 状态
待验证
