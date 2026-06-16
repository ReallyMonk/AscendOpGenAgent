# tbuf-direct-copy-row-wise

## 适用场景
- **算子类型**：add_rms_norm,add_layer_norm,rms_norm,layer_norm,gelu,relu
- **Shape 范围**：M=(1K,32K), N=(512,8192), dtype=fp16/bf16
- **硬件约束**：Ascend 910B, UB=192KB per core. Must ensure total TBuf allocation (xBuf+rBuf+yBuf+castX+castR+comp+reduceTmp) fits within UB for the target N.

## 核心思路
For element-wise fusion operators that process rows independently, replacing TQue (VECIN/VECOUT with AllocTensor/EnQue/DeQue/FreeTensor) with TBuf (VECCALC with direct DataCopy) eliminates massive per-row API call overhead. For 4096 rows, this removes ~49K TQue API calls, resulting in 20-25x speedup. Combined with ReduceSum for vectorized reduction, this pushes fusion operators from 0.1x to 2.5x vs PyTorch.

## 代码模板
```python
// TBuf for direct GM↔UB copy
pipe_->InitBuffer(xBuf_, t_.N * sizeof(T));
pipe_->InitBuffer(rBuf_, t_.N * sizeof(T));
pipe_->InitBuffer(yBuf_, t_.N * sizeof(T));
// In ProcessRow:
auto xL = xBuf_.Get<T>();
AscendC::DataCopy(xL, xGM_[ri * t_.N], t_.N);
// ... compute ...
auto yL = yBuf_.Get<T>();
AscendC::DataCopy(yGM_[ri * t_.N], yL, t_.N);
```

## 已知限制
- UB size limits row batch size. For N>8192, may not fit with fp32 cast buffers. TQue still preferred when pipelining across blocks is needed (e.g., matmul).

## 来源
- 首次发现于：add_rms_norm 优化 2026-06-15
- 相关 Case：case_add_rms_norm_sess_20260615_a1b2_1

## 状态
待验证
