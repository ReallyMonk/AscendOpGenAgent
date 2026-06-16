# pooling-nhwc-channel-vectorization

## 适用场景
- **算子类型**：pooling,avg_pool2d,max_pool2d
- **Shape 范围**：small to medium (M < 10000, C up to 512), k=2, s=2, p=0
- **硬件约束**：Ascend910B, CANN 8.5.0, 纯 AIV kernel 类型（非 MIX_AIC），Max cores 限制在 32 以内

## 核心思路
池化算子将 NCHW layout 转换为 NHWC，使 C 个通道在内存中连续。利用 DataCopy(count=C) 一次加载整行通道数据，再通过 Add/Muls 向量化指令完成 2×2 窗口求和与平均。使用纯 AIV kernel 类型避免 MIX_AIC 模式下的多 sub-block 竞争问题。TQue depth=1 管理 UB 缓冲区生命周期，PipeBarrier 保证 GM↔UB 同步。

## 代码模板
```python
TQue<VECIN,1> + DataCopy(count=C) + Add/Muls(count=C) + PipeBarrier
```

## 已知限制
- 仅支持 k=2,s=2,p=0, float32；小 shape 下 launch overhead 和 NCHW↔NHWC permute 开销占比大，性能约为 PyTorch 的 0.5x；MIX_AIC 模式下多 core 可能存在精度问题

## 来源
- 首次发现于：avg_pool2d 优化 2026-06-16
- 相关 Case：case_avg_pool2d_sess_20260616_4823_0,case_avgpool2d_sess_20250711_4823_0

## 状态
待验证
