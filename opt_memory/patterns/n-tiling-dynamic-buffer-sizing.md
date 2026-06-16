# n-tiling-dynamic-buffer-sizing

## 适用场景
- **算子类型**：activation,elementwise,normalization
- **Shape 范围**：N > 8192 (fp16/bf16), N > 4096 (fp32)
- **硬件约束**：Ascend910B, UB 256KB

## 核心思路
当算子行宽 N 超过 UB 容量时，沿 N 方向按固定 tile 大小分块处理。在 Init 阶段根据 N 动态选择 buffer 大小（bufN = min(N, TILE_N_MAX)），Process 阶段通过 needTiling_ 标志分发到 tiled 路径。每个 N-tile 独立完成 CopyIn→Compute→CopyOut 循环。Compute 函数接收可变 count 参数。TILE_N_MAX 通过 UB 预算公式计算：TILE_N_MAX ≤ UB_SIZE / (2*sizeof(T) + 2*sizeof(float)) ≈ 8192 for fp16。

## 代码模板
```python
// In Init: determine if N-tiling needed
needTiling_ = (tiling_.N > TILE_N_MAX);
int32_t bufN = needTiling_ ? TILE_N_MAX : tiling_.N;
// Allocate all buffers at bufN size
pipe_->InitBuffer(xInQ_, 1, bufN * sizeof(T));
// In Process: dispatch to tiled or full-row path
if (needTiling_) { ProcessRowTiled(ri); } else { ProcessRow(ri); }
// Tiled path loops over N in TILE_N_MAX chunks
for (int32_t col = 0; col < N; col += TILE_N_MAX) {
    int32_t count = min(TILE_N_MAX, N - col);
    CopyInTile(ri, col, count);
    ComputeTile(ri, count);  // uses 'count' not 'tiling_.N'
    CopyOutTile(ri, col, count);
}
```

## 已知限制
- tile 数量增加时性能线性下降。对超大 N（> 100K），建议配合 double buffer 减少 tile 间等待。

## 来源
- 首次发现于：gelu 优化 2026-06-10
- 相关 Case：case_gelu_sess_20260610_9895_1

## 状态
待验证
