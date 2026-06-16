# shape-adaptive-elementwise-tiling

## 适用场景
- **算子类型**：activation,elementwise,gelu,swish,silu,relu,tanh
- **Shape 范围**：M: 任意, N: 1024-32768 (覆盖 LLM intermediate sizes)
- **硬件约束**：Ascend 910B, UB 约 192KB, TILE_N_MAX=8192 确保双行+双buffer不超 UB

## 核心思路
对 element-wise 算子根据 N 维度大小自动选择最优执行路径：(1) N≤4096 时双行批处理增加计算强度；(2) 4096<N≤8192 时标准单行处理；(3) N>8192 时沿 N 轴分块 (TILE_N_MAX) 突破 UB 容量限制。动态 UB buffer 分配：bufN=min(N,TILE_N_MAX), bufferSize=bufN*batchSize*sizeof(T)

## 代码模板
```python
// Shape-adaptive pattern skeleton:
static constexpr int32_t TILE_N_MAX = 8192;
bool needTiling_ = (tiling_.N > TILE_N_MAX);
int32_t bufN = needTiling_ ? TILE_N_MAX : tiling_.N;
int32_t batchSize = (bufN <= 4096) ? 2 : 1;
// UB buffer: bufN * batchSize * sizeof(T)
// Process: for rows r+=batchSize { if(needTiling_) ProcessRowTiled(r,batchSize); else ProcessRowBatch(r,batchSize); }
```

## 已知限制
- 批次处理要求相邻行在 GM 中连续存储（row-major）；对于极窄 N (<256) 双行收益递减；不支持跨 M 维度的 tiling（仅沿 N 分块）

## 来源
- 首次发现于：gelu 优化 2026-06-13
- 相关 Case：case_gelu_sess_20260613_01_0

## 状态
待验证
