# vectorized-gather-data-copy

## 适用场景
- **算子类型**：gather,scatter,embedding,index_select
- **Shape 范围**：N>=32, any M/rows, num_indices<=8192
- **硬件约束**：Ascend910B, UB 256KB, TQUE depth=1, TILE_DIM<=256 for fp16 (512B per tile)

## 核心思路
对于纯数据搬运算子 (gather/scatter/embedding/index_select)，将逐元素标量 GetValue/SetValue 循环替换为 TQue VECIN→Adds(0)→VECOUT 向量化 DataCopy 管线。每行仅需 2*ceil(N/TILE_DIM) 次向量事务替代 2*N 次标量事务，加速比正比于 N。TILE_DIM=256 保证任意 dim 值 UB 安全

## 代码模板
```python
// Vectorized gather via TQue VECIN→VECOUT pattern
// Replaces scalar GetValue/SetValue per-element loop
for (int32_t d = 0; d < N; d += TILE_DIM) {
    int32_t dLen = min(TILE_DIM, N - d);
    
    // VECIN: GM→UB
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, srcGM[srcIdx * N + d], dLen);
    inQueue.EnQue(inLocal);
    
    // Vector compute: identity pass-through
    LocalTensor<T> inData = inQueue.DeQue<T>();
    LocalTensor<T> outData = outQueue.AllocTensor<T>();
    Adds(outData, inData, T(0), dLen);
    outQueue.EnQue(outData);
    inQueue.FreeTensor(inData);
    
    // VECOUT: UB→GM
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopy(dstGM[dstIdx * N + d], outLocal, dLen);
    outQueue.FreeTensor(outLocal);
}
```

## 已知限制
- 框架固定开销 ~0.15ms (pybind11+AscendC launch) 对小数据量占主导; 多行批处理无额外收益 (固定开销不随 batch 减少); 对极大数据量 (N>16K) 考虑多级 tiling

## 来源
- 首次发现于：gather_v2 优化 2026-06-13
- 相关 Case：case_gather_v2_sess_20260613_7241_0

## 状态
待验证
