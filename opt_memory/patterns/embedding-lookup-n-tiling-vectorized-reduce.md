# embedding-lookup-n-tiling-vectorized-reduce

## 适用场景
- **算子类型**：embedding_bag,embedding,nll_loss,gather_reduce
- **Shape 范围**：N>=128, M任意, V任意
- **硬件约束**：Ascend910B, UB 256KB, TILE_N≤256 for fp32

## 核心思路
对embedding lookup+reduce类算子进行N维度tiling(TILE_N=128)，将逐元素标量归约循环替换为向量DataCopy/Add/Muls/Cast链，消除标量指令开销。每个N-tile内使用TBuf双缓冲避免UB溢出

## 代码模板
```python
// N-tiling with vectorized gather-reduce for embedding lookup ops
for (col = 0; col < N; col += TILE_N) {
    count = min(TILE_N, N - col);
    Dups(acc, 0.0f, count);  // zero accumulator
    for (i = start; i < end; ++i) {
        id = idx[i];
        DataCopy(buf, w[id*N + col], count);  // vector load
        Cast(bufF32, buf, count);              // type cast
        Add(acc, acc, bufF32, count);          // vector accumulate
    }
    Muls(acc, acc, inv, count);                // normalize
    DataCopy(y[bag*N + col], acc, count);     // vector store
}
```

## 已知限制
- bag内indices数量过大时向量化收益递减(gather随机访存仍占主导); N<128时tiling无意义

## 来源
- 首次发现于：embedding_bag 优化 2025-12-16
- 相关 Case：case_embeddingbag_sess_20251216_01_1

## 状态
待验证
