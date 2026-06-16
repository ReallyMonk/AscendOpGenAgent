# reducesum-separate-dst

## 适用场景
- **算子类型**：normalization, reduction, elementwise
- **Shape 范围**：N>=256 的任意规约维度
- **硬件约束**：Ascend 910B AIV; require UB space for: 1 src copy buffer (N floats) + 1 tmp buffer (N floats) + 1 small dst buffer (256 floats)

## 核心思路
AscendC ReduceSum 必须使用独立的 dst buffer（不同于 src），避免二元树规约中间结果覆盖源数据。先 DataCopy 将数据拷贝到工作缓冲区作为 src，再用小结果缓冲区（如 256 floats）作为 dst。ReduceSum(dst, src_copy, tmpBuf, N) → 从 dst(0) 读取标量结果。

## 代码模板
```python
// Copy data to work buffer for ReduceSum
AscendC::DataCopy(workBuf, srcBuf, N);
AscendC::PipeBarrier<PIPE_MTE2>();
AscendC::ReduceSum<float>(dstSmall, workBuf, tmpBuf, N);
float result = dstSmall(0) * scalar;
```

## 已知限制
- 需要额外的 UB 空间：workBuf(N*4B) + tmpBuf(N*4B) + dstSmall(1KB)。对于超长向量 (N>16K)，UB 可能不足。

## 来源
- 首次发现于：add_layer_norm 优化 2025-06-16
- 相关 Case：case_add_layer_norm_sess_20260616_2847_1

## 状态
待验证
