# large-chunk-vectorized-reduction

## 适用场景
- **算子类型**：loss,normalization,reduction,activation
- **Shape 范围**：N≥32768, 任意M; 需要确保CHUNK*sizeof(float)*3 ≤ UB(~192KB)
- **硬件约束**：Ascend 910B系列; UB约192KB; CHUNK=2048时fp32 chunk+workBuf约20KB, 在UB限制内

## 核心思路
对于沿大N维（≥32K）做reduction的算子，将chunk从64增大到2048以减少循环迭代次数（~32x），同时使用AscendC基础API ReduceMax/ReduceSum替代标量循环进行chunk内归约。这大幅减少GM↔UB搬运频率并利用Vector单元硬件加速归约，典型加速2-3x。

## 代码模板
```python
constexpr int32_t CHUNK=2048; // ReduceMax: dst,src,workBuf,count; ReduceSum: dst,src,workBuf,count; ProcessRow: for chunk in V: load→cast fp32→Adds(-max)→Exp→ReduceSum
```

## 已知限制
- 小shape(<32K)收益递减，因为kernel launch overhead占主导; CHUNK>4096可能超出UB

## 来源
- 首次发现于：cross_entropy_loss 优化 2026-06-15
- 相关 Case：case_cross_entropy_loss_sess_20260615_7a3f_1

## 状态
待验证
