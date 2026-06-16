# vectorized-softmax-reduce-pipeline

## 适用场景
- **算子类型**：softmax,masked_softmax,scaled_masked_softmax,attention
- **Shape 范围**：S in [256, 4096]; rowFactor=4 for N<=1024, rowFactor=1 for N>=2048
- **硬件约束**：Ascend910B, UB~192KB; rowFactor*N*sizeof(float32)*3 + rowFactor*N + 2*N*sizeof(float32) < UB

## 核心思路
Softmax融合算子优化：将scale+mask+max+sum四个标量O(N)循环替换为向量化API（Muls+CompareScalar+Select+ReduceMax+ReduceSum），延迟从O(N)降至O(log N)。配合merge-n多行合并（rowFactor=4）摊销GM-UB事务开销，UB预算允许时有效提升吞吐。双路径设计按N维度自动选择merge-n(N<=1024)或single-row(N>=2048)

## 代码模板
```python
// Scale: AscendC::Muls(qkFp32, qkFp32, scale, N);
// Mask: AscendC::CompareScalar(cmpMask, maskFp32, 0.0f, CMPMODE::EQ, N);
//       AscendC::Select(qkFp32, cmpMask, negInf, qkFp32, VSEL_CMPMASK_SPR, N);
// Max:  AscendC::ReduceMax(perMax, qkFp32, reduceWork, N);
// Sub:  AscendC::Adds(qkFp32, qkFp32, -perMax, N);
// Exp:  AscendC::Exp(qkFp32, qkFp32, N);
// Sum:  AscendC::ReduceSum(perSum, qkFp32, reduceWork, N);
// Div:  AscendC::Muls(qkFp32, qkFp32, 1.0f/perSum, N);
```

## 已知限制
- Precision: AscendC::Exp与torch.exp存在~7%元素级微小偏差(max diff ~0.017); 极小shape(S<256)时kernel launch开销抵消收益; bfloat16未测试

## 来源
- 首次发现于：scaled_masked_softmax_v2 优化 2026-06-15
- 相关 Case：case_scaled_masked_softmax_v2_sess_20260615_8173_0

## 状态
待验证
