# fixpipe-safety-with-getphyaddr

## 适用场景
- **算子类型**：matmul, elementwise, reduction, normalization
- **Shape 范围**：任意shape，Fixpipe输出场景
- **硬件约束**：Ascend 910B系列，CANN 8.5.0+，Fixpipe输出需要量化时

## 核心思路
使用GetPhyAddr获取物理地址后通过SetGlobalBuffer创建带显式buffer大小的GlobalTensor，确保Fixpipe stride写入不会超出分配内存。配合输出tensor分配时预留额外安全行(blockM)，彻底避免DDR地址越界。同时设置FixpipeParamsV220.quantPre=F322F16实现float32→float16的正确量化。

## 代码模板
```python
// Fixpipe safety pattern:
GlobalTensor<half> cBlock;
cBlock.SetGlobalBuffer(cGM_.GetPhyAddr(offset), blockM * N);
// ... compute ...
AscendC::FixpipeParamsV220 fixParams(blockN, blockM, blockM, dstStride, false);
fixParams.quantPre = QuantMode_t::F322F16;
AscendC::Fixpipe(cBlock, l0cTensor, fixParams);

// Python side output allocation:
auto c_pad = at::empty({M_total + BLOCK_M, N_pad}, options);
```

## 已知限制
- 需要额外blockM行内存开销；仅适用于Fixpipe输出场景；L0C必须是float32类型

## 来源
- 首次发现于：mat_mul_v3 优化 2026-06-16
- 相关 Case：case_mat_mul_v3_sess_20260616_a75d_0, case_batch_mat_mul_v3_sess_20260616_a75d_0

## 状态
待验证
