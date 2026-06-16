# datacopypad-for-cann850

## 适用场景
- **算子类型**：all
- **Shape 范围**：any
- **硬件约束**：CANN 8.5.0 on Ascend 910B

## 核心思路
In CANN 8.5.0, plain AscendC::DataCopy for GM↔UB transfer produces all-zero output. Use DataCopyPad with explicit DataCopyExtParams{1, count*sizeof(T), 0, 0, 0} and DataCopyPadExtParams{true, 0, 0, 0} instead. Wrapper functions LoadGmToUb/StoreUbToGm provide clean abstraction.

## 代码模板
```python
template <typename T>
__aicore__ inline void LoadGmToUb(AscendC::LocalTensor<T> &dst, AscendC::GlobalTensor<T> src, uint32_t count) {
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}
```

## 已知限制
- May not be needed in other CANN versions

## 来源
- 首次发现于：masked_softmax_with_rel_pos_bias 优化 2026-06-14
- 相关 Case：case_masked_softmax_with_rel_pos_bias_sess_20260614_a3f2_0

## 状态
待验证
