# prefuse-inputs-single-vecin

## 适用场景
- **算子类型**：attention,elementwise,activation
- **Shape 范围**：any
- **硬件约束**：CANN 8.5.0 multi-VECIN-queue limitation

## 核心思路
When CANN 8.5.0 multi-VECIN-queue fails with TQue depth assertion, pre-fuse multiple inputs on the Python host side into a single tensor. For softmax with mask+bias: compute qk+bias on host, apply mask as -inf for masked positions, pass single fused input to kernel. Reduces VECIN queues from 3 to 1.

## 代码模板
```python

```

## 已知限制
- 

## 来源
- 首次发现于：masked_softmax_with_rel_pos_bias 优化 2026-06-14
- 相关 Case：case_masked_softmax_with_rel_pos_bias_sess_20260614_a3f2_0

## 状态
待验证
