# chunk-enlargement-gelu-fusion

## 适用场景
- **算子类型**：activation,fusion,elementwise
- **Shape 范围**：M=[1024,4096], N=[4096,59136]
- **硬件约束**：

## 核心思路
对Ascend 910B上memory-bound逐元素融合算子，将向量化chunk size从64扩大到512可将循环迭代次数减少8倍，显著减少指令发射开销。对于GELU/GLU类融合算子，x2Wk在x2^3计算中被覆写，需要额外备份缓冲区(wX2_)以避免从UB重载x2

## 代码模板
```python

```

## 已知限制
- 超大N(>65536)需要N-tiling配合；小shape受益有限(受kernel launch开销主导)

## 来源
- 首次发现于：ge_glu_v2 优化 2026-06-15
- 相关 Case：case_ge_glu_v2_sess_20260615_8472_0

## 状态
待验证
