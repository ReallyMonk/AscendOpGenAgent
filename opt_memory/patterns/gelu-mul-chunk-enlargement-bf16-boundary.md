# gelu-mul-chunk-enlargement-bf16-boundary

## 适用场景
- **算子类型**：activation,fusion,elementwise
- **Shape 范围**：M=1024~4096, inter=2048~29568
- **硬件约束**：

## 核心思路
对 elementwise fusion 算子，将内层向量化 chunk 从默认 64 增大到 256-512，可减少循环迭代次数并提升 SIMD 利用率。对于 bfloat16 支持，当 AscendC::Cast 有兼容性问题时，可在 pybind11 边界做 bf16↔fp16 转换后复用 fp16 快速内核。

## 代码模板
```python

```

## 已知限制
- chunk 过大可能增加寄存器压力；需确保 UB workspace 足够容纳 chunk 的计算缓冲区

## 来源
- 首次发现于：gelu_mul 优化 2026-06-15
- 相关 Case：case_gelu_mul_sess_20260615_8842_0,case_gelumul_sess_20250614_8842_0

## 状态
待验证
