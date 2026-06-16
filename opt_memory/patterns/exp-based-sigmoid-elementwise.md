# exp-based-sigmoid-elementwise

## 适用场景
- **算子类型**：activation, elementwise, fusion
- **Shape 范围**：M=1024-4096, inter=2048-29568, fp16
- **硬件约束**：Ascend 910B, UB >= 192KB, 支持 half 类型的 Exp/Reciprocal 向量指令

## 核心思路
对于需要 sigmoid/SiLU 激活的 element-wise 融合算子，使用 Exp-based sigmoid (1/(1+exp(-x))) 替代 tanh-based sigmoid，配合 fp16 直接计算，消除 Cast half↔float 开销。将 chunk size 从 64 增加到 256，充分利用 256-wide 向量单元。工作缓冲区使用 half dtype 降低 UB 压力。

## 代码模板
```python
// Exp-based SiLU chunk processing
for (int start = 0; start < halfN; start += 256) {
    // Load x1, x2 directly in fp16 (no Cast)
    AscendC::DataCopy(x1Wk, xInL_[start], chunk);
    AscendC::DataCopy(x2Wk, xInL_[halfN + start], chunk);
    // Compute: sigmoid = 1/(1+exp(-x2))
    AscendC::Muls(negWk, x2Wk, T(-1.0), chunk);
    AscendC::Exp(expWk, negWk, chunk);
    AscendC::Adds(denomWk, expWk, T(1.0), chunk);
    AscendC::Reciprocal(sigWk, denomWk, chunk);
    // SiLU: silu = x2 * sigmoid
    AscendC::Mul(siluWk, x2Wk, sigWk, chunk);
    // Output: y = x1 * silu
    AscendC::Mul(outWk, x1Wk, siluWk, chunk);
}
```

## 已知限制
- 小 shape (M<1024) 时 kernel launch overhead 抵消优化收益。需要 AscendC Exp/Reciprocal 支持 half 类型。部分旧版本 CANN 可能不支持 half Reciprocal。

## 来源
- 首次发现于：swi_glu 优化 2026-06-14
- 相关 Case：case_swi_glu_sess_20260614_4827_0, case_swiglu_sess_20260122_4827_0

## 状态
待验证
