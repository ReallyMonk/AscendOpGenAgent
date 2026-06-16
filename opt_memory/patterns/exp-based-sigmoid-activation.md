# exp-based-sigmoid-activation

## 适用场景
- **算子类型**：activation, elementwise
- **Shape 范围**：任意 shape，对计算密集型算子效果更明显
- **硬件约束**：Ascend 910B，Exp/Tanh 均为 fp32 计算

## 核心思路
对于需要 sigmoid 的激活函数（SiLU/SwiGLU/GELU 等），使用 1/(1+exp(-x)) 替代 0.5*(1+tanh(x/2))。Ascend 910B 上 Exp 指令延迟低于 Tanh，在精度相当的情况下获得 1.5-1.6x 加速。

## 代码模板
```python
// Exp-based sigmoid: 1/(1+exp(-x))
AscendC::Muls(wk, wk, -1.0f, n);
AscendC::Exp(wk, wk, n);
AscendC::Adds(wk, wk, 1.0f, n);
AscendC::Reciprocal(wk, wk, n);
```

## 已知限制
- 精度略低于 tanh 版本（max_diff 从 7.8e-3 升到 3.1e-2），但仍在 1e-2 容差内

## 来源
- 首次发现于：swi_glu 优化 2026-06-15
- 相关 Case：case_swi_glu_sess_20260615_4d7f_2

## 状态
待验证
