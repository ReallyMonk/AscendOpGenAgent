# python-ktile-split-kernel

## 适用场景
- **算子类型**：matmul,gemm,conv
- **Shape 范围**：K > blockK (128), 任意M,N
- **硬件约束**：Ascend 910B4, 需blockK对齐Cube单元(128)

## 核心思路
当K维度超过单次内核blockK限制时，在Python/Host端将K分割为多个blockK大小的tile，逐tile调用内核，用fp32累积中间结果。避免C++端复杂padding逻辑和流同步问题，降低内核复杂度。

## 代码模板
```python

```

## 已知限制
- 

## 来源
- 首次发现于：mat_mul_v3 优化 2026-06-16
- 相关 Case：case_mat_mul_v3_sess_20260616_7a3f_0

## 状态
待验证
