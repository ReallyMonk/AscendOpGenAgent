# conv2d-nhw-tiling-multi-core

## 适用场景
- **算子类型**：conv
- **Shape 范围**：N=[1,16], Co=[64,2048], H=[7,224]
- **硬件约束**：Ascend 910B 32核，blockM=128最优，sub-block数目决定向量宽度

## 核心思路
Conv2d 多核并行策略：按输出维度 (N*Co*H) 切分 M，每个核心处理 blockM 行输出，sub-block 并行处理向量数据。

## 代码模板
```python
// Block tiling: split M = N*Co*H across cores
int32_t M = N * Co * H;
int32_t blockM = 128;
int32_t numBlocks = (M + blockM - 1) / blockM;
// Core assignment
const int ci = GetBlockIdx() / GetSubBlockNum();
const int si = GetSubBlockIdx();
```

## 已知限制
- 

## 来源
- 首次发现于：conv2d_v2 优化 2026-06-13
- 相关 Case：

## 状态
待验证
