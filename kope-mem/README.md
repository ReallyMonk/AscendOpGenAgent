# Kope-Mem for AscendOpGenAgent

AscendC 算子优化记忆框架，专为 AscendOpGenAgent 优化。

## 功能

- **Journal 轨**：Markdown 叙事日志，记录优化历程
- **Case 轨**：JSON 结构化记录，积累可查询的优化经验
- **Baseline 管理**：性能基线记录与对比
- **Pattern 库**：跨算子可复用的优化模式

## 目录结构

```
opt_memory/                    # 根目录（自动创建）
├── operators/{算子名}/
│   ├── YYYY-MM-DD.md        # Journal 日志
│   └── cases/
│       ├── index.json         # Case 索引
│       └── case_*.json       # Case 文件
├── patterns/                   # 优化模式库
└── baselines/
    └── index.json             # 基线索引
```

## 使用方式

### 在 AscendOpGenAgent 中使用

本目录提供 `SKILL.md`，Agent 可通过 bash 命令调用：

```bash
# 追加 Journal 记录
cat >> opt_memory/operators/{op}/$(date +%Y-%m-%d).md << 'EOF'
## Iter 1: 核心分核优化
- **改动**：扩展至 40 核
- **性能**：1.25 TFLOPS，利用率 90%
- **状态**：✅通过
EOF

# 创建 Case
cat > opt_memory/operators/{op}/cases/case_{id}.json << 'EOF'
{json}
EOF
```

## 初始化

首次使用时，自动创建目录结构：

```bash
mkdir -p opt_memory/operators opt_memory/patterns opt_memory/baselines
```

## 与 ascend-kernel-optimizer 集成

在 ascend-kernel-optimizer.md 的以下阶段使用：

- **Phase 4 (性能分析)**：记录 baseline
- **Phase 5 (代码差异学习)**：创建 Case
- **Phase 6 (Trace 记录)**：追加 Journal
- **Phase 7 (优化记忆)**：整理 pattern
