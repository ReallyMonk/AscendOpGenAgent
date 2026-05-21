# 优化记忆约定

参与 AscendC 算子优化的 Agent 共同遵守以下约定。

## 数据目录

| 目录 | 管理层 | 用途 |
|------|--------|------|
| `opt_memory/operators/{算子名}/` | 自定义层 | 当前算子优化日志（Journal md + Case json） |
| `opt_memory/patterns/` | 自定义层 | 跨算子可复用模式 |
| `opt_memory/baselines/index.json` | 自定义层 | 全局性能基线索引 |
| `opt_memory/MEMORY.md` | ReMe 层 | 长期通用记忆 |
| `opt_memory/memory/` | ReMe 层 | 每日自动摘要 |
| `opt_memory/dialog/` | ReMe 层 | 原始对话归档 |
| `opt_memory/tool_result/` | ReMe 层 | 工具输出缓存 |

## 写约定：何时记录

| 时机 | Journal 轨 | Case 轨 | 命令 |
|------|-----------|---------|------|
| 每次优化迭代后 | ✅ 叙事记录 | 可选 | `uv run kope-mem journal add ...` |
| 关键优化步骤（需结构化查询） | 可选 | ✅ 结构化记录 | `uv run kope-mem case append ...` |
| 完成一轮评测 | ✅ 汇总结果 | ✅ 导入完整 Case | `uv run kope-mem case add --json-path ...` |
| 上下文接近 80k tokens | ✅ 压缩 | — | `uv run kope-mem compact --operator ...` |
| 发现可迁移模式 | — | ✅ 模式入库 | `uv run kope-mem pattern add ...` |

## 读约定：何时检索

| 时机 | 命令 |
|------|------|
| 开始优化新算子 | `uv run kope-mem case list --operator {算子名}` |
| 选择优化策略前 | `uv run kope-mem search --query "{策略描述}"` |
| 需要性能基线参考 | `uv run kope-mem baseline show --operator {算子名}` |
| 回顾过往决策 | `uv run kope-mem journal show --operator {算子名} --latest 10` |
| 分析优化链路 | `uv run kope-mem case chain --session-id {session_id}` |

## Journal 格式

```markdown
## Iter {N}: {策略名}

- **改动**：{关键代码变更}
- **性能**：{TFLOPS} TFLOPS，利用率 {X}%
- **状态**：✅通过 / ❌回退 / ⚠️部分改善
- **洞察**：{为什么有效/无效，一句话}
```

## Case 格式

遵循 `thoughts_case_schema.json` 定义的结构。最小有效 Case 需包含：`case_id`、`kernel`、`thought`、`scene`、`effect`、`chain`、`runtime` 七个顶层字段。

校验命令：`uv run kope-mem case validate --json-path {file}.json`
