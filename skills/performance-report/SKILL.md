---
name: performance-report
description: >
  整合 kope agent 优化过程中收集的多算子多 shape 性能测试数据, 生成单文件 HTML 性能报告。
  报告核心展示 AscendC 优化前后的执行速度加速比, PyTorch 参考实现速度作为旁路参考,
  并按知识卡片分类总结优化过程中使用的优化技巧。
  当 kope agent 完成算子优化、需要将多 shape 性能数据汇总为可视化报告时调用。
argument-hint: >
  输入: {output_dir}/perf_data.json (由 kope agent 整理生成, 支持多算子)
  输出: {output_dir}/performance_report.html
---

# Performance Report Skill

## 概述

本 skill 把 kope agent 在算子优化过程中收集到的多算子多 shape 性能数据(以及使用过的知识卡片)整理为单文件 HTML 报告。报告使用浅色主题, 在 header 区集中展示实验环境(CANN 版本、NPU 型号、驱动等), 主体按算子分章节呈现。

**核心指标**:
- AscendC 优化前 vs 优化后的加速比(主指标, 几何平均)
- PyTorch 参考实现的速度(列在旁, 不参与主加速比计算)
- 优化技巧总结(从知识卡片提取, 按 category 分组)

## 适用场景

- kope agent 完成算子优化后, 需要将多 shape 性能数据汇总为可视化报告
- 用户希望获得一份可以打开浏览器直接查看的 HTML 报告
- 优化报告需要包含优化技巧的系统性总结(从知识卡片溯源)
- 一次实验覆盖多个算子(本 skill 支持多算子单报告)

## 前置条件

- kope agent 已完成多算子多 shape 的性能测试
- 性能数据已整理为 `perf_data.json`(见下方 schema, 支持多算子)
- 知识卡片使用清单已整理(可选, 但建议提供)

## 输入数据 Schema (多算子)

调用方将数据组织为以下 JSON 结构, 写入 `{output_dir}/perf_data.json`:

```json
{
  "experiment_name": "exp_2026_06_06_litebench",
  "generated_at": "2026-06-06T10:30:00",
  "global_meta": {
    "device": "Ascend 910B",
    "npu_model": "Ascend 910B",
    "cann_version": "CANN 8.0.RC2",
    "soc_version": "Ascend910B1",
    "driver_version": "24.1.rc2",
    "npu_memory_gb": 64,
    "torch_version": "2.1.0",
    "torch_npu_version": "2.1.0.post6"
  },
  "operators": [
    {
      "op_name": "add_rms_norm",
      "shapes": [
        {
          "shape_id": "s1",
          "description": "M=128, N=128 (small)",
          "pytorch_ms": 0.15,
          "ascendc_before_ms": 0.45,
          "ascendc_after_ms": 0.08
        }
      ],
      "knowledge_cards": [
        {
          "card_id": "tiling_2d_v1",
          "title": "2D 分块策略",
          "category": "tiling",
          "application": "应用了 64x64 分块, 提升 L2 命中率"
        }
      ]
    }
  ],
  "knowledge_cards": []
}
```

### 字段说明

**顶层字段**:
- `experiment_name` (str, 必填): 实验名, 显示在 header
- `generated_at` (str, 必填): 报告生成时间 (ISO 8601)
- `global_meta` (object, 必填): 实验环境信息, 在 header 网格区展示
  - 常用键: `device` / `npu_model` / `cann_version` / `soc_version` / `driver_version` / `npu_memory_gb` / `torch_version` / `torch_npu_version`
  - 任意额外键会被原样展示
- `operators[]` (array, 必填, 至少 1 条): 算子列表, 每个算子独立渲染
  - `op_name` (str): 算子名
  - `shapes[]` (array): 多 shape 性能数据
    - `shape_id` (str): 唯一标识
    - `description` (str): 人类可读描述
    - `pytorch_ms` (float): PyTorch 参考耗时 (ms)
    - `ascendc_before_ms` (float): AscendC 优化前耗时 (ms)
    - `ascendc_after_ms` (float): AscendC 优化后耗时 (ms)
  - `knowledge_cards[]` (array, 可选): 该算子使用的知识卡片
    - `card_id` / `title` / `category` / `application`
- `knowledge_cards[]` (array, 可选): 实验级共享卡片, 在算子 section 外单独展示

### 向后兼容

旧 schema (单算子, `shapes` 在顶层) 仍然支持, 自动转换:
```json
{
  "op_name": "add_rms_norm",
  "device": "Ascend 910B",
  "shapes": [...],
  "knowledge_cards": [...]
}
```

## 输出

- `{output_dir}/performance_report.html`: 单文件 HTML 报告
  - 内嵌 CSS, 浅色主题, 无外部样式依赖
  - 图表使用 Chart.js (CDN 引入)
  - 中文界面

## 使用方式

### 命令行调用

```bash
python3 $SKILL_DIR/references/render_html.py \
  --input {output_dir}/perf_data.json \
  --output {output_dir}/performance_report.html
```

### 编程调用

```python
from references.render_html import render_report
render_report(perf_data_dict, output_path)
```

## 与 kope agent 的集成

kope agent 在完成以下步骤后调用本 skill:

1. **收集性能数据**: 在每轮优化迭代中, 对当前 best 实现和初始实现分别跑多 shape 性能测试
2. **整理知识卡片**: 通过 knowledge-cards MCP 的 `card_search` / `card_get` 提取本次优化使用过的卡片, 按算子归类
3. **生成 perf_data.json**: 把多算子的 shapes + 知识卡片组织为 JSON, 包含 `global_meta`(CANN/NPU 等环境信息)
4. **调用本 skill**: 把 JSON 传给 render_html.py, 生成 HTML 报告
5. **登记产出**: 把报告路径写入 `case_finalize` 的产出清单

## 标准格式（不可修改的模板规范）

**以下格式是 performance-report skill 输出的标准模板, 任何对 render_html.py 的修改都不得改变下列结构**。这保证了下游消费者 (kope agent / 用户 / CI) 对报告的解析和阅读习惯稳定:

### 章节顺序（固定）

1. **Header** (蓝色渐变)
   - 实验名、生成时间
   - `global_meta` 网格: 设备 / CANN / NPU 型号 / 驱动 / PyTorch / torch_npu / ...
2. **整体 Summary Cards**
   - 测试算子数
   - 总 Shape 数
   - 整体几何平均加速比(主指标, 高亮卡片)
   - 最大 / 最小加速比
3. **Per-Operator Section** (按 `operators` 循环, **1-based 序号**):
   - 算子标题: `算子 N: 算子名` (N 从 1 开始, 反映 `operators` 列表顺序, 便于多算子报告交叉引用)
   - 测试 shape 数
   - 算子 Summary Cards(测试 shape 数 / 几何平均加速比 / 最大 / 最小)
   - 算子图表(独立 canvas, 3 柱状图: PyTorch / Before / After)
   - 算子详细数据表
   - 算子优化技巧总结(按 category 分组, 如有 knowledge_cards)
4. **实验级知识卡片** (如有, 在算子 section 外)
5. **整体结论** (自动评价)

### 关键列 / 视觉约定（固定）

- 详细数据表固定列: `Shape ID` / `Description` / `PyTorch (ms)` / `Before (ms)` / `After (ms)` / `Speedup vs Before (主)` / `vs PyTorch (旁)`
- 主加速比 (`Speedup vs Before`) 在表格中**高亮**(粗体 + 浅色背景), `vs PyTorch` 为普通列
- 整体 / 算子 Summary Cards 的加速比卡片**高亮**(蓝色背景), 其他卡片为白色
- 图表固定为 3 柱状图 (PyTorch / Before / After), 颜色固定: PyTorch=灰, Before=橙, After=蓝
- 主题: 浅色, 字体 sans-serif, 主色 #1e6fff
- HTML 输出: 单文件, 内嵌 CSS, 仅依赖 Chart.js CDN

### 修改 render_html.py 时的禁止项

- 不得删除 / 重排以上 5 个章节
- 不得删除 / 改名以上 7 个表格列
- 不得删除 / 改名 Summary Cards 的 4 个卡片
- 不得修改主加速比的高亮样式规则
- 不得修改 3 柱状图的颜色方案
- 不得引入其他外部资源(除 Chart.js CDN)
- 不得输出多文件报告(必须单 HTML)

> 想要调整的, 应该是**容错策略** (见下) 而非标准格式。如果确实需要改格式 (如新增章节), 应通过 schema 升版而非直接动代码, 避免破坏下游消费方。

## 关键限制（建议性, 非硬性）

> ⚠️ 之前的硬性"必须包含 X / 缺少则失败"已废弃, 详见下节「异常处理」。本节仅作建议。

- 输入 JSON **建议**包含 `operators`(新 schema) 或 `shapes`(旧 schema) 数组
- 每个 shape **建议**有 `ascendc_before_ms` 和 `ascendc_after_ms` 字段(否则主加速比降级为 0)
- 知识卡片列表可为空(报告中会标注"未记录使用")
- 输出 HTML 仅依赖 Chart.js CDN, 无其他外部资源

## 异常处理

采用"轻清洗 + 允许留空"策略, **任何数据问题都不会让整份报告失败**, 调用方不会收到异常:

| 情况 | 处理方式 |
|------|---------|
| `data` 不是 dict (None / list / 其他) | 降级为 `{"operators": []}`, 输出"无算子数据"占位报告 |
| `operators` 不是 list (None / str 等) | 降级为 `[]` |
| `operators` 为空 | 输出"无算子数据"占位报告 (header + 结论卡片仍正常) |
| 单个 op 无 `shapes` 字段 / `shapes` 为空 | 跳过该 op, 继续渲染其他 op |
| 数值字段是 None / 字符串 (如 `"0.15 ms"`) / 异常类型 | `_safe_float()` 降级为 `0.0`, 报告仍生成 |
| 算子级 / 实验级 `knowledge_cards` 缺失 / 为空 / 非 list | 降级为 `[]`, 报告中标注"未记录使用" |
| `knowledge_card` 缺 `category` | 归入 `other` 分组 |
| `pytorch_ms` 为 0 | 跳过 vs PyTorch 加速比, 标注"无 PyTorch 数据" |
| `ascendc_before_ms` 或 `ascendc_after_ms` 为 0 | 跳过该 shape 的加速比计算 |
| `global_meta` 缺失或字段为 None/空 | 跳过该字段, 其他字段照常展示 |
| 算子名 / shape_id 含特殊字符 (/ & . Unicode 等) | `_safe_id` / `_escape` 保护, 报告正常 |
| 输出目录不存在 | 自动创建 |
| shape 含多余字段 | 忽略 |

### 验证方法

跑 `examples/fuzz_test.py` 用 28 个异常场景 (字段缺失 / None / 类型错 / 格式乱 / 边界值) 验证容错:

```bash
python3 examples/fuzz_test.py
# 期望输出: PASS: 28  STRUCT_BAD: 0  CRASH: 0  TOTAL: 28
```

