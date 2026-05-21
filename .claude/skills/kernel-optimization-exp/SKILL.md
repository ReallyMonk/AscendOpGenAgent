---
name: kernel-optimization-exp
description: 算子优化经验记录 Skill，用于保存特定算子的优化历史和已验证的优化策略，供下次优化时直接复用
---

## 概述

KernelOptimizationExp Skill 用于在 kope Agent 完成算子优化后，自动生成并保存该算子的优化经验文档。该文档记录了优化过程中的关键信息，包括性能数据、已验证的优化策略，最佳 DSL 配置等。

## 目标

- **记录优化历史**：保存每轮优化的性能数据、使用的策略、迭代结果
- **沉淀优化知识**：从 `exp_dir/perf_diff_log.jsonl` 提取有效的优化模式
- **加速未来优化**：下次优化同一算子时，可直接使用已验证的策略，跳过完整的 kope 探索流程

---

## 触发时机

在 kope Agent 的 "Finalization" 阶段完成后自动调用。

---

## 输入

### 必需输入

- `{op_name}`: 算子名称
- `{op_name}_dsl.py`: Baseline DSL 代码
- `{op_name}_optimized_*.py`: 各轮优化后的 DSL 代码
- `exp_dir/perf_diff_log.jsonl`: code_diff_learning 产生的优化模式记录（由 simple_evo 生成）
- `exp_dir/keypoints_analyzation/round_*.md`: 每轮优化的详细分析文档
- 优化思路来源: 通过 `ideapool_learner` 获取的优化思路记录

### 可选输入

- 性能评估结果（baseline time、optimized time、speedup）
- 优化迭代日志

---

## 输出

生成的 SKILL.md 文件保存在 `.opencode/skills/kernel_optimization_exp/{op_name}/` 目录下，包含以下内容：

### 1. 算子基本信息

```markdown
## 算子信息

- **算子名称**: {op_name}
- **生成时间**: {timestamp}
- **kope 版本**: {version}
```

### 2. 优化历史

```markdown
## 优化历史

### Baseline
- DSL: {path}
- Time: {baseline_time}
- Validation: PASS | FAIL

### Iteration {i}
- Candidate: {path}
- Strategy: {simple_evo/modify_idea/magic_idea}
- Keypoints Analysis: {exp_dir/keypoints_analyzation/round_{i}_{impact_label}.md}
- Build: PASS | FAIL
- Evaluation: PASS | FAIL
- Time: {time}
- Speedup vs Baseline: {speedup}
- Decision: ACCEPT | REJECT
```

### 3. 已验证的优化策略

```markdown
## 优化策略

### 有效策略
1. **{idea_name}**
   - 类别: {memory/compute/reduction/fusion/scheduling}
   - 描述: {策略描述}
   - 代码变更: {diff_summary}
   
2. ...

### 优先级
- **优先尝试**: {idea1}, {idea2}
- **可尝试**: {idea3}
- **谨慎使用**: {idea4}
```

### 4. 最佳 DSL 配置

```markdown
## 最佳配置

### 关键参数
- tiling_factor: {value}
- buffer_strategy: {value}
- ...
```

### 5. 使用条件

```markdown
## 使用条件

### 可直接复用
- 算子类型相同且输入 shape 相似
- 硬件环境相同 (同型号 NPU)

### 需要重新探索
- 输入 shape 差异较大
- 硬件环境变化
- 性能未达到预期
```

---

## 工作流程

### Step 1: 收集优化数据

1. 读取 `simple_evo` 执行日志，提取各轮迭代结果
2. 读取 `exp_dir/perf_diff_log.jsonl`，提取有效优化模式（由 code_diff_learning 生成）
3. 读取 `exp_dir/keypoints_analyzation/round_*.md`，获取每轮优化的详细分析
4. 读取最终的 best DSL 代码

### Step 2: 生成 SKILL.md

1. 创建 `.opencode/skills/kernel_optimization_exp/{op_name}/` 目录
2. 生成包含上述内容的 SKILL.md 文件

### Step 3: 保存验证

1. 确保文件创建成功
2. 输出保存路径

---

## 使用方式

当需要优化一个已有优化记忆的算子时：

1. 检查 `.opencode/skills/kernel_optimization_exp/{op_name}/` 是否存在
2. 如果存在，读取 SKILL.md 获取优化策略
3. 直接使用已验证的策略，跳过完整的 kope 探索流程
4. 如果不存在或优化失败，执行完整的 kope 流程
