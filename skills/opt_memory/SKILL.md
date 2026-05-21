# OptMemory Skill

## 用途

在 AscendC 算子优化过程中自动管理记忆：叙事日志（Journal）、结构化经验（ThoughtKernelCase）、性能基线、优化模式。

基于 ReMe 引擎提供对话压缩和语义检索能力。

## 何时使用

- Lingxi-code Agent 执行 AscendC 算子优化时自动激活
- 编译/benchmark 输出过长导致上下文紧张时
- 需要跨会话复用优化经验时
- 需要结构化记录优化路径供 SFT 训练时

## 快速参考

### Journal 轨（叙事日志）

```bash
# 追加迭代记录
uv run kope-mem journal add \
  --operator {算子名} \
  --iteration {N} \
  --strategy "{策略名}" \
  --change "{关键改动}" \
  --tflops {值} --utilization {利用率} \
  --status {pass|fail|partial} \
  --insight "{一句话洞察}"

# 查看历史
uv run kope-mem journal show --operator {算子名} --latest 10
```

### Case 轨（结构化记录）

```bash
# 快速追加（推荐用于迭代记录）
uv run kope-mem case append \
  --operator {算子名} \
  --thought-id {经验ID} \
  --thought-title "{标题}" \
  --thought-desc "{做了什么}" \
  --thought-reason "{为什么}" \
  --opt-layer {core_partition|tiling|pipeline|instruction|none} \
  --step-improvement {倍数} \
  --label {effective|partial|ineffective|negative} \
  --session-id {会话ID} \
  --step-index {N} \
  --dtype {float16|float32|bfloat16|int8|int32} \
  --shape-signature "{形状摘要}"

# 按条件查询
uv run kope-mem case list --operator {算子名} --label effective

# 重建链路
uv run kope-mem case chain --session-id {会话ID}

# 校验 Case 合法性
uv run kope-mem case validate --json-path {文件路径}
```

### 上下文压缩

```bash
# 压缩过长的优化对话
uv run kope-mem compact --operator {算子名}

# 指定对话文件
uv run kope-mem compact --dialog opt_memory/dialog/{日期}.jsonl
```

### 语义检索

```bash
# 通用检索
uv run kope-mem search --query "{自然语言}" --max-results 5

# 按算子过滤
uv run kope-mem search --query "{查询}" --operator {算子名}

# 结构化过滤
uv run kope-mem case list --thought-id {经验ID}
uv run kope-mem case list --opt-layer tiling --label effective
```

### 性能基线

```bash
# 记录基线
uv run kope-mem baseline add --operator {算子名} --tflops {值} --utilization {利用率}

# 查看趋势
uv run kope-mem baseline show --operator {算子名}

# 对比
uv run kope-mem baseline compare --operator {算子名} --date-a {日期1} --date-b {日期2}
```

### 文件监听

```bash
# 后台守护，自动压缩
uv run kope-mem watch --operator {算子名}
```

## 依赖

- **核心**：cyclopts, watchdog, jsonschema
- **ReMe 能力**（compact / search）：`uv sync --extra reme`
- **环境变量**：`.env` 中配置 `LLM_API_KEY`、`EMBEDDING_API_KEY` 等
