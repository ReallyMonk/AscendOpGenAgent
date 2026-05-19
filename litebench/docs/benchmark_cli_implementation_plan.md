# NPU Benchmark CLI 实现计划

## 目标

基于 [benchmark_cli_design.md](/Users/gujiazhen/Documents/projects/kernel_auto_evolve/OpenOps/benchmark/docs/benchmark_cli_design.md)，实现一个统一的 benchmark CLI，支持：

1. 通过 `torch_npu` / PyTorch API 运行 baseline
2. 上传本地 custom 目录到远程 server 编译并运行
3. 统一输出 E2E、profiling 和对比 CSV
4. 为后续知识迁移与自动调优提供稳定数据入口

## 范围

第一版只要求打通首批 5 个算子：

- `layer_norm_v3`
- `mat_mul_v3`
- `softmax_v2`
- `add_layer_norm`
- `swi_glu`

第一版只要求支持：

- `list`
- `run-baseline`
- `run-custom`
- `compare`

## Phase 0：配置资产整理

### 目标

把 benchmark 所需的静态配置从文档状态转成可执行资产。

### 要做

1. 建立目录：
   - `benchmark/registry/`
   - `benchmark/results/baseline_cache/`
   - `benchmark/results/runs/`
   - `benchmark/results/comparisons/`
   - `benchmark/scripts/`
2. 迁移 registry：
   - `docs/archive/lightweight_benchmark_registry_v2.json`
   - -> `benchmark/registry/lightweight_benchmark_registry_v2.json`
3. 新增：
   - `benchmark/registry/servers.json`
   - `benchmark/registry/runner_templates_v1.json`

### 交付物

- benchmark 静态目录就位
- registry、server、runner template 三类配置可以单独加载

## Phase 1：统一数据模型

### 目标

把 benchmark 的 case、server、执行结果统一成固定数据结构。

### 要做

1. 实现 loader：
   - `BenchmarkRegistry`
   - `ServerConfig`
   - `RunnerTemplateRegistry`
2. 定义数据结构：
   - `BenchmarkCase`
   - `ExpandedCase`
   - `RunConfig`
   - `RunResult`
   - `ProfilingSummary`
3. 实现 `CaseExpander`
   - 将 registry 中的语义 shape 转成 runner 可执行输入

### 交付物

- 给定 `op_name + shape_id` 能生成完整 case
- 所有结果对象可序列化为 JSON / CSV

## Phase 2：实现 `list`

### 目标

先打通最轻的一层，验证 registry 和 template 是否一致。

### 要做

1. 实现：
   - `python3 benchmark/scripts/npu_bench.py list`
2. 支持过滤：
   - `--set`
   - `--layer`
   - `--role`
   - `--category`
   - `--op`
3. 支持输出：
   - 算子元数据
   - shape 列表
   - anchor/probe 关系
   - 默认 server 信息

### 交付物

- 可以稳定列出所有 benchmark case
- 可以用来检查 registry 和 runner template 是否缺字段

## Phase 3：实现 baseline runner + cache

### 目标

支持直接调用 `torch_npu` / PyTorch API 跑 baseline，并引入 baseline cache。

### 要做

1. 实现 `BaselineRunner`
2. 实现：
   - `run-baseline`
3. 支持：
   - `--server`
   - `--device`
   - `--profiling`
   - `--refresh-baseline`
4. 实现 baseline cache：
   - 先查 cache
   - 命中则复用
   - 未命中则执行并写回
5. 输出：
   - `result.json`
   - `profiling.json`
   - `summary.csv`
   - `profiling_summary.csv`

### 交付物

- 单个 baseline case 可运行
- 同 case 第二次可命中 cache
- 结果可稳定落盘

## Phase 4：实现 custom remote runner

### 目标

支持上传本地 custom 目录到远程 server 编译运行，并和 baseline 使用统一结果格式。

### 要做

1. 实现 `CustomRemoteRunner`
2. 实现：
   - `run-custom`
3. 复用：
   - [compile_remote.py](/Users/gujiazhen/Documents/projects/kernel_auto_evolve/OpenOps/server/compile_remote.py)
4. 支持：
   - `--server`
   - `--device`
   - `--custom-dir`
   - `--profiling`
5. 实现本地结构校验：
   - `manifest.json` 存在则优先使用
   - 否则按 `compile_remote.py` 约定推断
6. 处理状态：
   - `success`
   - `compile_failed`
   - `install_failed`
   - `profiling_failed`

### 交付物

- 单个 custom case 可执行
- custom 与 baseline 共用相同 case key
- profiling 失败但 timing 成功时可保留结果

## Phase 5：实现 compare

### 目标

把 baseline/custom 结果自动 join 成统一对比表。

### 要做

1. 实现：
   - `compare`
2. join key：
   - `op_name`
   - `shape_id`
   - `dtype`
   - `server_name`
   - `device`
3. 输出：
   - `compare.csv`
4. 计算：
   - `e2e_speedup`
   - `task_speedup`
   - `vec/scalar/mte ratio delta`

### 交付物

- 可以一键输出 baseline/custom 对比 CSV

## Phase 6：批量执行能力

### 目标

支持按 set/layer 批量 sweep。

### 要做

1. 支持：
   - `--all-shapes`
   - `--set main_benchmark`
   - `--layer foundation`
2. 支持批量失败不中断
3. 支持批量 summary 汇总

### 交付物

- 可以跑一组 benchmark case
- 可以落一份完整批量结果

## 当前还缺什么

## 必须有

这些不补，第一版无法真正跑起来：

1. `benchmark/registry/servers.json`
2. `benchmark/registry/runner_templates_v1.json`
3. 首批 5 个算子的 input builder 规则
4. baseline cache key 规则
5. 统一结果 schema
6. `run-baseline`
7. `run-custom`
8. `compare`

## 建议有

这些会明显降低后续返工：

1. custom 目录最小 `manifest.json` 约定
2. 所有结果里都带：
   - `server_name`
   - `device`
3. run 级 `manifest.json`
4. profiling 失败但 timing 成功的状态区分
5. baseline cache 环境版本记录：
   - `torch`
   - `torch_npu`
   - `CANN`

## 后续再补

第一版不需要，但后面会有价值：

1. HTML report
2. 自动 SSH tunnel 管理
3. transfer probe 自动评估
4. profiling-analysis 自动接入
5. benchmark 结果自动回填知识库

## 当前最关键的 3 个缺口

### 1. `runner_templates_v1.json`

这是从“registry 配置”走向“可执行 case”的关键桥梁。

没有它，就只有：

- 测什么

没有：

- 怎么跑

### 2. input builder 规范

例如：

- `layer_norm_v3` 如何从 `B_S/H` 生成 `x/gamma/beta`
- `softmax_v2` 如何从 `B/heads/S` 生成 4D 输入
- `swi_glu` 如何从 `two_inter` 生成输入张量

### 3. baseline cache key 设计

如果这一层不先固定，后续 benchmark 结果会混乱，无法稳定复用。

## 推荐开发顺序

建议严格按下面顺序推进：

1. `servers.json`
2. `runner_templates_v1.json`
3. `CaseExpander`
4. `list`
5. `run-baseline`
6. baseline cache
7. `run-custom`
8. `compare`

## 验收标准

第一版最小验收标准：

1. 能 `list` 出首批 5 个算子及其 shape
2. `run-baseline` 能跑通至少 3 个算子
3. baseline cache 能命中
4. `run-custom` 能通过远程链路跑通至少 1 个算子
5. `compare` 能输出一份可读的 `compare.csv`

## 结论

第一版的关键不是一口气支持所有 registry 算子，而是先把：

- 配置
- case 展开
- baseline cache
- remote custom
- compare

这 5 条主链路打通。

只要这条骨架稳定，后面扩算子、扩 shape、扩 server 都是线性工作。
