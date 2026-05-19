# NPU Benchmark CLI 设计文档

## 1. 目标

设计一套统一的命令行 benchmark 工具，用于：

1. 直接调用 `torch_npu` / PyTorch API 运行官方 baseline
2. 上传一个本地可编译算子目录到远程 NPU server，远程编译并运行 custom kernel
3. 在同一套 case/shape 定义下，统一采集：
   - E2E 时间
   - profiling 摘要
   - 原始 profiling JSON
4. 输出统一的 `.csv` 和 JSON 结果，便于对比、复盘和知识沉淀

该工具服务于两类场景：

- `benchmark`：对基础算子、融合算子进行稳定性能评测
- `optimization`：对某个候选实现与 baseline 做统一对比

## 2. 非目标

第一版不解决以下问题：

- 不直接负责自动生成自定义算子代码
- 不直接负责 profiling 瓶颈分析结论生成
- 不直接负责知识蒸馏
- 不直接负责模型级 workload benchmark

这些能力可以建立在 benchmark 结果之上，但不属于 CLI 第一版职责。

## 3. 总体设计

工具采用三层结构：

1. `Registry`
   - 定义测什么
   - 包含算子、shape、anchor/probe、layer 等配置
2. `Runner Templates`
   - 定义怎么把一个 registry case 转成可执行输入
   - 包含 API 名、输入构造方式、attrs、dtype/layout 规则
3. `CLI Orchestrator`
   - 统一调度 baseline / custom 两种执行后端
   - 负责落盘、汇总、比较

## 4. 目录结构

建议新增统一目录：

```text
benchmark/
  docs/
    benchmark_cli_design.md
  registry/
    lightweight_benchmark_registry_v2.json
    runner_templates_v1.json
    servers.json
  results/
    baseline_cache/
    runs/
    comparisons/
  scripts/
    npu_bench.py
    baseline_runner.py
    custom_remote_runner.py
    case_expander.py
    result_writer.py
```

说明：

- `registry/`
  - 放静态 benchmark 配置
- `registry/servers.json`
  - 放远程 server 列表和默认连接配置
- `results/baseline_cache/`
  - 放复用的官方 baseline 结果
- `results/runs/`
  - 放每次实际 benchmark 执行结果
- `results/comparisons/`
  - 放汇总对比 CSV
- `scripts/`
  - 放 CLI 和执行器

## 5. Registry 资产

当前已有：

- `docs/archive/lightweight_benchmark_registry_v2.json`

建议迁移到：

- `benchmark/registry/lightweight_benchmark_registry_v2.json`

该 registry 继续承担：

- benchmark anchor / transfer probe 定义
- shape 集合定义
- operator layer / category / bottleneck tag
- benchmark set 定义

## 6. Runner Template 资产

除了 registry，还需要新增 runner template 文件，例如：

- `benchmark/registry/runner_templates_v1.json`
- `benchmark/registry/servers.json`

该文件用于把“语义 shape”展开成“可执行 case”。

每个算子模板至少需要：

- `runner_type`
  - `torch_api`
  - `torch_module`
  - `torch_npu_custom_api`
- `api`
  - 例如 `torch.nn.functional.softmax`
- `input_builder`
  - 例如 `layer_norm_2d`, `matmul_2d`, `softmax_4d`
- `attrs`
  - 例如 `dim=-1`, `eps=1e-5`
- `dtype_policy`
  - 默认 dtype，是否允许覆盖
- `profiling_supported`
- `custom_adapter`
  - custom 目录如何和这个 case 对齐

### 6.1 例子

`layer_norm_v3` 的 template 可以表达为：

```json
{
  "layer_norm_v3": {
    "runner_type": "torch_api",
    "api": "torch.nn.functional.layer_norm",
    "input_builder": "layer_norm_2d",
    "attrs": {
      "eps": 1e-5
    },
    "profiling_supported": true
  }
}
```

`mat_mul_v3` 的 template 可以表达为：

```json
{
  "mat_mul_v3": {
    "runner_type": "torch_api",
    "api": "torch.matmul",
    "input_builder": "matmul_2d",
    "profiling_supported": true
  }
}
```

## 7. CLI 子命令设计

统一入口建议为：

```bash
python3 benchmark/scripts/npu_bench.py ...
```

### 7.1 `list`

列出可用 benchmark case。

示例：

```bash
python3 benchmark/scripts/npu_bench.py list \
  --registry benchmark/registry/lightweight_benchmark_registry_v2.json
```

支持过滤：

- `--layer foundation`
- `--role anchor`
- `--category normalization`
- `--op layer_norm_v3`
- `--set main_benchmark`

### 7.2 `run-baseline`

直接通过 `torch_npu` / PyTorch API 运行 baseline。

示例：

```bash
python3 benchmark/scripts/npu_bench.py run-baseline \
  --registry benchmark/registry/lightweight_benchmark_registry_v2.json \
  --op layer_norm_v3 \
  --shape S2 \
  --device npu:7 \
  --warmup 20 \
  --repeat 100 \
  --profiling
```

支持批量：

- `--set main_benchmark`
- `--layer foundation`
- `--all-shapes`

### 7.3 `run-custom`

上传本地 custom 目录到远程 server，远程编译并执行。

示例：

```bash
python3 benchmark/scripts/npu_bench.py run-custom \
  --registry benchmark/registry/lightweight_benchmark_registry_v2.json \
  --op layer_norm_v3 \
  --shape S2 \
  --custom-dir output/layer_norm_v3_candidate_03 \
  --server 910b \
  --device 7 \
  --warmup 20 \
  --repeat 100 \
  --profiling
```

内部应复用：

- `server/compile_remote.py`

默认约束：

- 默认 server 来自 `servers.json` 中的 `default_server`
- `--device` 用于覆盖该 server 的默认 device id

### 7.4 `compare`

读取结果目录，输出对比 CSV。

示例：

```bash
python3 benchmark/scripts/npu_bench.py compare \
  --results-dir benchmark/results/runs/2026-04-09_120000 \
  --out-csv benchmark/results/comparisons/2026-04-09_compare.csv
```

## 8. Case 执行模型

统一 case 主键建议为：

- `op_name`
- `shape_id`
- `dtype`
- `device`
- `case_variant`

其中：

- baseline 的 `case_variant=baseline`
- custom 的 `case_variant=<candidate_name>`

这样可以天然支持 baseline/custom join。

## 9. Baseline Cache 设计

## 9.1 为什么要缓存

对于固定环境下的官方 baseline，不应每次重复运行。

原因：

- 成本高
- 有额外噪声
- 不利于快速比较 custom 结果
- 无法形成稳定历史基线

因此 baseline 应设计为可复用资产。

## 9.2 Cache 路径

建议目录：

```text
benchmark/results/baseline_cache/
  <server_name>/
    torch2.5.1_torch_npu2.5.1_cann8.5.0/
      layer_norm_v3/
        S2/
          baseline_result.json
          baseline_profiling.json
```

## 9.3 Cache Key

baseline cache key 至少包含：

- `server_name`
- `hardware`
- `device_model`
- `device_id`
- `torch_version`
- `torch_npu_version`
- `cann_version`
- `op_name`
- `shape_id`
- `dtype`
- `runner_template_hash`
- `benchmark_config_hash`

其中 `benchmark_config_hash` 至少由以下字段组成：

- `warmup`
- `repeat`
- `profiling_enabled`
- `device_id`

## 9.4 Cache 使用规则

默认逻辑：

1. 先查 baseline cache
2. 命中则复用
3. 未命中则执行并写回 cache

显式刷新：

```bash
--refresh-baseline
```

环境变化自动失效：

- `torch` 版本变化
- `torch_npu` 版本变化
- `CANN` 版本变化
- runner template 或 benchmark 配置变化

## 9.5 Cache 元数据

每条 baseline 结果应记录：

- `cache_key`
- `created_at`
- `server_name`
- `hardware`
- `device_id`
- `torch_version`
- `torch_npu_version`
- `cann_version`
- `warmup`
- `repeat`
- `profiling_enabled`
- `valid`
- `source`
  - `measured`
  - `reused_cache`

## 10. 两种执行后端

## 10.1 BaselineRunner

职责：

- 从 registry + runner template 生成可执行输入
- 调用 `torch_npu` / PyTorch API
- 采集 correctness、E2E、profiling
- 写入 baseline cache 和本次 run 结果

输出：

- `result.json`
- `profiling.json`
- summary 行

## 10.2 CustomRemoteRunner

职责：

- 校验 `custom-dir`
- 调用 `server/compile_remote.py`
- 统一解析返回结果
- 从 `servers.json` 解析远程 worker
- 使用 `--device` 或 server 默认 device

输入：

- `op_name`
- `shape_id`
- `custom_dir`
- registry case
- runner template

输出：

- `result.json`
- `profiling.json`
- summary 行

## 11. Custom 目录约束

建议 custom 目录支持最小 manifest：

```json
{
  "op_name": "layer_norm_v3",
  "impl_name": "candidate_03",
  "project_dir": "layer_norm_v3/layer_norm_v3Custom",
  "dsl_file": "layer_norm_v3/layer_norm_v3_dsl.py"
}
```

如果没有 manifest，则按已有 `compile_remote.py` 约定推断：

- `output/{op_name}/{OpName}Custom`
- `output/{op_name}/{op_name}_dsl.py`

CLI 应在上传前先做本地结构校验。

## 12. 结果目录设计

每次 benchmark run 建议落到：

```text
benchmark/results/runs/
  2026-04-09_154500/
    manifest.json
    baseline/
      layer_norm_v3/
        S2/
          result.json
          profiling.json
    custom/
      layer_norm_v3/
        S2/
          candidate_03/
            result.json
            profiling.json
    summary.csv
    profiling_summary.csv
    compare.csv
```

说明：

- `manifest.json`
  - 记录本次 run 的全局配置
- `summary.csv`
  - 一行一个执行结果
- `profiling_summary.csv`
  - 一行一个 profiling 摘要
- `compare.csv`
  - baseline/custom 对比表

## 13. CSV 设计

## 13.1 `summary.csv`

字段建议：

- `run_id`
- `mode`
  - `baseline`
  - `custom`
- `op_name`
- `shape_id`
- `benchmark_layer`
- `benchmark_role`
- `category`
- `dtype`
- `server_name`
- `device`
- `backend`
- `status`
- `correctness`
- `e2e_median_ms`
- `e2e_p95_ms`
- `profiling_available`
- `profiling_path`
- `custom_dir`
- `worker_url`
- `created_at`

## 13.2 `profiling_summary.csv`

字段建议：

- `run_id`
- `op_name`
- `shape_id`
- `task_duration_us`
- `aiv_vec_ratio`
- `aiv_scalar_ratio`
- `aiv_mte2_ratio`
- `aiv_mte3_ratio`
- `aiv_icache_miss_rate`
- `aic_mac_ratio`
- `cube_utilization`
- `profiling_error`

## 13.3 `compare.csv`

字段建议：

- `op_name`
- `shape_id`
- `baseline_e2e_ms`
- `custom_e2e_ms`
- `e2e_speedup`
- `baseline_task_duration_us`
- `custom_task_duration_us`
- `task_speedup`
- `vec_ratio_delta`
- `scalar_ratio_delta`
- `mte2_ratio_delta`
- `mte3_ratio_delta`

## 14. Manifest 设计

每次 run 建议生成一个 manifest，例如：

```json
{
  "run_id": "2026-04-09_154500",
  "registry_path": "benchmark/registry/lightweight_benchmark_registry_v2.json",
  "runner_template_path": "benchmark/registry/runner_templates_v1.json",
  "server_config_path": "benchmark/registry/servers.json",
  "server_name": "910b",
  "worker_url": "http://localhost:9007",
  "device": "7",
  "warmup": 20,
  "repeat": 100,
  "profiling": true,
  "created_at": "2026-04-09T15:45:00+08:00"
}
```

## 15. 错误处理

### 15.1 Baseline

- API 不存在
- 输入 shape 非法
- NPU 设备不可用
- profiling 失败

处理策略：

- `status=failed`
- 原始异常写入 `result.json`
- 在 summary 中保留错误信息摘要

### 15.2 Custom

- custom 目录结构不合法
- 远程 worker 不可达
- 编译失败
- 安装失败
- profiling 失败但 timing 可用

处理策略：

- 细分 `status`
  - `compile_failed`
  - `install_failed`
  - `profiling_failed`
  - `success`
- profiling 失败但 timing 成功时，不应丢弃整条结果

## 16. 第一版实现范围

第一版不建议直接支持全部 registry 算子。

建议先打通以下 5 个：

- `layer_norm_v3`
- `mat_mul_v3`
- `softmax_v2`
- `add_layer_norm`
- `swi_glu`

原因：

- 同时覆盖 foundation / fusion / challenge
- 同时覆盖 torch API 和 torch_npu custom API
- 同时覆盖 reduce / matmul / fused activation
- 足够验证 CLI 架构是否合理

## 17. 实现建议

按以下顺序落地最稳：

1. 迁移 registry 到 `benchmark/registry/`
2. 定义 `runner_templates_v1.json`
3. 实现 `list`
4. 实现 `run-baseline`
5. 实现 baseline cache
6. 实现 `run-custom`
7. 实现 `compare`

## 18. Server 配置

建议新增：

- `benchmark/registry/servers.json`

用于集中管理远程 server，而不是把 `worker_url` 写死在命令行或文档中。

建议结构：

```json
{
  "default_server": "910b",
  "servers": {
    "910b": {
      "worker_url": "http://localhost:9007",
      "ssh_host": "910b",
      "hardware": "Ascend910B",
      "default_device": "7",
      "client_id": "gjz_910b"
    },
    "a2": {
      "worker_url": "http://localhost:9010",
      "ssh_host": "atlas_a2",
      "hardware": "AscendA2",
      "default_device": "0",
      "client_id": "gjz_a2"
    }
  }
}
```

CLI 建议支持：

- `--server`
  - 选择使用哪个 server 配置
- `--device`
  - 指定实际运行的 device id
  - 如果未指定，则使用 `servers.json` 中该 server 的 `default_device`

## 19. 后续扩展

后续可以继续扩展：

- 多 shape sweep
- baseline cache 版本管理
- 自动生成 HTML benchmark report
- profiling analysis 直接接入
- knowledge transfer 实验报表
- model-level workload benchmark

## 20. 结论

这套 CLI 的正确定位不是一个单纯的 benchmark 脚本，而是：

**一个统一的 benchmark orchestrator，上接 benchmark registry、runner templates 和 `servers.json`，下接 `torch_npu baseline` 与 remote custom compile 两种执行后端，最终输出统一的性能与 profiling 结果。**

其中，baseline cache 是第一版必须纳入设计的核心能力，而不是后续优化项。
