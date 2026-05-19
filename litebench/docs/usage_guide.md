# Lite Benchmark 使用说明

## 目标

`benchmark/` 是一套自包含的 NPU 算子 benchmark 工具，支持两类评测：

- `baseline`
  - 直接调用 `torch` / `torch_npu` 官方 API
  - 获取 `e2e` 时间
  - 可选获取 profiling 结果
- `custom`
  - 输入一个可编译算子目录
  - 远端编译并运行
  - 获取 `e2e` 时间与 profiling 结果

工具统一提供：

- benchmark case registry
- runner 模板
- remote worker/client
- compile / profiling / compare

## 目录结构

```text
benchmark/
  registry/
    lightweight_benchmark_registry_v2.json
    runner_templates_v1.json
    servers.json
  scripts/
    npu_bench.py
  remote/
    worker_server.py
    remote_client.py
    compile_remote.py
    baseline_benchmark_runner.py
    profiling_runner.py
    start_worker.sh
  assets/
    evaluate.py
    generate_pybind.py
    template/
  results/
  docs/
```

## 核心配置

### registry

- [lightweight_benchmark_registry_v2.json](/Users/gujiazhen/Documents/projects/kernel_auto_evolve/OpenOps/benchmark/registry/lightweight_benchmark_registry_v2.json)
  - 定义 benchmark 算子、shape、anchor/probe、layer/category

### runner templates

- [runner_templates_v1.json](/Users/gujiazhen/Documents/projects/kernel_auto_evolve/OpenOps/benchmark/registry/runner_templates_v1.json)
  - 定义每个算子的 API、input builder、attrs、输入规格

### servers

- [servers.json](/Users/gujiazhen/Documents/projects/kernel_auto_evolve/OpenOps/benchmark/registry/servers.json)
  - 定义远端 server 名称、worker URL、默认 device、client_id

默认 server 当前是：

- `910b`
- `worker_url=http://127.0.0.1:9027`
- `default_device=7`

## 启动远端 worker

在远端机器上启动：

```bash
bash benchmark/remote/start_worker.sh \
  --port 9027 \
  --log-dir /path/to/benchmark_worker_logs \
  --devices 7 \
  --max-workers 2
```

如果 worker 跑在 conda 环境里，先激活包含 `torch_npu`、`fastapi`、`python-multipart` 的环境。

健康检查：

```bash
curl http://127.0.0.1:9027/health
```

## 常用命令

以下命令既可以在仓库根目录执行，也可以 `cd benchmark` 后执行。

### 1. 查看 benchmark case

```bash
python3 benchmark/scripts/npu_bench.py list --set main_benchmark
python3 benchmark/scripts/npu_bench.py list --set transfer_probes
python3 benchmark/scripts/npu_bench.py list --op layer_norm_v3
```

### 2. 跑 baseline

```bash
python3 benchmark/scripts/npu_bench.py run-baseline \
  --op layer_norm_v3 \
  --shape S2 \
  --server 910b \
  --device 7
```

带 profiling：

```bash
python3 benchmark/scripts/npu_bench.py run-baseline \
  --op layer_norm_v3 \
  --shape S2 \
  --server 910b \
  --device 7 \
  --profiling
```

### 3. 跑 custom

```bash
python3 benchmark/scripts/npu_bench.py run-custom \
  --op avg_pool2d \
  --shape S2 \
  --custom-dir output/avg_pool2d \
  --server 910b \
  --device 7
```

带 profiling：

```bash
python3 benchmark/scripts/npu_bench.py run-custom \
  --op avg_pool2d \
  --shape S2 \
  --custom-dir output/avg_pool2d \
  --server 910b \
  --device 7 \
  --profiling
```

### 4. 对比 baseline 和 custom

同一个 run 目录内比较：

```bash
python3 benchmark/scripts/npu_bench.py compare \
  --results-dir benchmark/results/runs/<run_id>
```

跨两个 run 目录比较：

```bash
python3 benchmark/scripts/npu_bench.py compare \
  --baseline-results-dir benchmark/results/runs/<baseline_run_id> \
  --custom-results-dir benchmark/results/runs/<custom_run_id> \
  --out-csv benchmark/results/comparisons/<name>.csv
```

## 结果目录

默认结果落在：

```text
benchmark/results/
  baseline_cache/
  runs/
  comparisons/
```

### baseline cache

同一 server / device / op / shape / dtype / runner 配置下，baseline 会缓存。

如果想强制重跑：

```bash
python3 benchmark/scripts/npu_bench.py run-baseline ... --refresh-baseline
```

### runs

每次 run 会生成：

- `manifest.json`
- `summary.csv`
- `profiling_summary.csv`
- 分 case 的 `result.json / profiling.json`

### comparisons

`compare` 会输出统一的 compare CSV，包含：

- baseline/custom e2e
- speedup
- task duration
- profiling ratio delta

## 当前已验证链路

已经验证过的关键路径：

- `layer_norm_v3:S2`
  - baseline + profiling
- `avg_pool2d:S2`
  - custom + profiling
  - baseline/custom compare
- `main_benchmark` 的 `S1`
  - 主集首档已逐项修通

## 注意事项

### 1. custom-dir 的含义

`run-custom` 需要传一个本地算子目录，例如：

```text
output/avg_pool2d
```

工具会按约定查找：

- `<op_name>_dsl.py`
- `*Custom/`
- reference/custom/cpp 文件

### 2. avg_pool2d 的 dtype

当前 benchmark registry 中，`avg_pool2d` 为了兼容现有 custom 实现，使用的是 `float32`。

### 3. worker 单卡并发

如果同一个 worker 只绑定了一张卡，不建议并行发太多 run。  
批量验证时更稳妥的方式是串行跑。

### 4. compare 口径

`e2e speedup` 与 `task_duration speedup` 可能不一致：

- `e2e` 反映端到端时间
- `task_duration_us` 反映 profiling 中的算子任务时间

这两个都应一起看。

## 推荐流程

### 调优前

1. `list` 看 case
2. `run-baseline` 拿 baseline e2e / profiling

### 调优中

1. LLM / 启发式生成 custom 实现
2. `run-custom` 跑远端编译与 profiling

### 调优后

1. `compare` 生成对比结果
2. 将有效配置或知识回写到调优系统
