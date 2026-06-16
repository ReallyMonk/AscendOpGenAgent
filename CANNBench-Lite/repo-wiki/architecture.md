# Architecture

## System Overview

CANNBench-Lite uses a **client-server architecture** where a lightweight CLI client sends benchmark requests to a remote Worker server running on an NPU machine.

```
┌─────────────────────────────────────────────────────────────┐
│  CLI Client (cannbench-lite)                                 │
│                                                              │
│  ┌──────────┐  ┌────────────────┐  ┌─────────────────────┐ │
│  │ Registry  │  │ Case Expander  │  │ RemoteWorkerClient   │ │
│  │ Loader    │  │ (expand_case)  │  │ (httpx async)        │ │
│  └─────┬────┘  └───────┬────────┘  └──────────┬──────────┘ │
│        │               │                      │             │
│  ┌─────┴───────────────┴──────────────────────┴──────────┐ │
│  │                    cli.py (argparse)                   │ │
│  │  list | run-baseline | run-custom | compare            │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (httpx)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker Server (cannbench-worker)                            │
│                                                              │
│  ┌──────────────┐  ┌───────────────────────────────────┐   │
│  │ FastAPI App   │  │ DevicePool (asyncio.Queue)        │   │
│  │ + Endpoints   │  │ RequestQueue (asyncio.Semaphore)  │   │
│  └──────┬───────┘  └───────────────────────────────────┘   │
│         │ subprocess                                         │
│  ┌──────┴───────────────────────────────────────────────┐   │
│  │  baseline_benchmark_runner.py  │  profiling_runner.py │   │
│  │  (torch/torch_npu benchmark)   │  (acl.prof + msprof) │   │
│  └────────────────────────────────┴──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Relationships

### Registry Loading Pipeline

```
lightweight_benchmark_registry_v2.json  ──>  BenchmarkRegistry
                                                  │
runner_templates_v1.json               ──>  RunnerTemplateRegistry
                                                  │
servers.json                           ──>  ServerRegistry
                                                  │
                                                  ▼
                                            case_expander.expand_case()
                                                  │
                                                  ▼
                                            ExpandedCase (op_name, shape_id, inputs, attrs, ...)
```

### Baseline Benchmark Flow

```
cannbench-lite run-baseline --op gelu --shape S1
  │
  ├── 1. Load registries (BenchmarkRegistry, RunnerTemplateRegistry, ServerRegistry)
  ├── 2. Filter + expand cases via expand_case()
  ├── 3. For each ExpandedCase:
  │     ├── Check baseline_cache/ for cached result
  │     │   └── If cache hit (same server + hardware + device + op + shape + dtype + warmup + repeat):
  │     │       └── Reuse cached result.json + profiling.json
  │     ├── If cache miss:
  │     │   ├── Build case payload via build_remote_case_payload()
  │     │   ├── RemoteWorkerClient.run_baseline_case() → POST /api/v1/run_baseline
  │     │   └── Worker subprocess: baseline_benchmark_runner.py
  │     ├── Persist result to results/runs/<run_id>/baseline/<op>/<shape>/
  │     └── Append to summary.csv + profiling_summary.csv
  └── 4. Return exit code
```

### Custom Kernel Benchmark Flow

```
cannbench-lite run-custom --op avg_pool2d --shape S2 --custom-dir output/avg_pool2d
  │
  ├── 1. Validate custom_dir (manifest.json or *Custom/ + *_dsl.py)
  ├── 2. Build subprocess command: python -m cannbench.remote.compile_remote
  │     └── Args: --worker-url, --op-name, --output-dir, --device-id, --run-evaluation|--run-profiling
  ├── 3. compile_remote.py:
  │     ├── Resolve bundled assets (evaluate.py, generate_pybind.py, shared/, template/)
  │     ├── RemoteWorkerClient.create_package() → tar archive
  │     ├── If --run-evaluation: POST /api/v1/compile
  │     │   └── Worker: extract tar → build.sh → install .run → generate_pybind → evaluate (with shared/ reporting)
  │     └── If --run-profiling: POST /api/v1/profile
  │       └── Worker: extract tar → build.sh → install .run → generate_pybind → profiling_runner
  ├── 4. Parse stdout for timing + profiling metrics
  └── 5. Persist to results/runs/<run_id>/custom/<op>/<shape>/<impl_name>/
```

### Compare Flow

```
cannbench-lite compare --results-dir results/runs/<run_id>
  │
  ├── Load baseline summary.csv + profiling_summary.csv
  ├── Load custom summary.csv + profiling_summary.csv
  ├── Match rows by key (op_name, shape_id, dtype, server_name, device)
  ├── Compute deltas: e2e_speedup, task_speedup, vec_ratio_delta, ...
  └── Write compare.csv
```

## Key Design Decisions

1. **Client-server separation**: The CLI requires only `httpx` (no torch/CANN). All NPU-dependent code runs on the worker server.

2. **Subprocess isolation**: Benchmark and profiling runners execute in subprocesses to avoid CUDA/NPU context pollution between runs.

3. **Baseline caching**: Results are cached by a SHA-256 hash of (server, hardware, device, op, shape, dtype, template, warmup, repeat, profiling). Cached results are reused unless `--refresh-baseline` is passed.

4. **Package data via importlib.resources**: Bundled assets (evaluate.py, generate_pybind.py, CppExtension templates) are accessed via `importlib.resources.files()` to work in both editable installs and wheel packages.

5. **python -m invocation**: Subprocess calls use `sys.executable -m cannbench.remote.xxx` instead of file paths, ensuring correct module resolution after package installation.
