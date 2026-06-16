# Data Flow & Result Formats

## Results Directory Structure

```
<results-root>/                          ← defaults to ./results
├── baseline_cache/                      ← Cached baseline results
│   └── <server_name>/
│       └── <op_name>/
│           └── <shape_id>/
│               └── dtype_<dtype>_device_<device>_<cache_key>/
│                   ├── baseline_result.json
│                   └── baseline_profiling.json
└── runs/
    └── <run_id>/                        ← e.g., 2025-01-15_143022
        ├── manifest.json                ← Run metadata
        ├── summary.csv                  ← Combined baseline + custom summary rows
        ├── profiling_summary.csv        ← Combined profiling rows
        ├── baseline/
        │   └── <op_name>/
        │       └── <shape_id>/
        │           ├── result.json
        │           └── profiling.json
        ├── custom/
        │   └── <op_name>/
        │       └── <shape_id>/
        │           └── <impl_name>/
        │               ├── result.json
        │               └── profiling.json
        └── compare.csv                  ← Output of compare command
```

---

## File Formats

### manifest.json

Written at the start of each run by `cli.py`:

```json
{
  "run_id": "2025-01-15_143022",
  "registry_path": "/path/to/registry.json",
  "runner_template_path": "/path/to/templates.json",
  "server_config_path": "/path/to/servers.json",
  "server_name": "910b",
  "worker_url": "http://127.0.0.1:9027",
  "device": "7",
  "warmup": 20,
  "repeat": 100,
  "profiling": true,
  "created_at": "2025-01-15T14:30:22.123456",
  "mode": "baseline"
}
```

The `mode` field is only present in custom runs (`"custom"`). Baseline runs omit it.

### Baseline result.json

Written by `baseline_runner.persist_run_output()`:

```json
{
  "mode": "baseline",
  "source": "measured" | "reused_cache",
  "cache_key": "a1b2c3d4e5f6g7h8",
  "server_name": "910b",
  "hardware": "Ascend910B",
  "device": "7",
  "op_name": "gelu",
  "shape_id": "S1",
  "dtype": "float16",
  "status": "success" | "failed",
  "correctness": "not_checked",
  "e2e_median_ms": 0.123456,
  "e2e_p95_ms": 0.156789,
  "warmup": 20,
  "repeat": 100,
  "profiling_enabled": true,
  "worker_url": "http://127.0.0.1:9027",
  "remote_stage": "baseline_runner",
  "remote_log_dir": "/path/to/worker/logs/...",
  "remote_error": "",
  "created_at": "2025-01-15T14:30:22.123456"
}
```

### Baseline profiling.json

Contains the raw profiling payload from the worker:

```json
{
  "task_duration_us": 123.456,
  "aiv_vec_ratio": 0.65,
  "aiv_scalar_ratio": 0.02,
  "aiv_mte2_ratio": 0.15,
  "aiv_mte3_ratio": 0.03,
  "aiv_icache_miss_rate": 0.01,
  "error": ""
}
```

For cube-type operators, the fields are different:
```json
{
  "task_duration_us": 456.789,
  "aicore_time_us": 400.0,
  "aic_mac_ratio": 0.75,
  "aic_scalar_ratio": 0.01,
  "aic_mte1_ratio": 0.05,
  "aic_mte2_ratio": 0.10,
  "aic_fixpipe_ratio": 0.02,
  "aic_icache_miss_rate": 0.005,
  "cube_utilization": 85.5,
  "error": ""
}
```

### Custom result.json

Written by `custom_remote_runner.persist_custom_output()`:

```json
{
  "mode": "custom",
  "status": "success" | "compile_or_eval_failed",
  "op_name": "avg_pool2d",
  "shape_id": "S2",
  "dtype": "float16",
  "server_name": "910b",
  "device": "7",
  "worker_url": "http://127.0.0.1:9027",
  "custom_dir": "/path/to/output/avg_pool2d",
  "compile_output_dir": "/path/to/output",
  "impl_name": "avg_pool2d",
  "command": ["python", "-m", "cannbench.remote.compile_remote", ...],
  "timing": {
    "ref_median_ms": 0.5,
    "custom_median_ms": 0.3,
    "speedup": 1.67
  },
  "profiling": {
    "task_duration_us": 123.456,
    "aiv_vec_ratio": 0.70,
    "..."
  },
  "stdout": "...",
  "stderr": "...",
  "returncode": 0,
  "error": "",
  "created_at": "2025-01-15T14:35:00.123456"
}
```

### summary.csv

Combined baseline + custom rows in a single CSV. Field names:

| Field | Description |
|-------|-------------|
| `run_id` | Run identifier |
| `mode` | `"baseline"` or `"custom"` |
| `op_name` | Operator name |
| `shape_id` | Shape identifier |
| `benchmark_layer` | foundation / fusion / challenge |
| `benchmark_role` | anchor / probe |
| `category` | Operator category |
| `dtype` | Data type |
| `server_name` | Server name |
| `device` | Device id |
| `backend` | `torch_api` for baseline, `remote_custom` for custom |
| `status` | success / failed |
| `correctness` | Correctness check result |
| `e2e_median_ms` | End-to-end median latency |
| `e2e_p95_ms` | End-to-end P95 latency |
| `profiling_available` | Whether profiling data exists |
| `profiling_path` | Path to profiling JSON |
| `custom_dir` | Custom implementation directory (custom rows only) |
| `worker_url` | Worker server URL |
| `created_at` | Timestamp |
| `source` | `"measured"` or `"reused_cache"` (baseline rows only) |
| `cache_key` | Cache key hash (baseline rows only) |

### profiling_summary.csv

| Field | Description |
|-------|-------------|
| `run_id` | Run identifier |
| `op_name` | Operator name |
| `shape_id` | Shape identifier |
| `server_name` | Server name |
| `device` | Device id |
| `task_duration_us` | Task duration in microseconds |
| `aiv_vec_ratio` | Vector compute ratio |
| `aiv_scalar_ratio` | Scalar compute ratio |
| `aiv_mte2_ratio` | Memory-to-Vector transfer ratio |
| `aiv_mte3_ratio` | Vector-to-Memory transfer ratio |
| `aiv_icache_miss_rate` | Instruction cache miss rate |
| `aic_mac_ratio` | MAC utilization ratio (cube ops) |
| `cube_utilization` | Cube utilization percentage |
| `profiling_error` | Error message if profiling failed |

### compare.csv

Generated by `cannbench-lite compare`. See [cli-reference.md](cli-reference.md) for field descriptions.

---

## Cache Mechanism

### Cache Key Generation

```python
def _make_cache_key(*, server_name, hardware, device_id, op_name, shape_id,
                     dtype, runner_template, warmup, repeat, profiling) -> str:
    payload = {k: v for k, v in locals().items()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
```

### Cache Lookup

1. Compute cache key from run parameters
2. Check `<cache_root>/<server>/<op>/<shape>/dtype_<dtype>_device_<device>_<key>/baseline_result.json`
3. If exists AND `result.status == "success"` AND `--refresh-baseline` is not set: reuse
4. Otherwise: execute remote baseline and write new cache entry

### Cache Invalidation

- `--refresh-baseline` flag forces re-execution
- Changing any parameter in the cache key (server, device, dtype, warmup, repeat, etc.) creates a new cache entry
- No automatic TTL or eviction

---

## Data Flow: Baseline

```
Registry JSON
     │
     ▼
BenchmarkRegistry.iter_operators(filters) ──> (op_name, operator)[]
     │
     ▼ (join with RunnerTemplateRegistry)
expand_case(op_name, operator, shape, template) ──> ExpandedCase
     │
     ▼
build_remote_case_payload(case, dtype) ──> Dict (case JSON payload)
     │
     ▼
RemoteWorkerClient.run_baseline_case(payload, ...) ──> POST /api/v1/run_baseline
     │
     ▼
Worker: baseline_benchmark_runner.run_case(case, device_id, warmup, repeat, profiling)
     │
     ├── _resolve_callable(api) ──> Python callable
     ├── _materialize_tensor(spec, device) ──> torch.Tensor on NPU
     ├── _prepare_call(case, tensors) ──> (args, kwargs)
     ├── _run_e2e(fn, args, kwargs, warmup, repeat) ──> {"median_ms", "p95_ms"}
     └── (optional) _run_profiling(...) ──> {"task_duration_us", "aiv_vec_ratio", ...}
     │
     ▼
{"success": True, "timing": {...}, "profiling": {...}}
     │
     ▼
run_baseline_case_remote() ──> BaselineRunOutput
     │
     ▼
persist_run_output() ──> result.json + profiling.json + summary.csv + profiling_summary.csv
```

## Data Flow: Custom

```
custom_dir/  (user-provided)
     │
     ▼
_validate_custom_dir() ──> {"project_dir", "dsl_file", "impl_name"}
     │
     ▼
compile_remote.py subprocess:
     │
     ├── _prepare_eval_files() ──> resolve bundled assets from importlib.resources
     │   ├── evaluate.py
     │   ├── generate_pybind.py
     │   ├── shared/reporting.py + shared/case_registry.py + shared/__init__.py
     │   └── template/CppExtension/ (csrc/op.cpp + setup.py + pytorch_npu_helper.hpp)
     ├── RemoteWorkerClient.create_package() ──> tar bytes
     ├── POST /api/v1/compile or /api/v1/profile
     │
     ▼
Worker: process_compilation_request() or process_profiling_request()
     │
     ▼
compile_remote.py stdout:
  "INFO:__main__:  Reference: 0.5ms"
  "INFO:__main__:  Custom: 0.3ms"
  "INFO:__main__:  Speedup: 1.67x"
  "PROFILING METRICS:"
  "INFO:__main__:task_duration_us: 123.456"
  ...
     │
     ▼
_extract_timing(stdout) + _extract_profiling(stdout) ──> CustomRunOutput
     │
     ▼
persist_custom_output() ──> result.json + profiling.json + summary.csv + profiling_summary.csv
```
