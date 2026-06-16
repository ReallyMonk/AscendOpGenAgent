# Worker Server REST API

Base URL: `http://<host>:<port>`

The worker server is a FastAPI application defined in `cannbench.remote.worker_server`.

## Health & Status

### `GET /health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "cann_available": true,
  "active_requests": 0,
  "available_devices": 4,
  "total_devices": 4
}
```

### `GET /api/v1/status`

Detailed server status.

**Response**:
```json
{
  "active_requests": 0,
  "max_workers": 4,
  "log_dir": "/tmp/cannbench-logs",
  "available_devices": 4,
  "total_devices": 4,
  "device_list": [0, 1, 2, 3]
}
```

---

## Compilation & Evaluation

### `POST /api/v1/compile`

Compile a custom operator package and optionally evaluate it.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `package` | file | Yes | TAR archive containing the operator project |
| `client_id` | str | Yes | Client identifier |
| `op_name` | str | Yes | Operator name |
| `timeout` | int | No | Request timeout in seconds (default: 1800) |
| `device_id` | int | No | Target NPU device id |

**TAR Package Structure**:
```
package.tar
  ├── manifest.json           ← {"op_name": "...", "client_id": "...", "run_evaluation": true/false}
  ├── <OpName>Custom/         ← AscendC project directory
  │   ├── build.sh
  │   ├── op_kernel/
  │   │   └── <op_name>_custom.cpp
  │   └── ...
  ├── <op_name>_dsl.py        ← (optional) DSL descriptor
  ├── evaluate.py             ← (if run_evaluation=true) Evaluation script
  ├── generate_pybind.py      ← (if run_evaluation=true) PyBind generator
  ├── shared/                 ← (if run_evaluation=true) Shared eval modules
  │   ├── __init__.py
  │   ├── reporting.py
  │   └── case_registry.py
  ├── output/<op_name>/
  │   ├── <op_name>_reference.py
  │   ├── <op_name>_custom.py
  │   └── <op_name>.cpp
  └── template/CppExtension/  ← (if run_evaluation=true) CppExtension template
```

**Worker Processing Pipeline**:
1. Extract TAR → find `*Custom/` directory
2. Run `build.sh` → produces `build_out/`
3. If `manifest.run_evaluation == true`:
   a. Install `custom_opp_ubuntu_aarch64.run`
   b. Copy vendors to output directory
   c. Run `generate_pybind.py` (if present)
   d. Run `evaluate.py` with `ASCEND_CUSTOM_OPP_PATH` and `ASCEND_DEVICE_ID` env vars
      (uses `shared/` for structured reporting + case registry)

**Response (success)**:
```json
{
  "success": true,
  "stage": "evaluation",
  "build_log": "...",
  "eval_log": "...",
  "log_dir": "/path/to/logs/client_opname_timestamp",
  "device_id": 7
}
```

**Response (failure)**:
```json
{
  "success": false,
  "stage": "compilation",
  "log": "Build failed: ...",
  "log_dir": "/path/to/logs/..."
}
```

Possible `stage` values: `compilation`, `evaluation`, `error`, `health_check`, `http_error`, `network_error`

---

## Profiling

### `POST /api/v1/profile`

Compile and profile a custom operator with hardware counters.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `package` | file | Yes | TAR archive (same format as compile) |
| `client_id` | str | Yes | Client identifier |
| `op_name` | str | Yes | Operator name |
| `num_trials` | int | No | Number of profiling trials (default: 10) |
| `timeout` | int | No | Request timeout in seconds (default: 1800) |
| `device_id` | int | No | Target NPU device id |

**Worker Processing Pipeline**:
1. Extract TAR → find `*Custom/` directory
2. Run `build.sh`
3. Install `custom_opp_ubuntu_aarch64.run`
4. Copy vendors to output directory
5. Run `generate_pybind.py` (if present)
6. Run profiling subprocess: `python -m cannbench.remote.profiling_runner`

**Response (success)**:
```json
{
  "success": true,
  "stage": "profiling",
  "timing": {
    "ref_median_ms": 0.1234,
    "custom_median_ms": 0.0567,
    "speedup": 2.1764
  },
  "profiling": {
    "task_duration_us": 567.123,
    "aiv_vec_ratio": 0.65,
    "aiv_scalar_ratio": 0.02,
    "aiv_mte2_ratio": 0.15,
    "aiv_mte3_ratio": 0.03,
    "aiv_icache_miss_rate": 0.01
  },
  "build_log": "...",
  "profile_log": "...",
  "log_dir": "/path/to/logs/...",
  "device_id": 7
}
```

---

## Baseline Benchmark

### `POST /api/v1/run_baseline`

Run a PyTorch/torch_npu baseline benchmark case.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_json` | file | Yes | JSON file containing the benchmark case payload |
| `server_name` | str | No | Server name (default: "default") |
| `device_id` | int | No | NPU device id (default: 0) |
| `warmup` | int | No | Warmup iterations (default: 20) |
| `repeat` | int | No | Measurement iterations (default: 100) |
| `profiling` | str | No | "true" or "false" (default: "false") |

**Case JSON Payload** (the `case_json` file content):
```json
{
  "op_name": "gelu",
  "shape_id": "S1",
  "api": "torch.nn.functional.gelu",
  "runner_type": "torch_api",
  "input_builder": "unary_2d",
  "attrs": {},
  "shape_params": {"B_S": 1024, "H": 1024},
  "inputs": {
    "x": {
      "shape": [1024, 1024],
      "dtype": "float32",
      "distribution": "normal",
      "low": null,
      "high": null,
      "threshold": null
    }
  }
}
```

**Worker Processing**:
The server runs `python -m cannbench.remote.baseline_benchmark_runner` as a subprocess with `--case-json`, `--device-id`, `--warmup`, `--repeat`, `--result-file`, and optionally `--profiling`.

**Response (success)**:
```json
{
  "success": true,
  "timing": {
    "median_ms": 0.123456,
    "p95_ms": 0.156789
  },
  "profiling": {
    "task_duration_us": 123.456,
    "aiv_vec_ratio": 0.65,
    "error": ""
  },
  "log_dir": "/path/to/logs/...",
  "device_id": 7,
  "stdout": "...",
  "stderr": "..."
}
```

---

## Project Generation

### `POST /api/v1/generate_project`

Generate AscendC project structure using msopgen. **Optional feature** — requires `gen_project` module.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_json` | file | Yes | Project definition JSON |
| `op_name` | str | Yes | Operator name |
| `client_id` | str | No | Client identifier (default: "default_client") |
| `force` | str | No | Force overwrite "true"/"false" (default: "false") |

**Response**: TAR archive containing the generated project directory.

**Error (501)**: If `gen_project` module is not installed on the worker.
