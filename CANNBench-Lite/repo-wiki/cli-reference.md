# CLI Reference

## Global Options

```
cannbench-lite [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--registry` | Bundled `lightweight_benchmark_registry_v2.json` | Benchmark registry JSON path |
| `--runner-templates` | Bundled `runner_templates_v1.json` | Runner templates JSON path |
| `--servers` | Bundled `servers.json` | Server configuration JSON path |

Default paths are resolved at runtime via `importlib.resources.files("cannbench.data.registry")`.

---

## `list` — List benchmark operators and shapes

### Usage

```bash
cannbench-lite list [OPTIONS]
```

### Options

| Flag | Type | Description |
|------|------|-------------|
| `--set` | str | Filter by benchmark set name (e.g., `main_benchmark`) |
| `--layer` | str | Filter by benchmark layer (e.g., `foundation`, `fusion`, `challenge`) |
| `--role` | str | Filter by benchmark role (e.g., `anchor`, `probe`) |
| `--category` | str | Filter by category (e.g., `activation`, `normalization`, `linear`) |
| `--op` | str | Filter by operator name (exact match) |
| `--shape` | str | Only show a specific shape id (e.g., `S1`, `S2`) |
| `--server` | str | Show resolved server info for this server name |
| `--json` | flag | Emit JSON output instead of human-readable text |

### Output Formats

**Text** (default):
```
server=910b worker_url=http://127.0.0.1:9027 default_device=7
matched_cases=1
- gelu:S1 [foundation/anchor/activation] runner=torch_api api=torch.nn.functional.gelu inputs=(x=[1024, 1024])
```

**JSON** (`--json`):
```json
{
  "server": {"name": "910b", "worker_url": "http://127.0.0.1:9027", ...},
  "rows": [
    {
      "op_name": "gelu",
      "shape_id": "S1",
      "shape_tier": "small",
      "benchmark_layer": "foundation",
      "benchmark_role": "anchor",
      "category": "activation",
      "runner_type": "torch_api",
      "api": "torch.nn.functional.gelu",
      "inputs": [{"name": "x", "shape": [1024, 1024]}],
      "server_name": "910b",
      "server_worker_url": "http://127.0.0.1:9027",
      "default_device": "7"
    }
  ]
}
```

Rows with `status: "missing_runner_template"` indicate operators in the registry without a corresponding runner template.

---

## `run-baseline` — Run baseline benchmark

### Usage

```bash
cannbench-lite run-baseline [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--set` | str | — | Filter by benchmark set name |
| `--layer` | str | — | Filter by benchmark layer |
| `--role` | str | — | Filter by benchmark role |
| `--category` | str | — | Filter by category |
| `--op` | str | — | Filter by operator name |
| `--shape` | str | — | Only run a specific shape id |
| `--all-shapes` | flag | — | Run all shapes for matched operators (default: only `main` tier) |
| `--server` | str | — | Server name from servers.json |
| `--device` | str | — | Override device id (defaults to server's `default_device`) |
| `--dtype` | str | — | Override dtype (defaults to operator's first dtype) |
| `--warmup` | int | `20` | Warmup iterations |
| `--repeat` | int | `100` | Measurement iterations |
| `--profiling` | flag | — | Enable hardware profiling (acl.prof + msprof) |
| `--refresh-baseline` | flag | — | Ignore cached results, force re-run |
| `--run-id` | str | timestamp | Run identifier |
| `--results-root` | str | `results` | Root results directory |

### Shape Selection Logic

- If `--shape` is specified: only that shape is run.
- If `--all-shapes` is specified: all shapes for matched operators are run.
- Otherwise: only shapes with `shape_tier == "main"` are run (skips non-main shapes when an operator has multiple).

### Baseline Cache

Results are cached under `<results-root>/baseline_cache/<server>/<op>/<shape>/dtype_<dtype>_device_<device>_<cache_key>/`. The cache key is a SHA-256 hash truncated to 16 hex chars of:

```json
{
  "server_name": "910b",
  "hardware": "Ascend910B",
  "device_id": "7",
  "op_name": "gelu",
  "shape_id": "S1",
  "dtype": "float32",
  "runner_template": {...},
  "warmup": 20,
  "repeat": 100,
  "profiling": false
}
```

If a cache hit occurs and `--refresh-baseline` is not set, the cached `result.json` + `profiling.json` are reused and `source` is set to `"reused_cache"`.

---

## `run-custom` — Run custom kernel benchmark

### Usage

```bash
cannbench-lite run-custom --op OP --shape SHAPE --custom-dir DIR [OPTIONS]
```

### Required Options

| Flag | Description |
|------|-------------|
| `--op` | Operator name |
| `--shape` | Shape id |
| `--custom-dir` | Custom implementation directory path |

### Optional Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--server` | str | — | Server name from servers.json |
| `--device` | str | — | Override device id |
| `--dtype` | str | — | Override dtype |
| `--profiling` | flag | — | Enable hardware profiling |
| `--run-id` | str | timestamp | Run identifier |
| `--results-root` | str | `results` | Root results directory |

### Custom Directory Layout

The `--custom-dir` must contain one of:

**Option A: With manifest.json**
```
custom-dir/
  manifest.json        ← {"project_dir": "...", "dsl_file": "...", "impl_name": "..."}
  <op_name>/
    <op_name>_dsl.py
    <op_name>Custom/
      build.sh
      op_kernel/
        <op_name>_custom.cpp
```

**Option B: Auto-discovery (no manifest)**
```
custom-dir/
  <op_name>/
    <op_name>_dsl.py
    *Custom/            ← First glob match used
      build.sh
      op_kernel/
        <op_name>_custom.cpp
```

Or if `custom-dir.name == op_name`:
```
custom-dir/             ← directory name matches op_name
  <op_name>_dsl.py
  *Custom/
```

---

## `compare` — Compare baseline and custom results

### Usage

```bash
# Single run directory (contains both baseline/ and custom/ subdirs)
cannbench-lite compare --results-dir results/runs/<run_id>

# Separate baseline and custom directories
cannbench-lite compare --baseline-results-dir PATH --custom-results-dir PATH [--out-csv PATH]
```

### Options

| Flag | Description |
|------|-------------|
| `--results-dir` | Single run directory containing both baseline and custom results |
| `--baseline-results-dir` | Baseline run directory (must pair with `--custom-results-dir`) |
| `--custom-results-dir` | Custom run directory (must pair with `--baseline-results-dir`) |
| `--out-csv` | Output CSV path (defaults to `<results-dir>/compare.csv`) |

### Compare Output Fields

| Field | Description |
|-------|-------------|
| `op_name` | Operator name |
| `shape_id` | Shape identifier |
| `dtype` | Data type |
| `server_name` | Server name |
| `device` | Device id |
| `baseline_status` | Baseline run status |
| `custom_status` | Custom run status |
| `baseline_e2e_ms` | Baseline end-to-end median latency (ms) |
| `custom_e2e_ms` | Custom end-to-end median latency (ms) |
| `e2e_speedup` | baseline_e2e_ms / custom_e2e_ms |
| `baseline_task_duration_us` | Baseline task duration from profiling (us) |
| `custom_task_duration_us` | Custom task duration from profiling (us) |
| `task_speedup` | baseline_task_duration / custom_task_duration |
| `vec_ratio_delta` | Change in aiv_vec_ratio |
| `scalar_ratio_delta` | Change in aiv_scalar_ratio |
| `mte2_ratio_delta` | Change in aiv_mte2_ratio |
| `mte3_ratio_delta` | Change in aiv_mte3_ratio |
| `baseline_profiling_path` | Path to baseline profiling JSON |
| `custom_profiling_path` | Path to custom profiling JSON |
| `custom_dir` | Custom implementation directory |

---

## Worker Server CLI

```
cannbench-worker --port PORT --log-dir DIR [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--port` | int | `9001` | Server port |
| `--host` | str | `0.0.0.0` | Server host |
| `--log-dir` | str | **required** | Directory for storing logs |
| `--max-workers` | int | `4` | Maximum concurrent workers |
| `--devices` | str | `"0"` | Comma-separated NPU device IDs (e.g., `"0,1,2,3"`) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ASCEND_HOME_PATH` | Yes | Path to CANN installation directory |
