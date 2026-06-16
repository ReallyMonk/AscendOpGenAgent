# Module Index

Package: `cannbench` (src-layout: `src/cannbench/`)

---

## `cannbench.cli`

CLI entry point. `cannbench-lite` command invokes `cannbench.cli:main`.

| Symbol | Type | Description |
|--------|------|-------------|
| `build_parser()` | function | Build argparse parser with 4 subcommands |
| `cmd_list(args)` | function | `list` subcommand handler |
| `cmd_run_baseline(args)` | function | `run-baseline` subcommand handler |
| `cmd_run_custom(args)` | function | `run-custom` subcommand handler |
| `cmd_compare(args)` | function | `compare` subcommand handler |
| `main(argv)` | function | Entry point: parse args, dispatch to handler, return exit code |
| `_get_default_registry_path()` | function | Resolve default registry path via importlib.resources |
| `_get_default_templates_path()` | function | Resolve default templates path via importlib.resources |
| `_get_default_servers_path()` | function | Resolve default servers config path via importlib.resources |

---

## `cannbench.scripts.registry_utils`

Registry loading and querying.

| Symbol | Type | Description |
|--------|------|-------------|
| `load_json(path)` | function | Load JSON file and return dict |
| `ServerInfo` | dataclass | Resolved server config: name, worker_url, ssh_host, hardware, default_device, client_id, notes |
| `ServerRegistry` | class | Load and query servers.json |
| `ServerRegistry.from_file(path)` | classmethod | Create from JSON file |
| `ServerRegistry.get(name)` | method | Get ServerInfo by name (None = default_server) |
| `RunnerTemplateRegistry` | class | Load and query runner_templates_v1.json |
| `RunnerTemplateRegistry.from_file(path)` | classmethod | Create from JSON file |
| `RunnerTemplateRegistry.get(op_name)` | method | Get template dict for an operator |
| `RunnerTemplateRegistry.has(op_name)` | method | Check if template exists for an operator |
| `BenchmarkRegistry` | class | Load and query benchmark registry |
| `BenchmarkRegistry.from_file(path)` | classmethod | Create from JSON file |
| `BenchmarkRegistry.get_operator(op_name)` | method | Get operator dict by name |
| `BenchmarkRegistry.get_shape(op_name, shape_id)` | method | Get shape dict by op and shape id |
| `BenchmarkRegistry.iter_operators(...)` | method | Iterate operators with optional filters: set_name, layer, role, category, op_name |

---

## `cannbench.scripts.case_expander`

Expand registry shapes into concrete executable input specifications.

| Symbol | Type | Description |
|--------|------|-------------|
| `ExpandedInput` | dataclass | Resolved input: name, shape (List[int]), spec (Dict) |
| `ExpandedCase` | dataclass | Fully resolved benchmark case |
| `expand_case(op_name, operator, shape, template)` | function | Create ExpandedCase from registry data |
| `_resolve_dim(value, params)` | function | Resolve a dimension value (int or param key) |
| `_resolve_value(value, params)` | function | Recursively resolve values with `_from_param` suffix support |

### ExpandedCase Fields

| Field | Type | Description |
|-------|------|-------------|
| `op_name` | str | Operator name |
| `shape_id` | str | Shape identifier |
| `benchmark_layer` | str | foundation / fusion / challenge |
| `benchmark_role` | str | anchor / probe |
| `category` | str | Operator category |
| `runner_type` | str | e.g., "torch_api" |
| `api` | str \| None | Python callable path |
| `attrs` | dict | Resolved attributes |
| `inputs` | List[ExpandedInput] | Resolved input tensor specs |
| `raw_shape` | dict | Original shape definition from registry |
| `raw_operator` | dict | Original operator definition from registry |
| `raw_template` | dict | Original runner template |

---

## `cannbench.scripts.baseline_runner`

Remote baseline execution and caching.

| Symbol | Type | Description |
|--------|------|-------------|
| `BaselineRunOutput` | dataclass | result, profiling, summary_row, profiling_row, cache_hit |
| `build_remote_case_payload(case, dtype)` | function | Build JSON payload for POST /api/v1/run_baseline |
| `run_baseline_case_remote(case, ...)` | function | Execute baseline via remote client with cache logic |
| `persist_run_output(output, ...)` | function | Write result.json, profiling.json, summary.csv, profiling_summary.csv |
| `_make_cache_key(...)` | function | Generate SHA-256 cache key from run parameters |
| `_build_summary_row(result, case)` | function | Build summary CSV row from result dict |
| `_build_profiling_row(result, profiling_payload, case)` | function | Build profiling CSV row |

---

## `cannbench.scripts.custom_remote_runner`

Custom kernel remote execution.

| Symbol | Type | Description |
|--------|------|-------------|
| `CustomRunOutput` | dataclass | result, profiling, summary_row, profiling_row |
| `run_custom_case(case, ...)` | function | Execute custom kernel benchmark via compile_remote subprocess |
| `persist_custom_output(output, ...)` | function | Write custom result files |
| `_validate_custom_dir(custom_dir, op_name)` | function | Check custom dir structure, return validation info |
| `_discover_op_root(custom_dir, op_name)` | function | Find operator root directory |
| `_normalize_compile_output_dir(custom_dir, op_name)` | function | Determine output_dir for compile_remote |
| `_extract_timing(stdout_text)` | function | Parse timing metrics from compile_remote stdout |
| `_extract_profiling(stdout_text)` | function | Parse profiling metrics from compile_remote stdout |
| `_build_command(...)` | function | Build subprocess command for compile_remote |

---

## `cannbench.scripts.compare_results`

Baseline vs custom result comparison.

| Symbol | Type | Description |
|--------|------|-------------|
| `KEY_FIELDS` | tuple | Fields used as match key: op_name, shape_id, dtype, server_name, device |
| `COMPARE_FIELDS` | list | Output CSV column names |
| `build_compare_rows(results_dir)` | function | Build compare rows from a single results dir |
| `build_compare_rows_from_dirs(baseline_dir, custom_dir)` | function | Build compare rows from separate dirs |
| `write_compare_csv(results_dir, out_csv, ...)` | function | Write compare.csv |
| `_load_csv(path)` | function | Load CSV file as list of dicts |
| `_key(row)` | function | Extract match key tuple from a row |
| `_delta(base, custom, field)` | function | Compute numeric delta between baseline and custom |

---

## `cannbench.scripts.result_writer`

Simple JSON and CSV writers.

| Symbol | Type | Description |
|--------|------|-------------|
| `ensure_parent(path)` | function | Create parent directories |
| `write_json(path, payload)` | function | Write dict as indented JSON |
| `append_csv(path, rows)` | function | Append dicts as CSV rows (write header if new file) |

---

## `cannbench.remote.worker_server`

FastAPI worker server. `cannbench-worker` command invokes `cannbench.remote.worker_server:main`.

| Symbol | Type | Description |
|--------|------|-------------|
| `app` | FastAPI | FastAPI application instance |
| `SERVER_CONFIG` | dict | Global server configuration (log_dir, max_workers, device_pool, etc.) |
| `DevicePool` | class | Async NPU device pool (acquire/release specific or any device) |
| `RequestQueue` | class | Async request queue with semaphore-based concurrency control |
| `process_compilation_request(...)` | async function | Handle compile+evaluate request |
| `process_profiling_request(...)` | async function | Handle compile+profile request |
| `sanitize_json_value(value)` | function | Recursively convert NaN/Inf to None for JSON serialization |
| `check_cann_environment()` | function | Verify ASCEND_HOME_PATH is set and exists |
| `create_log_directory(client_id, op_name)` | function | Create timestamped log directory |
| `run_build_script(project_dir, log_file)` | function | Execute build.sh and capture output |
| `run_evaluation_script(eval_script, ...)` | function | Execute evaluate.py with CANN env |
| `_get_python_executable()` | function | Return sys.executable |
| `main()` | function | CLI entry point for cannbench-worker |

---

## `cannbench.remote.remote_client`

Async HTTP client for worker communication.

| Symbol | Type | Description |
|--------|------|-------------|
| `RemoteWorkerClient` | class | Async client with retry + health check |
| `RemoteWorkerClient(worker_url, client_id)` | init | Initialize client |
| `.compile_operator(package_data, ...)` | async method | POST /api/v1/compile with retry |
| `.profile_operator(package_data, ...)` | async method | POST /api/v1/profile with retry |
| `.run_baseline_case(case_payload, ...)` | async method | POST /api/v1/run_baseline |
| `.check_health()` | async method | GET /health |
| `.get_status()` | async method | GET /api/v1/status |
| `.generate_project(op_name, project_json_path, ...)` | async method | POST /api/v1/generate_project |
| `.create_package(op_name, project_dir, ...)` | method | Build tar archive for upload |

### Retry Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_MAX_RETRIES` | 3 | Max retry attempts |
| `DEFAULT_RETRY_BACKOFF` | 3.0 | Exponential backoff base (seconds) |
| `DEFAULT_MAX_RETRY_WAIT` | 30.0 | Max wait between retries |
| `HEALTH_CHECK_TIMEOUT` | 10.0 | Health check request timeout |
| `HEALTH_RETRY_DELAY` | 10.0 | Delay between health check retries |
| `HEALTH_MAX_RETRIES` | 3 | Max health check attempts |

---

## `cannbench.remote.compile_remote`

CLI helper for remote compilation. Invoked as subprocess by `custom_remote_runner`.

| Symbol | Type | Description |
|--------|------|-------------|
| `compile_remote(worker_url, client_id, op_name, output_dir, ...)` | async function | Main compile logic |
| `_prepare_eval_files(output_path, op_name)` | function | Resolve bundled evaluate.py + generate_pybind.py + shared/ + templates |
| `_get_data_asset_path(*parts)` | function | Resolve path in cannbench.data.assets via importlib.resources |
| `main()` | function | CLI entry point (argparse) |

---

## `cannbench.remote.baseline_benchmark_runner`

Subprocess runner for single baseline case on NPU.

| Symbol | Type | Description |
|--------|------|-------------|
| `run_case(case, device_id, warmup, repeat, profiling)` | function | Execute a single benchmark case |
| `_resolve_callable(api_path)` | function | Import and return Python callable from dotted path |
| `_materialize_tensor(spec, device)` | function | Create torch.Tensor from spec on device |
| `_prepare_call(case, tensors)` | function | Assemble (args, kwargs) for API call based on input_builder |
| `_run_e2e(fn, args, kwargs, warmup, repeat)` | function | Run warmup + measurement iterations |
| `_run_profiling(fn, args, kwargs, device, device_id, task_type, num_trials)` | function | Run acl.prof profiling |
| `main()` | function | Subprocess entry point (--case-json, --device-id, --result-file) |

---

## `cannbench.remote.profiling_runner`

Hardware profiling using acl.prof + msprof.

| Symbol | Type | Description |
|--------|------|-------------|
| `ProfilingContext` | class | Context manager for acl.prof init/start/stop |
| `run_profiling(op_name, device_id, output_dir, eval_script_dir, ...)` | function | Execute profiling and return timing + metrics |
| `parse_op_summary(csv_path, task_type)` | function | Parse op_summary CSV and extract metrics |
| `msprof_export(path_prefix, path_suffix)` | function | Run msprof --export |
| `locate_op_summary_file(output_dir)` | function | Find op_summary_*.csv in profiling output |

---

## Bundled Data Assets

These modules are **not** importable as `cannbench.*` packages. They are bundled as data files
under `cannbench.data.assets/` and deployed to the worker server via tar archive for standalone
execution.

---

## `cannbench.data.assets.evaluate` (standalone script)

AscendC custom operator evaluation tool. Runs on the NPU worker server after tar extraction.
Uses `sys.path.insert(0, ...)` to import `shared/` modules.

| Symbol | Type | Description |
|--------|------|-------------|
| `AscendBackend` | class | Core evaluation backend: correctness check + performance measurement |
| `AscendBackend(eval_src, ref_src, seed_num, device)` | init | Initialize with eval/reference code strings |
| `.evaluate_correctness()` | method | Run correctness check, returns (bool, str, list[dict]) |
| `.measure_performance(model_name, ...)` | method | Measure median latency (ms) for a model |
| `.compare_performance(num_warmup, num_perf_trials)` | method | Compare ref vs custom median latency |
| `.cleanup()` | method | Release context and NPU cache |
| `evaluate_operator(eval_src_path, ref_src_path, project_root_path, ...)` | function | Full evaluation with structured report persistence |
| `replay_evaluation(registry_path, work_dir, op_name, ...)` | function | Replay failed cases from cases.json |
| `setup_ascend_runtime_environment(project_root)` | function | Set ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH |
| `resolve_work_dir(op_name, output_path)` | function | Resolve operator work directory |
| `set_seed(seed)` | function | Set torch + torch_npu random seed |

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `op_name` | positional | — | Operator name |
| `--output-path` | str | None | Operator output directory (default: `./output/{op_name}`) |
| `--device` | str | None | NPU device (default: from `ASCEND_DEVICE_ID` or `npu:0`) |
| `--replay` | str | None | Path to cases.json for replay mode |
| `--filter-status` | str | None | Comma-separated statuses to replay |
| `--case-id` | str | None | Replay a single case by ID |

---

## `cannbench.data.assets.shared.reporting` (standalone module)

Shared reporting module for evaluation skills. Provides unified data structures and file writers.

| Symbol | Type | Description |
|--------|------|-------------|
| `TestCaseMeta` | dataclass | Test case metadata: case_id, op_name, shape, dtype, generator, seed |
| `EvalResult` | dataclass | Evaluation result: status, match_rate, max_diff, speedup, timing |
| `CaseLogCapture` | class | Context manager to capture logging output during evaluation |
| `write_test_cases_csv(output_path, cases)` | function | Write test_cases.csv |
| `write_eval_report_csv(output_path, results)` | function | Write evaluation_report.csv |
| `write_summary_md(output_path, cases, results, ...)` | function | Generate human-readable summary.md report |
| `append_case_log(output_path, case_meta, result, ...)` | function | Append per-case detail to run_details.log |
| `write_run_log_header(output_path, ...)` | function | Write session header to run_details.log |
| `write_run_log_summary(output_path, results, ...)` | function | Append summary section to run_details.log |
| `parse_reference_source(source)` | function | Parse reference.py to extract generator and value_range |

---

## `cannbench.data.assets.shared.case_registry` (standalone module)

Case registry for persisting case spec + seed to cases.json, enabling seed-based replay.

| Symbol | Type | Description |
|--------|------|-------------|
| `CaseRecord` | dataclass | Single test case record: case_id, seed, generator, status, spec |
| `CaseRegistry` | class | Registry of test cases, serializes to/from cases.json |
| `CaseRegistry(skill_name, op_name, config)` | init | Create registry |
| `.add(record)` | method | Add a CaseRecord |
| `.update_status(case_id, status)` | method | Update status of a record |
| `.save(output_path)` | method | Write cases.json, return file path |
| `CaseRegistry.load(path)` | classmethod | Load from cases.json |
| `.filter(status, case_ids)` | method | Filter records by status and/or case_ids |

---

## `cannbench.data.assets.generate_pybind` (standalone script)

PyBind code generator and compiler for AscendC operators.

| Symbol | Type | Description |
|--------|------|-------------|
| `generate_pybind_bindings(work_dir, op_cpp)` | function | Generate PyBind bindings, compile wheel, install |
| `resolve_work_dir(op_name, output_path)` | function | Resolve operator work directory |

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `op_name` | positional | — | Operator name |
| `--output-path` | str | None | Operator output directory (default: `./output/{op_name}`) |
