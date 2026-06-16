# Registry Schema

Three JSON files define the benchmark configuration. They are loaded by `cannbench.scripts.registry_utils`.

---

## 1. `lightweight_benchmark_registry_v2.json`

Loaded by `BenchmarkRegistry.from_file(path)`.

### Top-Level Structure

```json
{
  "version": "2.0.0",
  "description": "...",
  "hardware": "Ascend 910B",
  "registry_goals": ["..."],
  "model_sources": { "...": { "hidden_size": ..., "intermediate_size": ..., ... } },
  "schema": {
    "operator_required_fields": ["benchmark_role", "benchmark_layer", "category", ...],
    "shape_required_fields": ["id", "shape_tier", "params"]
  },
  "benchmark_sets": {
    "main_benchmark": ["gelu", "layer_norm_v3", ...],
    "transfer_probes": ["relu", "rms_norm", ...],
    "stretch_goals": ["flash_attention_score"]
  },
  "operators": {
    "<op_name>": { /* OperatorDefinition */ }
  }
}
```

### OperatorDefinition

```json
{
  "benchmark_role": "anchor" | "probe",
  "benchmark_layer": "foundation" | "fusion" | "challenge",
  "category": "activation" | "normalization" | "linear" | "pooling" | "embedding" | "attention" | "loss" | "quantization",
  "pattern": "unary activation | binary fusion | ...",
  "ops_nn_path": "activation/gelu",
  "torch_npu_api": "torch.nn.functional.gelu",
  "description": "Human-readable description",
  "input_format": "[B*S, H]",
  "dtypes": ["float16", "bfloat16"],
  "bottleneck_tags": ["memory_bound", "elementwise", "approx_math"],
  "knowledge_targets": ["relu"],
  "shapes": [ /* ShapeDefinition[] */ ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `benchmark_role` | string | Yes | `anchor` = primary benchmark target; `probe` = knowledge transfer test |
| `benchmark_layer` | string | Yes | `foundation` = base op; `fusion` = fused op; `challenge` = hard fusion |
| `category` | string | Yes | Operator category for filtering |
| `pattern` | string | Yes | Computational pattern description |
| `ops_nn_path` | string | Yes | Path in operator namespace |
| `torch_npu_api` | string | Yes | Reference PyTorch/torch_npu API path |
| `description` | string | No | Human-readable description |
| `input_format` | string | Yes | Abstract input shape notation |
| `dtypes` | string[] | Yes | Supported data types |
| `bottleneck_tags` | string[] | Yes | Performance bottleneck classification |
| `knowledge_targets` | string[] | No | Operators that can benefit from this op's knowledge |
| `shapes` | ShapeDefinition[] | Yes | Shape variants for benchmarking |

### ShapeDefinition

```json
{
  "id": "S1",
  "shape_tier": "small" | "main" | "large",
  "source": "qwen2.5_7b",
  "params": {
    "B_S": 1024,
    "H": 1024
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Shape identifier (e.g., S1, S2, S3, S4) |
| `shape_tier` | string | Yes | Size classification; `main` is default for benchmarking |
| `source` | string | No | Model source name (references `model_sources`) |
| `params` | object | Yes | Concrete dimension values; keys are referenced by runner template `inputs[].shape` |

### model_sources

Model parameter references used to derive realistic shape sizes:

```json
{
  "qwen2.5_7b": {
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "vocab_size": 152064
  }
}
```

### benchmark_sets

Named sets of operator names used for `--set` filtering:

- `main_benchmark`: Core operators for primary benchmarking
- `transfer_probes`: Operators for knowledge transfer validation
- `stretch_goals`: Advanced operators (e.g., flash attention)

---

## 2. `runner_templates_v1.json`

Loaded by `RunnerTemplateRegistry.from_file(path)`.

### Top-Level Structure

```json
{
  "version": "1.1.0",
  "description": "...",
  "templates": {
    "<op_name>": { /* RunnerTemplate */ }
  }
}
```

### RunnerTemplate

```json
{
  "runner_type": "torch_api",
  "api": "torch.nn.functional.gelu",
  "input_builder": "unary_2d",
  "attrs": {},
  "profiling_supported": true,
  "supports_custom_remote": false,
  "case_signature": {
    "shape_params": ["B_S", "H"]
  },
  "inputs": {
    "x": {
      "shape": ["B_S", "H"],
      "dtype_from_case": true,
      "distribution": "normal"
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `runner_type` | string | Yes | Currently always `"torch_api"` |
| `api` | string | Yes | Python callable path (e.g., `"torch.nn.functional.gelu"`) |
| `input_builder` | string | Yes | Determines how inputs are passed to the API (see input_builder values below) |
| `attrs` | object | Yes | Additional keyword arguments for the API call |
| `profiling_supported` | bool | No | Whether hardware profiling is supported |
| `supports_custom_remote` | bool | No | Whether custom remote benchmarking is supported |
| `case_signature` | object | No | Metadata about which shape params are used |
| `inputs` | object | Yes | Input tensor specifications |

### Input Specification

```json
{
  "shape": ["B_S", "H"],
  "dtype_from_case": true,
  "distribution": "normal"
}
```

Or with explicit dtype:
```json
{
  "shape": [4, "N", "H"],
  "dtype": "int64",
  "distribution": "randint",
  "low": 0,
  "high": 1024
}
```

| Field | Type | Description |
|-------|------|-------------|
| `shape` | (int\|str)[] | Dimension values; strings reference `params` keys from the ShapeDefinition |
| `dtype_from_case` | bool | If true, dtype comes from the operator's `dtypes` list |
| `dtype` | string | Explicit dtype (used when `dtype_from_case` is false/absent) |
| `distribution` | string | Tensor fill strategy: `normal`, `ones`, `zeros`, `randint`, `bool_bernoulli` |
| `low` | int | Lower bound for `randint` distribution |
| `high` | int | Upper bound for `randint` distribution |
| `threshold` | float | Threshold for `bool_bernoulli` distribution |

### input_builder Values

The `input_builder` field determines how tensors are assembled into API call arguments. Handled in `baseline_benchmark_runner._prepare_call()`:

| input_builder | Input Tensors | Args Pattern |
|---------------|---------------|--------------|
| `unary_2d` | x | `[x], attrs` |
| `layer_norm_2d` | x, weight, bias | `[x, [H], weight, bias], attrs` |
| `rms_norm_2d` | x, weight | `[x, weight, eps], attrs` |
| `matmul_2d` | a, b | `[a, b], attrs` |
| `bmm_3d` | a, b | `[a, b], attrs` |
| `softmax_4d` | x | `[x], attrs` |
| `add_layer_norm_2d` | x1, x2, gamma, beta | `[x1, x2, gamma, beta, eps], attrs` |
| `add_rms_norm_2d` | x1, x2, gamma | `[x1, x2, gamma, eps], attrs` |
| `swiglu_2d` | x | `[x], attrs` |
| `gated_2d` | x | `[x], attrs` |
| `avg_pool2d_4d` | x | `[x], {"kernel_size": k, "stride": s, "padding": p}` |
| `conv2d_nchw` | x, weight | `[x, weight], {"stride": s, "padding": p}` |
| `gather_rows_2d` | x, index | `[x, dim, index], {}` |
| `embedding_bag_2d` | indices, weight, offsets | `[indices, weight, offsets], attrs` |
| `cross_entropy_2d` | logits, target | `[logits, target], attrs` |
| `masked_softmax_with_rel_pos_bias` | x, atten_mask, relative_pos_bias, head_num | Complex, see source |
| `scaled_masked_softmax` | x, mask | `[x, mask, scale, fixed_triu_mask], {}` |
| `add_rms_norm_quant_2d` | x1, x2, gamma, scales1 | `[x1, x2, gamma, scales1, None, None, None], attrs` |
| `add_rms_norm_dynamic_quant_2d` | x1, x2, gamma | `[x1, x2, gamma, eps], attrs` |
| `fusion_attention_bsnd` | query, key, value, head_num, ... | Complex, see source |

---

## 3. `servers.json`

Loaded by `ServerRegistry.from_file(path)`.

### Structure

```json
{
  "default_server": "910b",
  "servers": {
    "<server_name>": { /* ServerConfig */ }
  }
}
```

### ServerConfig

```json
{
  "worker_url": "http://127.0.0.1:9027",
  "ssh_host": "910b",
  "hardware": "Ascend910B",
  "default_device": "7",
  "client_id": "gjz_910b",
  "notes": "Current primary remote worker."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `worker_url` | string | Yes | HTTP URL of the worker server |
| `ssh_host` | string | No | SSH hostname for the NPU machine |
| `hardware` | string | No | Hardware identifier (e.g., "Ascend910B") |
| `default_device` | int\|str | No | Default NPU device id (default: "0") |
| `client_id` | string | No | Client identifier for requests (default: server_name) |
| `notes` | string | No | Human-readable notes |

### Resolved ServerInfo

`ServerRegistry.get(name)` returns a `ServerInfo` dataclass:

```python
@dataclass(frozen=True)
class ServerInfo:
    name: str              # server_name (or default_server)
    worker_url: str        # from servers[name].worker_url
    ssh_host: str | None   # from servers[name].ssh_host
    hardware: str | None   # from servers[name].hardware
    default_device: str    # from servers[name].default_device (default: "0")
    client_id: str         # from servers[name].client_id (default: server_name)
    notes: str | None      # from servers[name].notes
```
