# Development Guide

## Project Structure

```
CANNBench-Lite/
├── pyproject.toml              # Package configuration
├── llm.txt                     # LLM agent documentation index
├── repo-wiki/                  # Detailed LLM-optimized documentation
├── src/cannbench/              # Source package (src-layout)
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point
│   ├── scripts/                # Client-side logic
│   ├── remote/                 # Server-side + client communication
│   └── data/                   # Bundled package data
│       ├── registry/           # JSON registries
│       ├── assets/             # evaluate.py, generate_pybind.py, shared/, templates
│       └── __init__.py
└── README.md
```

## Build & Install

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Install as Tool (Production)

```bash
# Basic CLI (client only)
uv tool install .

# With worker server dependencies
uv tool install ".[server]"

# Verify installation
cannbench-lite --help
cannbench-worker --help
```

### Install Editable (Development)

```bash
# Create venv and install in editable mode
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or use uv tool in editable mode
uv tool install -e .
```

### Build Wheel

```bash
uv build
# Output: dist/cannbench_lite-0.1.0-py3-none-any.whl
```

## Package Configuration (pyproject.toml)

```toml
[project]
name = "cannbench-lite"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["httpx>=0.24"]

[project.optional-dependencies]
server = ["fastapi>=0.100", "uvicorn>=0.23", "pandas>=1.5"]
dev = ["ruff>=0.4"]

[project.scripts]
cannbench-lite = "cannbench.cli:main"
cannbench-worker = "cannbench.remote.worker_server:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"cannbench.data" = ["registry/*.json", "assets/*.py", "assets/shared/*.py", "assets/template/**/*"]

[tool.ruff]
target-version = "py310"
line-length = 120
```

### Key Configuration Notes

1. **src-layout**: All source code is under `src/cannbench/`. This prevents accidental imports from the project root.

2. **Package data**: Non-Python files (JSON, py, template files) under `src/cannbench/data/` are included in the wheel via `setuptools.package-data`. This includes `shared/*.py` for the evaluation reporting modules.

3. **Bundled data assets**: `evaluate.py`, `generate_pybind.py`, and `shared/` (reporting.py, case_registry.py) are **data assets**, not importable Python packages. They are deployed to worker servers via tar archive and run as standalone scripts. `evaluate.py` uses `sys.path.insert(0, ...)` to import `shared/` at runtime.
   - `cannbench-lite` → `cannbench.cli:main`
   - `cannbench-worker` → `cannbench.remote.worker_server:main`

4. **Optional dependencies**: `[server]` extra isolates heavy NPU-side dependencies (fastapi, uvicorn, pandas) that are not needed on the client.

## Dependency Architecture

```
Client (any machine):
  cannbench-lite
  └── httpx          ← Only required dependency

Server (NPU machine):
  cannbench-worker
  ├── fastapi        ← [server] extra
  ├── uvicorn        ← [server] extra
  ├── pandas         ← [server] extra
  ├── torch          ← System install (pip install torch torch_npu)
  ├── torch_npu      ← System install
  └── acl            ← CANN toolkit (pip install or bundled)
```

## Code Style

- **Formatter/Linter**: ruff (configured in pyproject.toml)
- **Target Python**: 3.10+
- **Line length**: 120
- **Type hints**: `from __future__ import annotations` for modern union syntax (X | Y)

```bash
# Lint
ruff check src/

# Format
ruff format src/
```

## Bundled Data Access Pattern

All bundled data files are accessed via `importlib.resources` to support both editable installs and wheel packages:

```python
from importlib.resources import files as pkg_files

# Access registry files
registry_path = pkg_files("cannbench.data.registry").joinpath("lightweight_benchmark_registry_v2.json")

# Access asset files
eval_script = pkg_files("cannbench.data.assets").joinpath("evaluate.py")

# Access shared modules
shared_dir = pkg_files("cannbench.data.assets").joinpath("shared")

# Access template directory
template_dir = Path(str(pkg_files("cannbench.data.assets").joinpath("template")))
```

**Do NOT use** `Path(__file__).parent / "..."` patterns — they break when the package is installed as a wheel.

## Subprocess Invocation Pattern

Subprocess calls to internal modules use `python -m` invocation:

```python
import sys
import subprocess

subprocess.run(
    [sys.executable, "-m", "cannbench.remote.baseline_benchmark_runner", ...],
    capture_output=True,
    text=True,
)
```

**Do NOT use** `Path(__file__).resolve().parents[N] / "xxx.py"` patterns — they break after package installation.

## Common Development Tasks

### Adding a New Operator

1. Add operator definition to `src/cannbench/data/registry/lightweight_benchmark_registry_v2.json`:
   ```json
   "new_op": {
     "benchmark_role": "anchor",
     "benchmark_layer": "foundation",
     "category": "activation",
     "pattern": "...",
     "ops_nn_path": "...",
     "torch_npu_api": "torch.nn.functional.new_op",
     "input_format": "[B*S, H]",
     "dtypes": ["float16", "bfloat16"],
     "bottleneck_tags": ["..."],
     "shapes": [{"id": "S1", "shape_tier": "main", "params": {"B_S": 1024, "H": 1024}}]
   }
   ```

2. Add runner template to `src/cannbench/data/registry/runner_templates_v1.json`:
   ```json
   "new_op": {
     "runner_type": "torch_api",
     "api": "torch.nn.functional.new_op",
     "input_builder": "unary_2d",
     "attrs": {},
     "inputs": {"x": {"shape": ["B_S", "H"], "dtype_from_case": true, "distribution": "normal"}}
   }
   ```

3. If the operator needs a new `input_builder`, add it to `baseline_benchmark_runner._prepare_call()`.

4. Verify: `cannbench-lite list --op new_op --json`

### Adding a New Server

Add to `src/cannbench/data/registry/servers.json`:

```json
{
  "default_server": "910b",
  "servers": {
    "910b": { "..." },
    "new_server": {
      "worker_url": "http://192.168.1.100:9027",
      "ssh_host": "new-server",
      "hardware": "Ascend910B",
      "default_device": "0",
      "client_id": "client_new",
      "notes": "New server"
    }
  }
}
```

### Modifying Package Data

After modifying files under `src/cannbench/data/`, reinstall the package for changes to take effect:

```bash
uv pip install -e .   # editable: changes are immediate
# or
uv tool install -e .  # tool install: changes are immediate
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'cannbench'`

- Ensure the package is installed: `uv tool install .` or `uv pip install -e .`
- Check `pip list | grep cannbench`

### `FileNotFoundError` for registry/assets

- Ensure package data is installed: `python -c "from importlib.resources import files; print(files('cannbench.data.registry').joinpath('lightweight_benchmark_registry_v2.json')))"`
- Reinstall: `uv pip install -e . --force-reinstall`

### Worker: `ASCEND_HOME_PATH not set`

- Set the environment variable on the NPU machine: `export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest`

### Worker: `ModuleNotFoundError: No module named 'fastapi'`

- Install with server extra: `uv pip install ".[server]"` or `uv tool install ".[server]"`

### Worker: `ModuleNotFoundError: No module named 'torch_npu'`

- torch and torch_npu must be installed separately on the NPU machine (not via pip dependencies)
- Follow Huawei's installation guide for torch_npu
