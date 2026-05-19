#!/usr/bin/env python3
"""Custom remote runner using server/compile_remote.py."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from case_expander import ExpandedCase
from result_writer import append_csv, write_json


def _discover_op_root(custom_dir: Path, op_name: str) -> Path:
    direct = custom_dir / op_name
    if direct.exists():
        return direct
    if custom_dir.name == op_name:
        return custom_dir
    raise FileNotFoundError(f"cannot find operator root for {op_name} under {custom_dir}")


def _normalize_compile_output_dir(custom_dir: Path, op_name: str) -> Path:
    """Return the directory that should be passed to compile_remote.py.

    compile_remote.py expects a layout like:
      <output_dir>/<op_name>/<op_name>_dsl.py
      <output_dir>/<op_name>/*Custom

    Users may naturally pass either:
    - the parent output directory
    - the operator root directory itself
    """
    if (custom_dir / op_name).exists():
        return custom_dir
    if custom_dir.name == op_name:
        return custom_dir.parent
    return custom_dir


def _validate_custom_dir(custom_dir: Path, op_name: str) -> Dict[str, Any]:
    if not custom_dir.exists():
        raise FileNotFoundError(f"custom directory not found: {custom_dir}")

    manifest_path = custom_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "manifest_path": str(manifest_path),
            "project_dir": manifest.get("project_dir"),
            "dsl_file": manifest.get("dsl_file"),
            "impl_name": manifest.get("impl_name", custom_dir.name),
        }

    op_root = _discover_op_root(custom_dir, op_name)
    dsl_file = op_root / f"{op_name}_dsl.py"
    custom_candidates = list(op_root.glob("*Custom"))
    if not dsl_file.exists():
        raise FileNotFoundError(f"dsl file not found: {dsl_file}")
    if not custom_candidates:
        raise FileNotFoundError(f"no *Custom project directory found under: {op_root}")
    return {
        "manifest_path": None,
        "project_dir": str(custom_candidates[0]),
        "dsl_file": str(dsl_file),
        "impl_name": custom_dir.name,
    }


def _render_avg_pool2d_reference(case: ExpandedCase, dtype: str) -> str:
    params = case.raw_shape["params"]
    torch_dtype = f"torch.{dtype}"
    return f"""import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    2D Average Pooling: compute average over kernel_size x kernel_size windows
    with given stride and padding.
    \"\"\"
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool(x)

batch_size = {params["N"]}
channels = {params["C"]}
height = {params["H"]}
width = {params["W"]}
kernel_size = {params["kernel"]}
stride = {params["stride"]}
padding = 0

def get_inputs():
    x = torch.rand(batch_size, channels, height, width, dtype={torch_dtype})
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
"""


def _prepare_case_specific_custom_dir(case: ExpandedCase, custom_dir: Path, dtype: str) -> tuple[Path, Path | None]:
    """Create a staged custom output dir when a fixed sample needs case injection."""
    if case.op_name != "avg_pool2d":
        return custom_dir, None

    op_root = _discover_op_root(custom_dir, case.op_name)
    temp_root = Path(tempfile.mkdtemp(prefix=f"bench_custom_{case.op_name}_"))
    staged_output_root = temp_root / "output"
    staged_output_root.mkdir(parents=True, exist_ok=True)
    staged_op_root = staged_output_root / case.op_name
    shutil.copytree(op_root, staged_op_root, dirs_exist_ok=True)

    reference_path = staged_op_root / f"{case.op_name}_reference.py"
    reference_path.write_text(_render_avg_pool2d_reference(case, dtype), encoding="utf-8")
    return staged_op_root, temp_root


def _build_command(
    *,
    server_name: str,
    worker_url: str,
    client_id: str,
    op_name: str,
    compile_output_dir: Path,
    device_id: str,
    profiling: bool,
) -> list[str]:
    compile_remote = Path(__file__).resolve().parents[1] / "remote" / "compile_remote.py"
    cmd = [
        sys.executable,
        str(compile_remote),
        "--worker-url",
        worker_url,
        "--client-id",
        client_id,
        "--op-name",
        op_name,
        "--output-dir",
        str(compile_output_dir),
        "--device-id",
        str(device_id),
    ]
    if profiling:
        cmd.append("--run-profiling")
    else:
        cmd.append("--run-evaluation")
    return cmd


@dataclass
class CustomRunOutput:
    result: Dict[str, Any]
    profiling: Dict[str, Any]
    summary_row: Dict[str, Any]
    profiling_row: Dict[str, Any]


def run_custom_case(
    case: ExpandedCase,
    *,
    server_name: str,
    worker_url: str,
    client_id: str,
    device_id: str,
    dtype: str,
    custom_dir: Path,
    profiling: bool,
) -> CustomRunOutput:
    custom_dir = custom_dir.resolve()
    staged_custom_dir, temp_root = _prepare_case_specific_custom_dir(case, custom_dir, dtype)
    try:
        validation = _validate_custom_dir(staged_custom_dir, case.op_name)
        compile_output_dir = _normalize_compile_output_dir(staged_custom_dir, case.op_name)
        cmd = _build_command(
            server_name=server_name,
            worker_url=worker_url,
            client_id=client_id,
            op_name=case.op_name,
            compile_output_dir=compile_output_dir,
            device_id=str(device_id),
            profiling=profiling,
        )

        completed = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)

    status = "success" if completed.returncode == 0 else "compile_or_eval_failed"
    error_text = ""
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout).strip()

    log_text = "\n".join(part for part in [completed.stdout or "", completed.stderr or ""] if part)
    timing = _extract_timing(log_text)
    profiling_summary = _extract_profiling(log_text)
    profiling_available = bool(profiling_summary)
    profiling_payload = {
        "profiling_available": profiling_available,
        "timing": timing,
        "profiling": profiling_summary,
        "error": "" if (profiling and completed.returncode == 0 and profiling_available) else (error_text or ("profiling_not_captured" if profiling else "")),
    }

    result = {
        "mode": "custom",
        "status": status,
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "dtype": dtype,
        "server_name": server_name,
        "device": str(device_id),
        "worker_url": worker_url,
        "custom_dir": str(custom_dir),
        "compile_output_dir": str(compile_output_dir),
        "impl_name": validation["impl_name"],
        "command": cmd,
        "timing": timing,
        "profiling": profiling_summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "error": error_text,
        "created_at": datetime.now().isoformat(),
    }

    summary_row = {
        "run_id": "",
        "mode": "custom",
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "benchmark_layer": case.benchmark_layer,
        "benchmark_role": case.benchmark_role,
        "category": case.category,
        "dtype": dtype,
        "server_name": server_name,
        "device": str(device_id),
        "backend": "remote_custom",
        "status": status,
        "correctness": "",
        "e2e_median_ms": timing.get("custom_median_ms", ""),
        "e2e_p95_ms": "",
        "profiling_available": profiling_available,
        "profiling_path": "",
        "custom_dir": str(custom_dir),
        "worker_url": worker_url,
        "created_at": result["created_at"],
        "source": "measured",
        "cache_key": "",
    }

    profiling_row = {
        "run_id": "",
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "server_name": server_name,
        "device": str(device_id),
        "task_duration_us": profiling_summary.get("task_duration_us", ""),
        "aiv_vec_ratio": profiling_summary.get("aiv_vec_ratio", ""),
        "aiv_scalar_ratio": profiling_summary.get("aiv_scalar_ratio", ""),
        "aiv_mte2_ratio": profiling_summary.get("aiv_mte2_ratio", ""),
        "aiv_mte3_ratio": profiling_summary.get("aiv_mte3_ratio", ""),
        "aiv_icache_miss_rate": profiling_summary.get("aiv_icache_miss_rate", ""),
        "aic_mac_ratio": profiling_summary.get("aic_mac_ratio", ""),
        "cube_utilization": profiling_summary.get("cube_utilization", ""),
        "profiling_error": profiling_payload.get("error", ""),
    }

    return CustomRunOutput(result=result, profiling=profiling_payload, summary_row=summary_row, profiling_row=profiling_row)


def persist_custom_output(
    output: CustomRunOutput,
    *,
    run_root: Path,
    run_id: str,
    op_name: str,
    shape_id: str,
    impl_name: str,
) -> None:
    case_dir = run_root / "custom" / op_name / shape_id / impl_name
    result_path = case_dir / "result.json"
    profiling_path = case_dir / "profiling.json"
    write_json(result_path, output.result)
    write_json(profiling_path, output.profiling)

    summary_row = dict(output.summary_row)
    summary_row["run_id"] = run_id
    summary_row["profiling_path"] = str(profiling_path)
    profiling_row = dict(output.profiling_row)
    profiling_row["run_id"] = run_id

    append_csv(run_root / "summary.csv", [summary_row])
    append_csv(run_root / "profiling_summary.csv", [profiling_row])


def _extract_timing(stdout_text: str) -> Dict[str, Any]:
    timing: Dict[str, Any] = {}
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("INFO:__main__:  Reference:"):
            timing["ref_median_ms"] = _parse_ms_value(line.split("Reference:")[1].strip())
        elif line.startswith("INFO:__main__:  Custom:"):
            timing["custom_median_ms"] = _parse_ms_value(line.split("Custom:")[1].strip())
        elif line.startswith("INFO:__main__:  Speedup:"):
            value = line.split("Speedup:")[1].strip().rstrip("x")
            try:
                timing["speedup"] = float(value)
            except ValueError:
                pass
    return timing


def _parse_ms_value(text: str) -> float | str:
    value = text.replace("ms", "").strip()
    try:
        return float(value)
    except ValueError:
        return text


def _extract_profiling(stdout_text: str) -> Dict[str, Any]:
    profiling: Dict[str, Any] = {}
    in_metrics = False
    for line in stdout_text.splitlines():
        line = line.strip()
        if "PROFILING METRICS:" in line:
            in_metrics = True
            continue
        if in_metrics:
            if not line.startswith("INFO:__main__:"):
                continue
            payload = line.split("INFO:__main__:", 1)[1].strip()
            if not payload or ":" not in payload:
                continue
            key, value = payload.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "raw_rows":
                continue
            try:
                profiling[key] = float(value)
            except ValueError:
                profiling[key] = value
    return profiling
