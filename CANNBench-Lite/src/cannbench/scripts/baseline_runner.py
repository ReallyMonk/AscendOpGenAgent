#!/usr/bin/env python3
"""Remote baseline runner and cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from cannbench.scripts.case_expander import ExpandedCase
from cannbench.scripts.result_writer import append_csv, write_json


def _make_cache_key(
    *,
    server_name: str,
    hardware: str,
    device_id: str,
    op_name: str,
    shape_id: str,
    dtype: str,
    runner_template: Dict[str, Any],
    warmup: int,
    repeat: int,
    profiling: bool,
) -> str:
    payload = {
        "server_name": server_name,
        "hardware": hardware,
        "device_id": device_id,
        "op_name": op_name,
        "shape_id": shape_id,
        "dtype": dtype,
        "runner_template": runner_template,
        "warmup": warmup,
        "repeat": repeat,
        "profiling": profiling,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class BaselineRunOutput:
    result: Dict[str, Any]
    profiling: Dict[str, Any]
    summary_row: Dict[str, Any]
    profiling_row: Dict[str, Any]
    cache_hit: bool


def build_remote_case_payload(case: ExpandedCase, dtype: str) -> Dict[str, Any]:
    return {
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "api": case.api,
        "runner_type": case.runner_type,
        "input_builder": case.raw_template["input_builder"],
        "attrs": case.attrs,
        "shape_params": case.raw_shape["params"],
        "inputs": {
            inp.name: {
                "shape": inp.shape,
                "dtype": inp.spec.get("dtype", dtype),
                "distribution": inp.spec.get("distribution", "normal"),
                "low": inp.spec.get("low"),
                "high": inp.spec.get("high"),
                "threshold": inp.spec.get("threshold"),
            }
            for inp in case.inputs
        },
    }


def run_baseline_case_remote(
    case: ExpandedCase,
    *,
    server_name: str,
    hardware: str,
    worker_url: str,
    device_id: str,
    client,
    dtype: str,
    warmup: int,
    repeat: int,
    profiling: bool,
    cache_root: Path,
    refresh_baseline: bool,
) -> BaselineRunOutput:
    cache_key = _make_cache_key(
        server_name=server_name,
        hardware=hardware,
        device_id=device_id,
        op_name=case.op_name,
        shape_id=case.shape_id,
        dtype=dtype,
        runner_template=case.raw_template,
        warmup=warmup,
        repeat=repeat,
        profiling=profiling,
    )
    cache_dir = cache_root / server_name / case.op_name / case.shape_id / f"dtype_{dtype}_device_{device_id}_{cache_key}"
    result_path = cache_dir / "baseline_result.json"
    profiling_path = cache_dir / "baseline_profiling.json"

    if not refresh_baseline and result_path.exists() and profiling_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        profiling_payload = json.loads(profiling_path.read_text(encoding="utf-8"))
        if result.get("status") == "success":
            result["source"] = "reused_cache"
            summary_row = _build_summary_row(result, case)
            profiling_row = _build_profiling_row(result, profiling_payload, case)
            return BaselineRunOutput(result, profiling_payload, summary_row, profiling_row, cache_hit=True)

    case_payload = build_remote_case_payload(case, dtype)
    remote_result = client.run_baseline_case(
        case_payload,
        server_name=server_name,
        device_id=int(device_id),
        warmup=warmup,
        repeat=repeat,
    )
    if hasattr(remote_result, "__await__"):
        raise RuntimeError("run_baseline_case_remote expects a resolved remote client call result")

    timing = remote_result.get("timing", {})
    profiling_payload = remote_result.get("profiling", {})
    result = {
        "mode": "baseline",
        "source": "measured",
        "cache_key": cache_key,
        "server_name": server_name,
        "hardware": hardware,
        "device": str(device_id),
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "dtype": dtype,
        "status": "success" if remote_result.get("success") else "failed",
        "correctness": "not_checked",
        "e2e_median_ms": timing.get("median_ms", ""),
        "e2e_p95_ms": timing.get("p95_ms", ""),
        "warmup": warmup,
        "repeat": repeat,
        "profiling_enabled": profiling,
        "worker_url": worker_url,
        "remote_stage": remote_result.get("stage", ""),
        "remote_log_dir": remote_result.get("log_dir", ""),
        "remote_error": remote_result.get("log", ""),
        "created_at": datetime.now().isoformat(),
    }
    write_json(result_path, result)
    write_json(profiling_path, profiling_payload)
    summary_row = _build_summary_row(result, case)
    profiling_row = _build_profiling_row(result, profiling_payload, case)
    return BaselineRunOutput(result, profiling_payload, summary_row, profiling_row, cache_hit=False)


def _build_summary_row(result: Dict[str, Any], case: ExpandedCase) -> Dict[str, Any]:
    return {
        "run_id": "",
        "mode": "baseline",
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "benchmark_layer": case.benchmark_layer,
        "benchmark_role": case.benchmark_role,
        "category": case.category,
        "dtype": result["dtype"],
        "server_name": result["server_name"],
        "device": result["device"],
        "backend": case.runner_type,
        "status": result["status"],
        "correctness": result["correctness"],
        "e2e_median_ms": result["e2e_median_ms"],
        "e2e_p95_ms": result["e2e_p95_ms"],
        "profiling_available": bool(result.get("profiling_enabled")),
        "profiling_path": "",
        "custom_dir": "",
        "worker_url": result.get("worker_url", ""),
        "created_at": result["created_at"],
        "source": result["source"],
        "cache_key": result["cache_key"],
    }


def _build_profiling_row(result: Dict[str, Any], profiling_payload: Dict[str, Any], case: ExpandedCase) -> Dict[str, Any]:
    return {
        "run_id": "",
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "server_name": result["server_name"],
        "device": result["device"],
        "task_duration_us": profiling_payload.get("task_duration_us", ""),
        "aiv_vec_ratio": profiling_payload.get("aiv_vec_ratio", ""),
        "aiv_scalar_ratio": profiling_payload.get("aiv_scalar_ratio", ""),
        "aiv_mte2_ratio": profiling_payload.get("aiv_mte2_ratio", ""),
        "aiv_mte3_ratio": profiling_payload.get("aiv_mte3_ratio", ""),
        "aiv_icache_miss_rate": profiling_payload.get("aiv_icache_miss_rate", ""),
        "aic_mac_ratio": profiling_payload.get("aic_mac_ratio", ""),
        "cube_utilization": profiling_payload.get("cube_utilization", ""),
        "profiling_error": profiling_payload.get("error", ""),
    }


def persist_run_output(
    output: BaselineRunOutput,
    *,
    run_root: Path,
    run_id: str,
    server_name: str,
    worker_url: str,
    op_name: str,
    shape_id: str,
) -> None:
    case_dir = run_root / "baseline" / op_name / shape_id
    result_path = case_dir / "result.json"
    profiling_path = case_dir / "profiling.json"
    write_json(result_path, output.result)
    write_json(profiling_path, output.profiling)

    summary_row = dict(output.summary_row)
    summary_row["run_id"] = run_id
    summary_row["profiling_path"] = str(profiling_path)
    summary_row["worker_url"] = worker_url
    profiling_row = dict(output.profiling_row)
    profiling_row["run_id"] = run_id

    append_csv(run_root / "summary.csv", [summary_row])
    append_csv(run_root / "profiling_summary.csv", [profiling_row])
