#!/usr/bin/env python3
"""Compare baseline and custom benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


KEY_FIELDS = ("op_name", "shape_id", "dtype", "server_name", "device")
COMPARE_FIELDS = [
    "op_name",
    "shape_id",
    "dtype",
    "server_name",
    "device",
    "baseline_status",
    "custom_status",
    "baseline_e2e_ms",
    "custom_e2e_ms",
    "e2e_speedup",
    "baseline_task_duration_us",
    "custom_task_duration_us",
    "task_speedup",
    "vec_ratio_delta",
    "scalar_ratio_delta",
    "mte2_ratio_delta",
    "mte3_ratio_delta",
    "baseline_profiling_path",
    "custom_profiling_path",
    "custom_dir",
]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _key(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def _to_float(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _split_rows(rows: List[Dict[str, str]]) -> tuple[dict, dict]:
    baselines: Dict[tuple, Dict[str, str]] = {}
    customs: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        key = _key(row)
        mode = row.get("mode")
        if mode == "baseline":
            baselines[key] = row
        elif mode == "custom":
            customs.setdefault(key, []).append(row)
    return baselines, customs


def build_compare_rows(results_dir: Path) -> List[Dict[str, object]]:
    return build_compare_rows_from_dirs(results_dir, results_dir)


def build_compare_rows_from_dirs(
    baseline_results_dir: Path,
    custom_results_dir: Path,
) -> List[Dict[str, object]]:
    baseline_rows = _load_csv(baseline_results_dir / "summary.csv")
    baseline_profiling_rows = _load_csv(baseline_results_dir / "profiling_summary.csv")
    custom_rows = _load_csv(custom_results_dir / "summary.csv")
    custom_profiling_rows = _load_csv(custom_results_dir / "profiling_summary.csv")

    baselines, _ = _split_rows(baseline_rows)
    _, customs = _split_rows(custom_rows)

    compare_rows: List[Dict[str, object]] = []
    for key, custom_rows in customs.items():
        baseline = baselines.get(key)
        for custom in custom_rows:
            baseline_ms = _to_float(baseline.get("e2e_median_ms", "") if baseline else "")
            custom_ms = _to_float(custom.get("e2e_median_ms", ""))
            e2e_speedup = ""
            if baseline_ms and custom_ms and custom_ms > 0:
                e2e_speedup = round(baseline_ms / custom_ms, 6)

            baseline_prof = _find_profiling_row(baseline_profiling_rows, baseline)
            custom_prof = _find_profiling_row(custom_profiling_rows, custom)
            baseline_task = _to_float(baseline_prof.get("task_duration_us", "") if baseline_prof else "")
            custom_task = _to_float(custom_prof.get("task_duration_us", "") if custom_prof else "")
            task_speedup = ""
            if baseline_task and custom_task and custom_task > 0:
                task_speedup = round(baseline_task / custom_task, 6)

            compare_rows.append(
                {
                    "op_name": custom["op_name"],
                    "shape_id": custom["shape_id"],
                    "dtype": custom["dtype"],
                    "server_name": custom["server_name"],
                    "device": custom["device"],
                    "baseline_status": baseline.get("status", "") if baseline else "missing",
                    "custom_status": custom.get("status", ""),
                    "baseline_e2e_ms": baseline.get("e2e_median_ms", "") if baseline else "",
                    "custom_e2e_ms": custom.get("e2e_median_ms", ""),
                    "e2e_speedup": e2e_speedup,
                    "baseline_task_duration_us": baseline_prof.get("task_duration_us", "") if baseline_prof else "",
                    "custom_task_duration_us": custom_prof.get("task_duration_us", "") if custom_prof else "",
                    "task_speedup": task_speedup,
                    "vec_ratio_delta": _delta(baseline_prof, custom_prof, "aiv_vec_ratio"),
                    "scalar_ratio_delta": _delta(baseline_prof, custom_prof, "aiv_scalar_ratio"),
                    "mte2_ratio_delta": _delta(baseline_prof, custom_prof, "aiv_mte2_ratio"),
                    "mte3_ratio_delta": _delta(baseline_prof, custom_prof, "aiv_mte3_ratio"),
                    "baseline_profiling_path": baseline.get("profiling_path", "") if baseline else "",
                    "custom_profiling_path": custom.get("profiling_path", ""),
                    "custom_dir": custom.get("custom_dir", ""),
                }
            )

    return compare_rows


def write_compare_csv(
    results_dir: Path | None = None,
    out_csv: Path | None = None,
    *,
    baseline_results_dir: Path | None = None,
    custom_results_dir: Path | None = None,
) -> Path:
    if baseline_results_dir is not None and custom_results_dir is not None:
        compare_rows = build_compare_rows_from_dirs(baseline_results_dir, custom_results_dir)
        target = out_csv or (custom_results_dir / "compare.csv")
    else:
        if results_dir is None:
            raise ValueError("results_dir is required when baseline/custom results dirs are not provided")
        compare_rows = build_compare_rows(results_dir)
        target = out_csv or (results_dir / "compare.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARE_FIELDS)
        writer.writeheader()
        writer.writerows(compare_rows)
    return target


def _find_profiling_row(rows: List[Dict[str, str]], summary_row: Dict[str, str] | None) -> Dict[str, str] | None:
    if not summary_row:
        return None
    for row in rows:
        if (
            row.get("run_id") == summary_row.get("run_id")
            and row.get("op_name") == summary_row.get("op_name")
            and row.get("shape_id") == summary_row.get("shape_id")
            and row.get("server_name") == summary_row.get("server_name")
            and row.get("device") == summary_row.get("device")
        ):
            return row
    return None


def _delta(base: Dict[str, str] | None, custom: Dict[str, str] | None, field: str) -> float | str:
    b = _to_float(base.get(field, "") if base else "")
    c = _to_float(custom.get(field, "") if custom else "")
    if b is None or c is None:
        return ""
    return round(c - b, 6)
