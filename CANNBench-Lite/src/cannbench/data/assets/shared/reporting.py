"""Shared reporting module for evaluation skills.

Provides unified data structures and file writers for test case metadata,
evaluation results, and human-readable summary reports.
"""

import csv
import logging
import os
import re
from dataclasses import dataclass, fields
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List


@dataclass
class TestCaseMeta:
    """Metadata for a single test case. All fields are required."""
    case_id: str
    op_name: str
    torch_api_name: str  # "" for ascendc-evaluation
    shape: str           # e.g. "[32, 1024]"; multiple inputs separated by ";"
    dtype: str           # e.g. "torch.float32"; multiple inputs separated by ";"
    value_range: str     # e.g. "[0.0, 1.0)"; multiple inputs separated by ";"
    generator: str       # "uniform" / "double_ended" / "PDS"
    seed: str
    case_source: str     # "csv" / "op_desc"


@dataclass
class EvalResult:
    """Evaluation result for a single test case.

    Optional fields (match_rate through speedup) depend on status:
    - consistent: all fields populated
    - inconsistent: precision metrics populated, performance fields empty
    - op_failed / torch_failed / compare_failed: all optional fields empty
    """
    case_id: str
    status: str             # consistent/inconsistent/op_failed/torch_failed/compare_failed
    match_rate: str         # e.g. "99.98%" or ""
    max_diff: str           # e.g. "1.23e-05" or ""
    mean_diff: str          # e.g. "4.56e-07" or ""
    max_relative_diff: str  # e.g. "2.34e-04" or ""
    ref_time_ms: str        # e.g. "0.234" or ""
    execute_time_ms: str    # e.g. "0.156" or ""
    speedup: str            # e.g. "1.50x" or ""
    timing_mode: str        # "benchmark_median" / "single_run"
    ref_execute_device: str # "npu:0" / "cpu"
    execute_device: str     # "npu:0"


class CaseLogCapture:
    """Context manager that captures INFO+ logging output during a case execution.

    Attaches a StringIO handler to the root logger on enter, removes it on exit.
    Captures INFO/WARNING/ERROR/CRITICAL — DEBUG is intentionally excluded to avoid
    triggering formatting errors in third-party code that uses {} style with logging.
    The captured text is available via the ``output`` property after exit.
    """

    def __init__(self, logger_name=None):
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        self.handler.setLevel(logging.INFO)
        self.logger = logging.getLogger(logger_name)
        self._original_level = None

    def __enter__(self):
        # Temporarily lower logger level so INFO messages reach our handler
        self._original_level = self.logger.level
        if self.logger.level == logging.NOTSET or self.logger.level > logging.INFO:
            self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.handler.close()
        if self._original_level is not None:
            self.logger.setLevel(self._original_level)

    @property
    def output(self) -> str:
        return self.buffer.getvalue()


def _write_dataclass_csv(file_path: Path, items: list) -> Path:
    """Write a list of dataclass instances to CSV."""
    if not items:
        return file_path
    os.makedirs(file_path.parent, exist_ok=True)
    field_names = [f.name for f in fields(items[0])]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for item in items:
            writer.writerow({f.name: getattr(item, f.name) for f in fields(item)})
    return file_path


def write_test_cases_csv(output_path: Path, cases: List[TestCaseMeta]) -> Path:
    """Write test case metadata to test_cases.csv under output_path."""
    return _write_dataclass_csv(output_path / "test_cases.csv", cases)


def write_eval_report_csv(output_path: Path, results: List[EvalResult]) -> Path:
    """Write evaluation results to evaluation_report.csv under output_path."""
    return _write_dataclass_csv(output_path / "evaluation_report.csv", results)


def write_summary_md(
    output_path: Path,
    cases: List[TestCaseMeta],
    results: List[EvalResult],
    skill_name: str = "",
) -> Path:
    """Generate a human-readable summary.md report."""
    os.makedirs(output_path, exist_ok=True)
    file_path = output_path / "summary.md"

    # Build lookup: case_id -> TestCaseMeta
    case_map = {c.case_id: c for c in cases}

    # Classify results
    passed = [r for r in results if r.status == "consistent"]
    failed = [r for r in results if r.status != "consistent"]
    total = len(results)
    pass_pct = f"{len(passed) / total * 100:.1f}" if total else "0.0"
    fail_pct = f"{len(failed) / total * 100:.1f}" if total else "0.0"

    op_name = cases[0].op_name if cases else "unknown"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# Evaluation Report: {op_name}\n")
    lines.append("## Overview")
    lines.append(f"- **Op name**: {op_name}")
    lines.append(f"- **Skill**: {skill_name}")
    lines.append(f"- **Date**: {timestamp}")
    lines.append(f"- **Total cases**: {total}")
    lines.append(f"- **Passed**: {len(passed)} ({pass_pct}%)")
    lines.append(f"- **Failed**: {len(failed)} ({fail_pct}%)")
    lines.append("")

    # Results by dtype
    dtype_stats: dict = {}
    for r in results:
        c = case_map.get(r.case_id)
        dtype = c.dtype if c else "unknown"
        if dtype not in dtype_stats:
            dtype_stats[dtype] = {"total": 0, "passed": 0, "failed": 0}
        dtype_stats[dtype]["total"] += 1
        if r.status == "consistent":
            dtype_stats[dtype]["passed"] += 1
        else:
            dtype_stats[dtype]["failed"] += 1

    lines.append("## Results by dtype")
    lines.append("| dtype | total | passed | failed | pass_rate |")
    lines.append("|-------|-------|--------|--------|-----------|")
    for dtype, stats in dtype_stats.items():
        rate = f"{stats['passed'] / stats['total'] * 100:.1f}%" if stats["total"] else "0.0%"
        lines.append(f"| {dtype} | {stats['total']} | {stats['passed']} | {stats['failed']} | {rate} |")
    lines.append("")

    # Performance (passed cases only)
    lines.append("## Performance (passed cases only)")
    lines.append("| case_id | dtype | shape | ref_time_ms | execute_time_ms | speedup | timing_mode | ref_execute_device | execute_device |")
    lines.append("|---------|-------|-------|-------------|-----------------|---------|-------------|-------------------|----------------|")
    for r in passed:
        c = case_map.get(r.case_id)
        dtype = c.dtype if c else ""
        shape = c.shape if c else ""
        lines.append(
            f"| {r.case_id} | {dtype} | {shape} | {r.ref_time_ms} | {r.execute_time_ms} "
            f"| {r.speedup} | {r.timing_mode} | {r.ref_execute_device} | {r.execute_device} |"
        )
    lines.append("")

    # Failed cases
    lines.append("## Failed cases")
    lines.append("| case_id | status | match_rate | detail |")
    lines.append("|---------|--------|------------|--------|")
    for r in failed:
        detail_parts = []
        if r.max_diff:
            detail_parts.append(f"max_diff={r.max_diff}")
        if r.match_rate:
            detail_parts.append(f"match_rate={r.match_rate}")
        detail = ", ".join(detail_parts) if detail_parts else ""
        lines.append(f"| {r.case_id} | {r.status} | {r.match_rate} | {detail} |")
    lines.append("")

    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


def append_case_log(
    output_path: Path,
    case_meta: "TestCaseMeta | None",
    result: EvalResult,
    error_detail: str = "",
    captured_log: str = "",
    case_index: int = 0,
    total_cases: int = 0,
) -> None:
    """Append per-case execution detail to run_details.log (incremental, written after each case).

    Format mirrors pytest -v output: status tag, inputs, metrics,
    captured logging output, and error traceback.
    """
    os.makedirs(output_path, exist_ok=True)
    log_path = output_path / "run_details.log"

    status_tag = "PASS" if result.status == "consistent" else "FAIL"

    # Header line: ---- [PASS/FAIL] case_id (N/M) ----
    progress = f" ({case_index}/{total_cases})" if total_cases > 0 else ""
    lines = [f"---- [{status_tag}] {result.case_id}{progress} ----"]

    # Input info from TestCaseMeta
    if case_meta:
        lines.append(f"  shape       : {case_meta.shape}")
        lines.append(f"  dtype       : {case_meta.dtype}")
        lines.append(f"  generator   : {case_meta.generator}")
        lines.append(f"  value_range : {case_meta.value_range}")
        lines.append(f"  seed        : {case_meta.seed}")
        lines.append(f"  api         : {case_meta.torch_api_name or '(none)'}")

    # Status (only for failures — PASS is obvious)
    if result.status != "consistent":
        lines.append(f"  status      : {result.status}")

    # Precision metrics
    if result.match_rate or result.max_diff:
        lines.append(
            f"  match_rate  : {result.match_rate}  "
            f"max_diff: {result.max_diff}  "
            f"mean_diff: {result.mean_diff}  "
            f"max_rel_diff: {result.max_relative_diff}"
        )

    # Timing
    if result.ref_time_ms or result.execute_time_ms:
        speedup_str = f"  speedup: {result.speedup}" if result.speedup else ""
        lines.append(
            f"  timing      : ref={result.ref_time_ms}ms ({result.ref_execute_device})"
            f"  exec={result.execute_time_ms}ms ({result.execute_device})"
            f"{speedup_str}  [{result.timing_mode}]"
        )

    # Captured logging output
    if captured_log and captured_log.strip():
        lines.append("  --- captured log ---")
        for line in captured_log.rstrip().split("\n"):
            lines.append(f"  {line}")

    # Error detail (exception traceback)
    if error_detail:
        lines.append("  --- error traceback ---")
        for line in error_detail.rstrip().split("\n"):
            lines.append(f"    {line}")

    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_run_log_header(
    output_path: Path,
    op_name: str,
    skill_name: str,
    total_cases: int = 0,
    device: str = "",
    api_name: str = "",
    env_info: dict | None = None,
) -> None:
    """Write session header to run_details.log (call once before any cases)."""
    os.makedirs(output_path, exist_ok=True)
    log_path = output_path / "run_details.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = "=" * 68
    lines = [banner]
    lines.append(f"Evaluation Run: {op_name}  |  {skill_name}  |  {timestamp}")
    detail_parts = [f"Op: {op_name}"]
    if api_name:
        detail_parts.append(f"API: {api_name}")
    if total_cases > 0:
        detail_parts.append(f"Cases: {total_cases}")
    if device:
        detail_parts.append(f"Device: {device}")
    lines.append("  |  ".join(detail_parts))
    if env_info:
        for k, v in env_info.items():
            lines.append(f"{k}: {v}")
    lines.append(banner)
    lines.append("")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_run_log_summary(
    output_path: Path,
    results: List[EvalResult],
    total_time_sec: float,
) -> None:
    """Append summary section to run_details.log (call once after all cases)."""
    log_path = output_path / "run_details.log"

    passed = [r for r in results if r.status == "consistent"]
    failed = [r for r in results if r.status != "consistent"]

    banner = "=" * 20 + " Summary " + "=" * 20
    lines = [banner]
    lines.append(
        f"{len(results)} cases in {total_time_sec:.2f}s "
        f"— {len(passed)} passed, {len(failed)} failed"
    )
    lines.append("")

    if passed:
        passed_ids = ", ".join(r.case_id for r in passed)
        lines.append(f"PASSED:  {passed_ids}")

    if failed:
        lines.append("FAILED:")
        for r in failed:
            lines.append(f"  {r.case_id:<40}  {r.status}")

    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_reference_source(source: str) -> dict:
    """Parse reference.py source to extract generator and value_range.

    Handles the constrained templates from reference-generation skill:
    - torch.rand(...)           -> uniform [0.0, 1.0)
    - torch.randn(...)          -> uniform (-inf, inf)  (normal distribution)
    - torch.randint(low, high, ...) -> uniform [low, high)
    - torch.rand(...).to(dtype) -> uniform [0.0, 1.0)

    Returns:
        dict with keys "generator" and "value_range"
    """
    # Extract get_inputs function body
    get_inputs_match = re.search(
        r'def get_inputs\(\):\s*\n((?:[ \t]+.*\n)*)', source
    )
    if not get_inputs_match:
        return {"generator": "", "value_range": ""}

    body = get_inputs_match.group(1)

    # torch.randint(low, high, ...)
    randint_match = re.search(r'torch\.randint\(\s*(-?\d+)\s*,\s*(-?\d+)', body)
    if randint_match:
        low, high = randint_match.group(1), randint_match.group(2)
        return {"generator": "uniform", "value_range": f"[{low}, {high})"}

    # torch.rand(...)
    if "torch.rand(" in body:
        return {"generator": "uniform", "value_range": "[0.0, 1.0)"}

    # torch.randn(...)
    if "torch.randn(" in body:
        return {"generator": "uniform", "value_range": "(-inf, inf)"}

    return {"generator": "", "value_range": ""}
