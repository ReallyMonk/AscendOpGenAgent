#!/usr/bin/env python3
"""
Profiling Runner for AscendC Operators

Executes hardware profiling using acl.prof + msprof to collect op_summary metrics.
Core logic extracted from KernelStorm's AscendPerformanceTest.py, adapted for
the OpenOps remote worker pipeline.

Usage:
    from profiling_runner import run_profiling, parse_op_summary

    result = run_profiling(op_name, device_id=0, output_dir="/tmp/prof", num_trials=10)
"""

import os
import sys
import csv
import glob
import shutil
import logging
import datetime
import subprocess
import statistics
from io import StringIO
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ProfilingContext:
    """ACL profiling context manager.

    Wraps acl.prof init/start/stop/finalize lifecycle with AICORE metrics collection.
    Reference: KernelStorm AscendPerformanceTest.py:32-77
    """

    def __init__(self, output_path: str, device_id: int):
        self.output_path = output_path
        self.device_id = device_id
        self.prof_config = None
        self.step_info = None

    def __enter__(self):
        import acl

        ret = acl.prof.init(self.output_path)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize profiling (ret={ret})")

        ACL_PROF_ACL_API = 0x0001
        ACL_PROF_TASK_TIME = 0x0002
        ACL_PROF_AICORE_METRICS = 0x0004

        self.prof_config = acl.prof.create_config(
            [self.device_id],
            1,
            0,
            ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS,
        )

        ret = acl.prof.start(self.prof_config)
        if ret != 0:
            raise RuntimeError(f"Failed to start profiling (ret={ret})")

        self.step_info = acl.prof.create_step_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import acl

        if self.step_info:
            acl.prof.destroy_step_info(self.step_info)
        if self.prof_config:
            acl.prof.stop(self.prof_config)
            acl.prof.destroy_config(self.prof_config)
        acl.prof.finalize()


def msprof_export(path_prefix: str, path_suffix: str) -> None:
    """Run msprof --export=on to generate op_summary CSV.

    Reference: KernelStorm AscendPerformanceTest.py:80-105
    """
    command = f"msprof --export=on --output={path_suffix}"
    try:
        subprocess.run(command, shell=True, check=True, cwd=path_prefix)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"msprof export failed (exit code {e.returncode})")
    except FileNotFoundError:
        raise RuntimeError(f"msprof not found or directory '{path_prefix}' does not exist")


def locate_op_summary_file(output_dir: str) -> str:
    """Find the single op_summary_*.csv file in output directory.

    Reference: KernelStorm AscendPerformanceTest.py:108-121
    """
    op_summary_files = glob.glob(
        os.path.join(output_dir, "**", "op_summary_*.csv"), recursive=True
    )

    if not op_summary_files:
        raise FileNotFoundError(
            f"No op_summary_*.csv found in {output_dir}"
        )
    if len(op_summary_files) > 1:
        logger.warning(
            f"Found multiple op_summary files, using first: {op_summary_files}"
        )

    return os.path.abspath(op_summary_files[0])


def parse_op_summary(csv_path: str, task_type: str = "vector") -> Dict[str, Any]:
    """Parse op_summary CSV and extract structured profiling metrics.

    Reads the op_summary CSV exported by msprof, extracts metrics based on
    task_type (vector/cube/cv-mix), and returns a structured dict.

    Args:
        csv_path: Path to op_summary_*.csv file
        task_type: One of "vector", "cube", "cv-mix"

    Returns:
        Dict with profiling metrics including task_duration_us, aiv_vec_ratio, etc.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Extract operator rows using ReduceMax/MatMulV3 sentinel pattern
    # Reference: KernelStorm AscendPerformanceTest.py:233-286
    op_types = df["OP Type"].values

    start_idxs = [i for i, op in enumerate(op_types) if op == "ReduceMax"]
    end_idxs = [i for i, op in enumerate(op_types) if op == "MatMulV3"]

    # Pair ReduceMax→MatMulV3 intervals
    pattern_intervals = []
    for start in start_idxs:
        ends_after = [e for e in end_idxs if e > start]
        if ends_after:
            pattern_intervals.append((start, ends_after[0]))

    # Extract pattern durations (excluding first warmup pattern)
    pattern_durations = []
    pattern_rows = []
    for start, end in pattern_intervals:
        if end - start > 1:
            pattern_slice = df.iloc[start + 1 : end]
            pattern_durations.append(pattern_slice["Task Duration(us)"].sum())
            pattern_rows.append(pattern_slice)

    if len(pattern_durations) <= 1:
        # Fallback: use all rows if sentinel pattern not found
        logger.warning("Sentinel pattern not found, using all rows for profiling")
        return _parse_all_rows(df, task_type)

    # Use median pattern (skip first)
    durations = pattern_durations[1:]
    rows = pattern_rows[1:]
    sorted_idx = sorted(range(len(durations)), key=lambda i: durations[i])
    median_idx = len(durations) // 2
    median_duration = durations[sorted_idx[median_idx]]
    median_rows = rows[sorted_idx[median_idx]]

    return _extract_metrics(median_rows, median_duration, task_type)


def _parse_all_rows(df, task_type: str) -> Dict[str, Any]:
    """Fallback parser when sentinel pattern is not found."""
    if df.empty:
        return {"task_duration_us": 0.0, "raw_rows": []}

    total_duration = df["Task Duration(us)"].sum() if "Task Duration(us)" in df.columns else 0.0
    return _extract_metrics(df, total_duration, task_type)


def _extract_metrics(df, total_duration: float, task_type: str) -> Dict[str, Any]:
    """Extract structured metrics from a DataFrame of profiling rows.

    Reference: KernelStorm AscendPerformanceTest.py:123-228 (column mappings)
    """
    import pandas as pd

    result = {
        "task_duration_us": round(total_duration, 3),
        "raw_rows": [],
    }

    # Column mappings by task type
    vector_cols = {
        "aiv_time_us": "aiv_time(us)",
        "aiv_vec_ratio": "aiv_vec_ratio",
        "aiv_scalar_ratio": "aiv_scalar_ratio",
        "aiv_mte2_ratio": "aiv_mte2_ratio",
        "aiv_mte3_ratio": "aiv_mte3_ratio",
        "aiv_icache_miss_rate": "aiv_icache_miss_rate",
    }

    cube_cols = {
        "aicore_time_us": "aicore_time(us)",
        "aic_mac_ratio": "aic_mac_ratio",
        "aic_scalar_ratio": "aic_scalar_ratio",
        "aic_mte1_ratio": "aic_mte1_ratio",
        "aic_mte2_ratio": "aic_mte2_ratio",
        "aic_fixpipe_ratio": "aic_fixpipe_ratio",
        "aic_icache_miss_rate": "aic_icache_miss_rate",
        "cube_utilization": "cube_utilization(%)",
    }

    t = task_type.lower()
    if t == "vector":
        col_map = vector_cols
    elif t == "cube":
        col_map = cube_cols
    elif t in ("cv-mix", "unknown"):
        col_map = {**vector_cols, **cube_cols}
    else:
        col_map = vector_cols

    # Aggregate metrics across rows (weighted average by task duration)
    for metric_key, csv_col in col_map.items():
        if csv_col in df.columns:
            values = pd.to_numeric(df[csv_col], errors="coerce").dropna()
            if not values.empty:
                if "ratio" in metric_key or "rate" in metric_key or "utilization" in metric_key:
                    # For ratios: weighted average by task duration
                    if "Task Duration(us)" in df.columns:
                        durations = pd.to_numeric(
                            df.loc[values.index, "Task Duration(us)"], errors="coerce"
                        ).fillna(0)
                        total_dur = durations.sum()
                        if total_dur > 0:
                            result[metric_key] = round(
                                (values * durations).sum() / total_dur, 4
                            )
                        else:
                            result[metric_key] = round(values.mean(), 4)
                    else:
                        result[metric_key] = round(values.mean(), 4)
                else:
                    # For absolute values: sum
                    result[metric_key] = round(values.sum(), 3)
            else:
                result[metric_key] = 0.0
        else:
            result[metric_key] = 0.0

    # Store raw rows for detailed analysis
    for _, row in df.iterrows():
        result["raw_rows"].append(row.to_dict())

    return result


def run_profiling(
    op_name: str,
    device_id: int,
    output_dir: str,
    eval_script_dir: str,
    num_trials: int = 10,
    task_type: str = "vector",
) -> Dict[str, Any]:
    """Execute profiling for an operator and return structured metrics.

    This is the main entry point. It:
    1. Loads Model/ModelNew from the evaluation code
    2. Runs warmup + profiled execution using acl.prof
    3. Exports op_summary via msprof
    4. Parses and returns structured metrics

    Args:
        op_name: Operator name (e.g., "mse_loss")
        device_id: NPU device ID
        output_dir: Base output directory (contains output/{op_name}/)
        eval_script_dir: Directory containing evaluate.py and operator files
        num_trials: Number of profiling trials
        task_type: Operator type ("vector", "cube", "cv-mix")

    Returns:
        Dict with keys: timing, profiling, profile_log
    """
    import torch
    import torch_npu

    logger.info(f"Starting profiling for {op_name} on device {device_id}")

    # Setup paths
    work_dir = Path(eval_script_dir) / "output" / op_name
    eval_src_path = work_dir / f"{op_name}_custom.py"
    ref_src_path = work_dir / f"{op_name}_reference.py"

    if not eval_src_path.exists():
        raise FileNotFoundError(f"Custom operator file not found: {eval_src_path}")
    if not ref_src_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_src_path}")

    # Setup runtime environment — LD_LIBRARY_PATH first so custom_ops_lib can import;
    # ASCEND_CUSTOM_OPP_PATH is set AFTER ref model timing to avoid interfering with
    # standard CANN op dispatch during reference model measurement.
    custom_opp_path = work_dir / "vendors" / "customize"
    lib_path = custom_opp_path / "op_api" / "lib" if custom_opp_path.exists() else None
    if lib_path and lib_path.exists():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        lib_str = str(lib_path)
        if lib_str not in existing:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_str}:{existing}".rstrip(":")

    # Load operator code (read now, exec selectively below)
    eval_code = eval_src_path.read_text(encoding="utf-8")
    ref_code = ref_src_path.read_text(encoding="utf-8")

    # exec reference code first — it only defines Model and get_inputs/get_init_inputs.
    # Do NOT exec eval_code yet: custom.py does `import custom_ops_lib` at module level,
    # which triggers CANN to load libopapi.so and read ASCEND_CUSTOM_OPP_PATH at import
    # time.  We must defer that import until after ref model timing.
    ref_context = {}
    exec(ref_code, ref_context)

    # Setup device
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device_id)

    # Prepare inputs using the reference context
    get_inputs = ref_context["get_inputs"]
    get_init_inputs = ref_context["get_init_inputs"]

    def move_to_device(data):
        if isinstance(data, list):
            return [move_to_device(x) for x in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data

    init_inputs = move_to_device(get_init_inputs())
    inputs = move_to_device(get_inputs())

    ModelClass = ref_context["Model"]
    ref_model = ModelClass(*init_inputs).to(device)

    # --- Timing (non-profiling, using NPU events) ---
    def measure_time(model, num_warmup=10, num_perf=100):
        elapsed_times = []
        with torch.no_grad():
            for _ in range(num_warmup):
                model(*inputs)
                torch.npu.synchronize()
            for _ in range(num_perf):
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                start_event.record()
                model(*inputs)
                end_event.record()
                torch.npu.synchronize()
                elapsed_times.append(start_event.elapsed_time(end_event))
        return statistics.median(elapsed_times)

    # Time reference model BEFORE loading custom op to avoid CANN dispatch interference.
    ref_median_ms = measure_time(ref_model)

    # Now set ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH, THEN exec custom.py so that
    # `import custom_ops_lib` and CANN's libopapi.so loading see the correct paths.
    if custom_opp_path.exists():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)

    eval_context = {}
    exec(eval_code, eval_context)
    ModelNewClass = eval_context["ModelNew"]

    new_model = ModelNewClass(*init_inputs).to(device)
    custom_median_ms = measure_time(new_model)
    speedup = ref_median_ms / custom_median_ms if custom_median_ms > 0 else float("inf")

    logger.info(
        f"Timing: ref={ref_median_ms:.3f}ms, custom={custom_median_ms:.3f}ms, speedup={speedup:.2f}x"
    )

    # --- Profiling (using acl.prof) ---
    prof_base_dir = Path(output_dir) / "profiling_data"
    prof_base_dir.mkdir(parents=True, exist_ok=True)

    # Clean old cache
    atc_data = Path("/root/atc_data")
    if atc_data.exists():
        shutil.rmtree(atc_data, ignore_errors=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_suffix = f"device{device_id}_{timestamp}"
    prof_output_path = str(prof_base_dir / path_suffix)

    # Sentinel tensors for warmup/cache clearing (same as KernelStorm)
    mm1 = torch.rand((10240, 10240), dtype=torch.float16).npu()
    mm2 = torch.rand((10240, 10240), dtype=torch.float16).npu()
    reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).npu()

    profile_log = []

    try:
        with ProfilingContext(prof_output_path, device_id):
            for i in range(num_trials):
                # Cache clearing + frequency boost via sentinel ops
                _ = torch.matmul(mm1, mm2)
                torch.npu.synchronize()
                _ = torch.max(reduce_input)
                torch.npu.synchronize()

                # Profile the custom operator
                with torch.no_grad():
                    _ = new_model(*inputs)
                torch.npu.synchronize()

        profile_log.append("Profiling capture completed successfully")

        # Export profiling data
        msprof_export(str(prof_base_dir), path_suffix)
        profile_log.append("msprof export completed")

        # Parse op_summary
        op_summary_path = locate_op_summary_file(prof_output_path)
        profiling_metrics = parse_op_summary(op_summary_path, task_type)
        profiling_metrics["op_name"] = op_name
        profile_log.append(f"Parsed op_summary: {op_summary_path}")

    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        profile_log.append(f"Profiling failed: {e}")
        profiling_metrics = {"op_name": op_name, "error": str(e)}

    # Cleanup sentinel tensors
    del mm1, mm2, reduce_input

    return {
        "timing": {
            "ref_median_ms": round(ref_median_ms, 4),
            "custom_median_ms": round(custom_median_ms, 4),
            "speedup": round(speedup, 4),
        },
        "profiling": profiling_metrics,
        "profile_log": "\n".join(profile_log),
    }


if __name__ == "__main__":
    """CLI entry point for subprocess execution.

    Usage:
        python3 profiling_runner.py <op_name> <device_id> <output_dir> <eval_script_dir> \
            [--num-trials N] [--task-type TYPE] [--result-file PATH]

    Writes JSON result to --result-file (default: <output_dir>/profiling_result.json).
    """
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="AscendC Operator Profiling Runner")
    parser.add_argument("op_name", help="Operator name")
    parser.add_argument("device_id", type=int, help="NPU device ID")
    parser.add_argument("output_dir", help="Output directory for profiling data")
    parser.add_argument("eval_script_dir", help="Directory containing output/{op_name}/ with operator files")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of profiling trials")
    parser.add_argument("--task-type", default="vector", help="Task type: vector, cube, cv-mix")
    parser.add_argument("--result-file", default=None, help="Path to write JSON result")

    args = parser.parse_args()

    result_file = args.result_file or os.path.join(args.output_dir, "profiling_result.json")

    try:
        result = run_profiling(
            op_name=args.op_name,
            device_id=args.device_id,
            output_dir=args.output_dir,
            eval_script_dir=args.eval_script_dir,
            num_trials=args.num_trials,
            task_type=args.task_type,
        )
        result["success"] = True
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        result = {"success": False, "error": str(e)}

    # Sanitize float values (NaN/Inf are not JSON-compliant)
    def sanitize(obj):
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            if obj == float("inf") or obj == float("-inf"):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    result = sanitize(result)

    # Write result JSON
    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Result written to: {result_file}")
    sys.exit(0 if result.get("success") else 1)
