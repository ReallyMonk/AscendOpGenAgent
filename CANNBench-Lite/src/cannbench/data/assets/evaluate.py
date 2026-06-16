#!/usr/bin/env python3
"""
AscendC 自定义算子评估工具

用于评估自定义算子的正确性和性能表现。

命令行用法:
    python evaluate.py <op_name> [--output-path <path>] [--device <device>]

示例:
    python evaluate.py Add
    python evaluate.py MyCustomOperator --device npu:7

API 用法:
    from evaluate import AscendBackend

    # 初始化后端
    backend = AscendBackend(eval_code, ref_code)

    # 1. 精度评估
    success, message, metrics = backend.evaluate_correctness()
    # 返回: (bool, str, list[dict]) - 是否通过及详细信息和指标

    # 2. 性能测试 - 单个模型
    median_time = backend.measure_performance('ModelNew', num_warmup=10, num_perf_trials=100)
    # 返回: float - 中位数耗时(毫秒)

    # 3. 性能对比 - 两个模型
    ref_median, custom_median = backend.compare_performance(num_warmup=10, num_perf_trials=100)
    # 返回: (float, float) - 参考模型和自定义模型的中位数耗时(毫秒)

输入代码要求:
    评估代码必须包含以下组件:
    - Model: 参考实现类 (torch.nn.Module)
    - ModelNew: 自定义算子实现类 (torch.nn.Module)
    - get_inputs(): 返回测试输入数据的函数
    - get_init_inputs(): 返回模型初始化参数的函数

目录结构:
    <output_path>/
        <op_name>_reference.py  # 参考代码
        <op_name>_custom.py     # 自定义算子代码

环境要求:
    - vendors/customize/op_api/lib/ 库文件必须存在
    - 需要设置 ASCEND_CUSTOM_OPP_PATH 环境变量
"""

import os
import sys
import re
import logging
import argparse
import traceback
from pathlib import Path
from typing import Tuple

import torch
import torch_npu

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared.reporting import (
    TestCaseMeta, EvalResult, write_test_cases_csv,
    write_eval_report_csv, write_summary_md, parse_reference_source,
    append_case_log, write_run_log_header, write_run_log_summary,
    CaseLogCapture,
)
from shared.case_registry import CaseRecord, CaseRegistry


def resolve_work_dir(op_name: str, output_path: str | None = None) -> Path:
    """Resolve operator work directory.

    When output_path is not provided, keep backward-compatible layout:
    ./output/{op_name}
    """
    if output_path is not None:
        normalized = output_path.strip()
        if not normalized:
            raise ValueError("--output-path is provided but empty")
        return Path(normalized).expanduser().resolve()
    return Path("output").joinpath(op_name).resolve()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


class AscendBackend:
    def __init__(self, eval_src: str, ref_src: str, seed_num: int = 1024, num_correct_trials: int = 5, device: str | None = None):
        self.context = {}
        if device is not None:
            self.device = torch.device(device)
            torch_npu.npu.set_device(self.device)
        else:
            _device_id = int(os.environ.get('ASCEND_DEVICE_ID', '0'))
            self.device = torch.device(f'npu:{_device_id}')
            # Must explicitly set the device so CANN's ACL runtime initializes the correct
            # device context. Without this, getCurrentNPUStream() defaults to device 0, causing
            # 507057 (DDR address out of range) when tensors are on a different device.
            torch_npu.npu.set_device(_device_id)
        self.seed_num = seed_num
        self.num_correct_trials = num_correct_trials
        self.last_traceback: str = ""
        self._set_context(eval_src, ref_src)

    def _set_context(self, eval_src: str, ref_src: str):
        try:
            exec(eval_src, self.context)
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")

    def _synchronize(self):
        """Synchronize NPU operations."""
        torch_npu.npu.synchronize(self.device)

    def _move_to_device(self, data):
        """Move tensor data to NPU device."""
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(x) for x in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def _prepare_inputs(self):
        """Get inputs from context and move to device."""
        get_inputs = self.context["get_inputs"]
        inputs = get_inputs()
        return self._move_to_device(inputs)

    def _prepare_init_inputs(self):
        """Get init_inputs from context and move to device."""
        get_init_inputs = self.context["get_init_inputs"]
        init_inputs = get_init_inputs()
        return self._move_to_device(init_inputs)

    def _create_model(self, model_name='ModelNew', init_inputs=None):
        """Create model instance and move to device.

        Args:
            model_name: Name of the model class in context
            init_inputs: Optional pre-prepared init inputs. If None, will call _prepare_init_inputs()
        """
        ModelClass = self.context[model_name]
        if init_inputs is None:
            init_inputs = self._prepare_init_inputs()

        model = ModelClass(*init_inputs).to(self.device)
        self._synchronize()
        return model

    def _normalize_output(self, output, index):
        """Extract single output tensor from list/tuple or return as-is."""
        if isinstance(output, (list, tuple)):
            return output[index]
        return output

    def _check_shape(self, ref_output, new_output, output_idx):
        """Check if output shapes match. Returns error message or None."""
        if ref_output.shape != new_output.shape:
            return f"[FAIL] Output shape mismatch at output {output_idx}: Expected {ref_output.shape}, got {new_output.shape}"
        return None

    def _check_values(self, ref_output, new_output, output_idx, atol=1e-02, rtol=1e-02):
        """Check if output values are close. Returns (error_msg, pass_info, metrics) tuple."""
        if ref_output.dtype in (torch.float16, torch.bfloat16):
            ref_output = ref_output.float()
            new_output = new_output.float()
        close_mask = torch.isclose(ref_output, new_output, atol=atol, rtol=rtol)
        total = close_mask.numel()
        matched = close_mask.sum().item()
        match_rate = matched / total

        diff = (ref_output - new_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.float().mean().item()
        ref_abs = ref_output.abs().clamp(min=1e-12)
        max_relative_diff = (diff / ref_abs).max().item()

        metrics = {
            "match_rate": f"{match_rate * 100:.2f}%",
            "max_diff": f"{max_diff:.5e}",
            "mean_diff": f"{mean_diff:.5e}",
            "max_relative_diff": f"{max_relative_diff:.5e}",
        }

        if torch.allclose(ref_output, new_output, atol=atol, rtol=rtol):
            pass_info = (
                f"Output {output_idx}: shape={list(ref_output.shape)}, "
                f"match_rate=100.00% ({matched}/{total}), "
                f"max_diff={max_diff:.5e}, mean_diff={mean_diff:.5e}"
            )
            return None, pass_info, metrics

        mismatch_idx = (~close_mask).nonzero(as_tuple=False)[0]
        mismatch_idx_tuple = tuple(mismatch_idx.tolist())
        ref_val = ref_output[mismatch_idx_tuple].item()
        new_val = new_output[mismatch_idx_tuple].item()

        error_msg = (
            f"[FAIL] Output {output_idx} mismatch\n"
            f"Match rate: {match_rate * 100:.2f}% ({matched}/{total})\n"
            f"Example mismatch at index {tuple(mismatch_idx.tolist())}: "
            f"ref={ref_val}, new={new_val}"
        )
        return error_msg, None, metrics

    def evaluate_correctness(self):
        """Execute correctness check between reference and custom models."""
        try:
            set_seed(self.seed_num)
            inputs = self._prepare_inputs()
            init_inputs = self._prepare_init_inputs()
            ref_model = self._create_model('Model', init_inputs)
            new_model = self._create_model('ModelNew', init_inputs)
            with torch.no_grad():
                ref_output = ref_model(*inputs)
                new_output = new_model(*inputs)
            self._synchronize()

            has_error, message, metrics = self._compare_outputs(ref_output, new_output)

            if has_error:
                return False, message, metrics
            return True, message, metrics

        except Exception as e:
            logging.error("[FAIL] runtime error when evaluating correctness")
            self.last_traceback = traceback.format_exc()
            return False, f"[OP_FAILED] {str(e)}", []

    def _compare_outputs(self, ref_output, new_output):
        """Compare model outputs and return (has_error, message, metrics_list)."""
        error_parts = []
        pass_parts = []
        all_metrics = []
        num_outputs = len(ref_output) if isinstance(ref_output, (list, tuple)) else 1

        for i in range(num_outputs):
            ref_out = self._normalize_output(ref_output, i).to("cpu")
            new_out = self._normalize_output(new_output, i).to("cpu")

            error = self._check_shape(ref_out, new_out, i)
            if error:
                error_parts.append(error)
                continue

            error, pass_info, metrics = self._check_values(ref_out, new_out, i)
            all_metrics.append(metrics)
            if error:
                error_parts.append(error)
            else:
                pass_parts.append(pass_info)

        if error_parts:
            return True, "\n".join(error_parts), all_metrics
        return False, "[PASS]\n" + "\n".join(pass_parts), all_metrics

    def measure_performance(self, model_name='ModelNew', num_warmup=10, num_perf_trials=100):
        """Measure performance for specified model.

        Args:
            model_name: Model name to measure ('Model' or 'ModelNew')
            num_warmup: Number of warmup iterations
            num_perf_trials: Number of performance measurement iterations

        Returns:
            Median elapsed time in milliseconds
        """
        import statistics

        event_class = torch_npu.npu.Event
        elapsed_times = []

        model = self._create_model(model_name)
        inputs = self._prepare_inputs()

        with torch.no_grad():
            def _run_performance_test(kernel_fn, times_list):
                for _ in range(num_warmup):
                    kernel_fn(*inputs)
                    self._synchronize()
                for _ in range(num_perf_trials):
                    start_event = event_class(enable_timing=True)
                    end_event = event_class(enable_timing=True)
                    start_event.record()
                    kernel_fn(*inputs)
                    end_event.record()
                    self._synchronize()
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    times_list.append(elapsed_time_ms)

            _run_performance_test(model, elapsed_times)

        return statistics.median(elapsed_times)

    def compare_performance(self, num_warmup=10, num_perf_trials=100):
        """Compare performance between reference and custom models.

        Args:
            num_warmup: Number of warmup iterations
            num_perf_trials: Number of performance measurement iterations

        Returns:
            Tuple of (ref_median_time, custom_median_time) in milliseconds
        """
        ref_median = self.measure_performance('Model', num_warmup, num_perf_trials)
        custom_median = self.measure_performance('ModelNew', num_warmup, num_perf_trials)

        return ref_median, custom_median

    def cleanup(self):
        del self.context
        torch_npu.npu.empty_cache()
        self._synchronize()

def setup_ascend_runtime_environment(project_root: Path) -> None:
    """
    设置 AscendC 自定义算子所需的运行时环境变量。

    Args:
        project_root (Path): 项目根目录路径
    Sets:
        - ASCEND_CUSTOM_OPP_PATH
        - LD_LIBRARY_PATH (追加自定义库路径)
    """
    project_root = Path(project_root).resolve()

    # 1. 设置 ASCEND_CUSTOM_OPP_PATH
    custom_opp_path = project_root.joinpath("vendors/customize")

    if not custom_opp_path.exists():
        raise FileNotFoundError(f"ASCEND custom OPP directory not found: {custom_opp_path}")

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
    logging.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp_path}")

    # 2. 更新 LD_LIBRARY_PATH
    custom_lib_path = Path(custom_opp_path).joinpath("op_api/lib").resolve()

    if not custom_lib_path.exists():
        raise FileNotFoundError(f"ASCEND custom OP API library directory not found: {custom_lib_path}")

    custom_lib_path_str = str(custom_lib_path)

    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    # 避免重复添加（防止路径膨胀）
    if custom_lib_path_str not in existing_ld_path:
        new_ld_path = f"{custom_lib_path_str}:{existing_ld_path}".rstrip(":")
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        logging.info(f"Updated LD_LIBRARY_PATH to include: {custom_lib_path_str}")
    else:
        logging.info(f"LD_LIBRARY_PATH already contains: {custom_lib_path_str}")

def evaluate_operator(
    eval_src_path: Path,
    ref_src_path: Path,
    project_root_path: Path,
    op_name: str | None = None,
    device: str | None = None,
) -> Tuple[bool, str]:
    """
    评估算子代码正确性，并将报告持久化到 reports/ascendc-evaluation/。

    Args:
        eval_src_path (Path): 要评估的代码文件路径（包含自定义算子实现的 ModelNew 类）
        ref_src_path (Path): 参考代码文件路径（包含参考实现的 Model 类）
        project_root_path (Path): 项目根目录路径，用于设置运行时环境
        op_name (str): 算子名称（可选，默认从文件名推断）
        device (str | None): NPU 设备（可选，默认从 ASCEND_DEVICE_ID 环境变量推断）

    Returns:
        Tuple of (success, output_message)
    """
    setup_ascend_runtime_environment(project_root_path)

    if not eval_src_path.exists():
        raise FileNotFoundError(f'Evaluation code file not found: {eval_src_path}')
    if not ref_src_path.exists():
        raise FileNotFoundError(f'Reference code file not found: {ref_src_path}')

    eval_code = eval_src_path.read_text(encoding='utf-8')
    ref_code = ref_src_path.read_text(encoding='utf-8')

    if op_name is None:
        op_name = eval_src_path.stem.replace("_custom", "")

    ascend_backend = AscendBackend(eval_code, ref_code, device=device)

    # Extract test case metadata from inputs
    set_seed(ascend_backend.seed_num)
    raw_inputs = ascend_backend.context["get_inputs"]()
    shapes = []
    dtypes = []
    for inp in raw_inputs:
        if isinstance(inp, torch.Tensor):
            shapes.append(str(list(inp.shape)))
            dtypes.append(str(inp.dtype))
    ref_info = parse_reference_source(ref_code)

    case_meta = TestCaseMeta(
        case_id="eval_0",
        op_name=op_name,
        torch_api_name="",
        shape=";".join(shapes),
        dtype=";".join(dtypes),
        value_range=ref_info["value_range"],
        generator=ref_info["generator"],
        seed=str(ascend_backend.seed_num),
        case_source="op_desc",
    )

    # Run correctness evaluation (capture logging output during execution)
    run_start = __import__("time").time()
    with CaseLogCapture() as capture:
        flag_bool, run_info, metrics = ascend_backend.evaluate_correctness()
        logging.info(f"Evaluation correctness: {run_info}")

        # Build EvalResult
        combined_metrics = metrics[0] if metrics else {}
        ref_time_ms = ""
        execute_time_ms = ""
        speedup_str = ""

        if flag_bool:
            status = "consistent"
            ref_median, custom_median = ascend_backend.compare_performance()
            speedup = ref_median / custom_median if custom_median > 0 else float('inf')
            ref_time_ms = f"{ref_median:.3f}"
            execute_time_ms = f"{custom_median:.3f}"
            speedup_str = f"{speedup:.2f}x"
            logging.info(f"Evaluation performance: ref={ref_median:.3f}ms, custom={custom_median:.3f}ms, speedup={speedup:.2f}x")
        elif run_info.startswith("[OP_FAILED]"):
            status = "op_failed"
        else:
            status = "inconsistent"

    eval_result = EvalResult(
        case_id="eval_0",
        status=status,
        match_rate=combined_metrics.get("match_rate", ""),
        max_diff=combined_metrics.get("max_diff", ""),
        mean_diff=combined_metrics.get("mean_diff", ""),
        max_relative_diff=combined_metrics.get("max_relative_diff", ""),
        ref_time_ms=ref_time_ms,
        execute_time_ms=execute_time_ms,
        speedup=speedup_str,
        timing_mode="benchmark_median",
        ref_execute_device=str(ascend_backend.device),
        execute_device=str(ascend_backend.device),
    )

    # Persist reports
    report_dir = project_root_path / "evaluation_results" / "ascendc-evaluation"
    write_run_log_header(
        report_dir, op_name, "ascendc-evaluation",
        total_cases=1, device=str(ascend_backend.device),
        env_info={"ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")},
    )
    append_case_log(report_dir, case_meta, eval_result,
                    error_detail=ascend_backend.last_traceback,
                    captured_log=capture.output,
                    case_index=1, total_cases=1)
    write_test_cases_csv(report_dir, [case_meta])
    write_eval_report_csv(report_dir, [eval_result])
    write_summary_md(report_dir, [case_meta], [eval_result], skill_name="ascendc-evaluation")
    write_run_log_summary(report_dir, [eval_result],
                          total_time_sec=__import__("time").time() - run_start)
    registry = CaseRegistry(
        "ascendc-evaluation", op_name,
        {"base_seed": ascend_backend.seed_num, "device": str(ascend_backend.device)},
    )
    registry.add(CaseRecord(
        case_id="eval_0",
        seed=ascend_backend.seed_num,
        generator=ref_info.get("generator", ""),
        status=status,
        spec={
            "reference_file": ref_src_path.name,
            "custom_file": eval_src_path.name,
        },
    ))
    registry.save(report_dir)
    logging.info(f"Reports written to {report_dir}")

    return flag_bool, run_info

def replay_evaluation(
    registry_path: Path,
    work_dir: Path,
    op_name: str,
    device: str | None = None,
    filter_status: str | None = None,
    filter_case_id: str | None = None,
):
    """Replay evaluation from a cases.json registry."""
    from shared.case_registry import CaseRegistry
    import time

    registry = CaseRegistry.load(registry_path)
    records = registry.filter(
        status=filter_status.split(",") if filter_status else None,
        case_ids=[filter_case_id] if filter_case_id else None,
    )
    if not records:
        logging.warning("No cases matched the replay filter.")
        return

    logging.info(f"Replaying {len(records)} case(s) from {registry_path}")

    for rec in records:
        ref_file = rec.spec.get("reference_file", f"{op_name}_reference.py")
        custom_file = rec.spec.get("custom_file", f"{op_name}_custom.py")
        eval_src = work_dir / custom_file
        ref_src = work_dir / ref_file

        if not eval_src.exists() or not ref_src.exists():
            logging.error(f"Source files not found: {eval_src}, {ref_src}")
            continue

        success, info = evaluate_operator(
            eval_src, ref_src, work_dir, op_name=op_name, device=device
        )
        new_status = "consistent" if success else (
            "op_failed" if "[OP_FAILED]" in info else "inconsistent"
        )
        registry.update_status(rec.case_id, new_status)

    registry.save(registry_path.parent)
    logging.info("Replay complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Evaluate AscendC custom operator correctness")
    parser.add_argument("op_name", type=str, help="Operator name")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="operator output directory (default: ./output/{op_name})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="NPU device to run evaluation on (default: from ASCEND_DEVICE_ID or npu:0)",
    )
    parser.add_argument("--replay", type=str, default=None,
                        help="Path to cases.json for replay mode")
    parser.add_argument("--filter-status", type=str, default=None,
                        help="Comma-separated statuses to replay")
    parser.add_argument("--case-id", type=str, default=None,
                        help="Replay a single case by ID")

    args = parser.parse_args()

    if args.replay:
        try:
            work_dir = resolve_work_dir(args.op_name, args.output_path)
            replay_evaluation(
                Path(args.replay), work_dir, args.op_name,
                device=args.device,
                filter_status=args.filter_status,
                filter_case_id=args.case_id,
            )
        except Exception as e:
            logging.error(f"Replay error: {e}")
        sys.exit(0)

    try:
        work_dir = resolve_work_dir(args.op_name, args.output_path)
        eval_src = work_dir.joinpath(f"{args.op_name}_custom.py")
        ref_src = work_dir.joinpath(f"{args.op_name}_reference.py")
        evaluate_operator(eval_src, ref_src, work_dir, op_name=args.op_name, device=args.device)
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
