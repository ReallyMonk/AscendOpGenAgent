#!/usr/bin/env python3
"""Generic torch/torch_npu baseline benchmark runner on remote server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torch_npu

from profiling_runner import ProfilingContext, locate_op_summary_file, msprof_export, parse_op_summary


def _resolve_callable(api_path: str):
    if api_path.startswith("benchmark_composite."):
        attr_name = api_path.split(".", 1)[1]
        return getattr(_BenchmarkCompositeOps, attr_name)
    module_path, attr_name = api_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[attr_name])
    return getattr(module, attr_name)


class _BenchmarkCompositeOps:
    @staticmethod
    def gelu_mul(x, dim: int = -1):
        x1, x2 = torch.chunk(x, 2, dim=dim)
        return F.gelu(x1) * x2


def _device_from_id(device_id: int) -> torch.device:
    torch_npu.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def _materialize_tensor(spec: Dict[str, Any], device: torch.device):
    dtype = getattr(torch, spec["dtype"])
    shape = spec["shape"]
    distribution = spec.get("distribution", "normal")
    if distribution == "ones":
        return torch.ones(shape, dtype=dtype, device=device)
    if distribution == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if distribution == "bool_bernoulli":
        return (torch.rand(shape, device=device) > float(spec.get("threshold", 0.5))).to(dtype)
    if distribution == "randint":
        low = int(spec.get("low", 0))
        high = int(spec.get("high", 1024))
        return torch.randint(low, high, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def _prepare_call(case: Dict[str, Any], tensors: Dict[str, Any]):
    builder = case["input_builder"]
    attrs = dict(case.get("attrs", {}))
    params = case.get("shape_params", {})

    if builder == "layer_norm_2d":
        h = params["H"]
        return [tensors["x"], [h], tensors["weight"], tensors["bias"]], attrs
    if builder == "unary_2d":
        return [tensors["x"]], attrs
    if builder == "rms_norm_2d":
        eps = attrs.pop("eps", 1e-5)
        return [tensors["x"], tensors["weight"], eps], attrs
    if builder == "matmul_2d":
        return [tensors["a"], tensors["b"]], attrs
    if builder == "bmm_3d":
        return [tensors["a"], tensors["b"]], attrs
    if builder == "softmax_4d":
        return [tensors["x"]], attrs
    if builder == "add_layer_norm_2d":
        eps = attrs.pop("eps", 1e-5)
        return [tensors["x1"], tensors["x2"], tensors["gamma"], tensors["beta"], eps], attrs
    if builder == "add_rms_norm_2d":
        eps = attrs.pop("eps", 1e-5)
        return [tensors["x1"], tensors["x2"], tensors["gamma"], eps], attrs
    if builder == "swiglu_2d":
        return [tensors["x"]], attrs
    if builder == "gated_2d":
        return [tensors["x"]], attrs
    if builder == "avg_pool2d_4d":
        kernel = params["kernel"]
        stride = params["stride"]
        padding = attrs.get("padding", 0)
        return [tensors["x"]], {"kernel_size": kernel, "stride": stride, "padding": padding}
    if builder == "conv2d_nchw":
        stride = attrs.get("stride", params["stride"])
        padding = attrs.get("padding", params["padding"])
        return [tensors["x"], tensors["weight"]], {"stride": stride, "padding": padding}
    if builder == "gather_rows_2d":
        dim = attrs.get("dim", 0)
        return [tensors["x"], dim, tensors["index"]], {}
    if builder == "embedding_bag_2d":
        return [tensors["indices"], tensors["weight"], tensors["offsets"]], attrs
    if builder == "cross_entropy_2d":
        return [tensors["logits"], tensors["target"]], attrs
    if builder == "masked_softmax_with_rel_pos_bias":
        scale_value = attrs.get("scale_value", 1.0)
        inner_precise = attrs.get("inner_precise", 0)
        return [tensors["x"], tensors["atten_mask"], tensors["relative_pos_bias"], scale_value, inner_precise], {}
    if builder == "scaled_masked_softmax":
        scale = attrs.get("scale", 0.125)
        fixed_triu_mask = attrs.get("fixed_triu_mask", False)
        return [tensors["x"], tensors["mask"], scale, fixed_triu_mask], {}
    if builder == "add_rms_norm_quant_2d":
        axis = attrs.get("axis", -1)
        epsilon = attrs.get("eps", 1e-5)
        div_mode = attrs.get("div_mode", True)
        return [
            tensors["x1"],
            tensors["x2"],
            tensors["gamma"],
            tensors["scales1"],
            None,
            None,
            None,
        ], {"axis": axis, "epsilon": epsilon, "div_mode": div_mode}
    if builder == "add_rms_norm_dynamic_quant_2d":
        eps = attrs.pop("eps", 1e-5)
        return [tensors["x1"], tensors["x2"], tensors["gamma"], eps], attrs
    if builder == "fusion_attention_bsnd":
        scale = attrs.get("scale", 1.0)
        keep_prob = attrs.get("keep_prob", 1.0)
        pre_tockens = attrs.get("pre_tockens", 2147483647)
        next_tockens = attrs.get("next_tockens", 2147483647)
        input_layout = attrs.get("input_layout", "BSND")
        return [
            tensors["query"],
            tensors["key"],
            tensors["value"],
            tensors["head_num"],
            input_layout,
            tensors.get("pse"),
            tensors.get("padding_mask"),
            tensors.get("atten_mask"),
            tensors.get("prefix"),
            tensors.get("actual_seq_qlen"),
            tensors.get("actual_seq_kvlen"),
            keep_prob,
            scale,
            pre_tockens,
            next_tockens,
            tensors.get("sparse_mode", 0),
        ], {}
    raise ValueError(f"unsupported input_builder: {builder}")


def _sync():
    torch_npu.npu.synchronize()


def _percentile(values: List[float], pct: int) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def _run_e2e(fn, args, kwargs, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    _sync()

    latencies_ms: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        _sync()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return {
        "median_ms": round(statistics.median(latencies_ms), 6),
        "p95_ms": round(_percentile(latencies_ms, 95), 6),
    }


def _infer_task_type(case: Dict[str, Any]) -> str:
    builder = case["input_builder"]
    if builder == "matmul_2d":
        return "cube"
    return "vector"


def _build_sentinel_tensors(device: torch.device):
    mm1 = torch.rand((10240, 10240), dtype=torch.float16, device=device)
    mm2 = torch.rand((10240, 10240), dtype=torch.float16, device=device)
    reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16, device=device)
    return mm1, mm2, reduce_input


def _run_profiling(fn, args, kwargs, *, device: torch.device, device_id: int, task_type: str, num_trials: int) -> Dict[str, Any]:
    prof_base_dir = Path(tempfile.mkdtemp(prefix=f"baseline_prof_{device_id}_"))
    path_suffix = f"device{device_id}_{int(time.time())}"
    prof_output_path = str(prof_base_dir / path_suffix)
    mm1, mm2, reduce_input = _build_sentinel_tensors(device)

    with ProfilingContext(prof_output_path, device_id):
        for _ in range(num_trials):
            _ = torch.matmul(mm1, mm2)
            _sync()
            _ = torch.max(reduce_input)
            _sync()
            _ = fn(*args, **kwargs)
            _sync()

    msprof_export(str(prof_base_dir), path_suffix)
    op_summary_path = locate_op_summary_file(prof_output_path)
    metrics = parse_op_summary(op_summary_path, task_type)
    metrics["error"] = ""
    return metrics


def run_case(case: Dict[str, Any], device_id: int, warmup: int, repeat: int, profiling: bool = False) -> Dict[str, Any]:
    device = _device_from_id(device_id)
    fn = _resolve_callable(case["api"])
    tensors = {name: _materialize_tensor(spec, device) for name, spec in case["inputs"].items()}
    args, kwargs = _prepare_call(case, tensors)
    timing = _run_e2e(fn, args, kwargs, warmup, repeat)
    profiling_result = {"error": "profiling_disabled"}
    if profiling:
        try:
            profiling_result = _run_profiling(
                fn,
                args,
                kwargs,
                device=device,
                device_id=device_id,
                task_type=_infer_task_type(case),
                num_trials=min(max(repeat, 5), 10),
            )
        except Exception as exc:
            profiling_result = {"error": str(exc)}
    return {
        "success": True,
        "timing": timing,
        "profiling": profiling_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    result = run_case(case, args.device_id, args.warmup, args.repeat, profiling=args.profiling)
    Path(args.result_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
