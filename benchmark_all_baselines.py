#!/usr/bin/env python3
"""
Run baseline performance benchmarks for all 23 CANNBench-Lite operators on npu:2.
Collects execution speed (median_ms, p95_ms) for each operator/shape combination.
Generates an HTML performance baseline report.

Usage: python3 benchmark_all_baselines.py
Output: baseline_lite/baseline_performance_report.html
"""

import json
import sys
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch_npu

# Add CANNBench-Lite to path
sys.path.insert(0, "/home/hrl/AscendOpGenAgent/CANNBench-Lite/src")
from cannbench.scripts.registry_utils import BenchmarkRegistry, RunnerTemplateRegistry
from cannbench.scripts.case_expander import expand_case
from cannbench.remote.baseline_benchmark_runner import run_case

BASELINE_LITE = Path("/home/hrl/AscendOpGenAgent/baseline_lite")
REGISTRY_PATH = "/home/hrl/AscendOpGenAgent/CANNBench-Lite/src/cannbench/data/registry/lightweight_benchmark_registry_v2.json"
TEMPLATES_PATH = "/home/hrl/AscendOpGenAgent/CANNBench-Lite/src/cannbench/data/registry/runner_templates_v1.json"
NPU_DEVICE = 2
WARMUP = 20
REPEAT = 100

OPERATORS = [
    "add_layer_norm", "add_rms_norm", "add_rms_norm_dynamic_quant",
    "add_rms_norm_quant", "avg_pool2d", "batch_mat_mul_v3",
    "clipped_swiglu", "conv2d_v2", "cross_entropy_loss",
    "embedding_bag", "flash_attention_score", "gather_v2",
    "ge_glu_v2", "gelu", "gelu_mul", "layer_norm_v3",
    "masked_softmax_with_rel_pos_bias", "mat_mul_v3", "relu",
    "rms_norm", "scaled_masked_softmax_v2", "softmax_v2", "swi_glu",
]


def benchmark_operator(op_name: str, shape_id: str, shape_tier: str, dtype: str,
                       case_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run benchmark for a single operator/shape combination."""
    result = run_case(case_payload, NPU_DEVICE, WARMUP, REPEAT, profiling=False)
    return {
        "op_name": op_name,
        "shape_id": shape_id,
        "shape_tier": shape_tier,
        "dtype": dtype,
        "success": result.get("success", False),
        "median_ms": result.get("timing", {}).get("median_ms", None),
        "p95_ms": result.get("timing", {}).get("p95_ms", None),
        "warmup": WARMUP,
        "repeat": REPEAT,
        "device": NPU_DEVICE,
    }


def build_case_payload(op_name: str, operator: Dict, shape: Dict,
                       template: Dict, dtype: str) -> Dict[str, Any]:
    """Build the case payload for baseline benchmarking."""
    case = expand_case(op_name, operator, shape, template)
    inputs = {}
    for inp in case.inputs:
        d = dtype
        spec_dtype = inp.spec.get("dtype")
        if spec_dtype:
            d = spec_dtype
        inputs[inp.name] = {
            "shape": inp.shape,
            "dtype": d,
            "distribution": inp.spec.get("distribution", "normal"),
            "low": inp.spec.get("low"),
            "high": inp.spec.get("high"),
            "threshold": inp.spec.get("threshold"),
        }
    return {
        "op_name": case.op_name,
        "shape_id": case.shape_id,
        "api": case.api,
        "runner_type": case.runner_type,
        "input_builder": case.raw_template["input_builder"],
        "attrs": case.attrs,
        "shape_params": case.raw_shape["params"],
        "inputs": inputs,
    }


def main():
    print("=" * 80)
    print("CANNBench-Lite: Baseline Performance Benchmark")
    print(f"Device: npu:{NPU_DEVICE}, Warmup: {WARMUP}, Repeat: {REPEAT}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    registry = BenchmarkRegistry.from_file(REGISTRY_PATH)
    templates = RunnerTemplateRegistry.from_file(TEMPLATES_PATH)

    # Set device
    torch_npu.npu.set_device(NPU_DEVICE)
    print(f"Using device: npu:{NPU_DEVICE}")

    all_results = []
    errors = []
    total_cases = 0

    for op_idx, op_name in enumerate(OPERATORS):
        operator = registry.get_operator(op_name)
        if not templates.has(op_name):
            print(f"\n[{op_idx+1}/23] {op_name}: SKIP (no template)")
            errors.append({"op": op_name, "error": "no template"})
            continue

        template = templates.get(op_name)
        dtypes = operator["dtypes"]
        shapes = operator["shapes"]

        print(f"\n[{op_idx+1}/23] {op_name}: {len(shapes)} shapes, dtypes={dtypes}")

        for shape in shapes:
            shape_id = shape["id"]
            shape_tier = shape.get("shape_tier", "")
            dtype = dtypes[0]  # Use first dtype

            total_cases += 1
            print(f"  {shape_id} ({shape_tier}, {dtype}) ... ", end="", flush=True)

            try:
                case_payload = build_case_payload(op_name, operator, shape, template, dtype)
                result = benchmark_operator(op_name, shape_id, shape_tier, dtype, case_payload)

                if result["success"]:
                    median = result["median_ms"]
                    p95 = result["p95_ms"]
                    print(f"✅ median={median:.4f}ms p95={p95:.4f}ms")
                else:
                    print(f"❌ failed")
                    errors.append({"op": op_name, "shape": shape_id, "error": "benchmark returned success=false"})

                all_results.append(result)
            except Exception as e:
                print(f"❌ {type(e).__name__}: {e}")
                errors.append({"op": op_name, "shape": shape_id, "error": str(e)})
                all_results.append({
                    "op_name": op_name,
                    "shape_id": shape_id,
                    "shape_tier": shape_tier,
                    "dtype": dtype,
                    "success": False,
                    "median_ms": None,
                    "p95_ms": None,
                    "error": str(e),
                })

    # Save raw results
    results_path = BASELINE_LITE / "baseline_performance_results.json"
    results_data = {
        "config": {
            "device": NPU_DEVICE, "warmup": WARMUP, "repeat": REPEAT,
            "timestamp": datetime.now().isoformat(),
        },
        "total_cases": total_cases,
        "results": all_results,
        "errors": errors,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    # Generate HTML report
    generate_html_report(all_results, errors, total_cases, OPERATORS)
    print(f"Report saved: {BASELINE_LITE / 'baseline_performance_report.html'}")


def generate_html_report(results: List[Dict], errors: List[Dict],
                         total_cases: int, operators: List[str]):
    """Generate a comprehensive HTML performance report."""

    success_count = sum(1 for r in results if r.get("success"))
    fail_count = total_cases - success_count

    # Group results by operator
    by_op = {}
    for r in results:
        op = r["op_name"]
        if op not in by_op:
            by_op[op] = []
        by_op[op].append(r)

    # Find global min/max for color scaling
    all_medians = [r["median_ms"] for r in results if r.get("success") and r["median_ms"] is not None]
    global_min = min(all_medians) if all_medians else 0
    global_max = max(all_medians) if all_medians else 1

    def speed_color(ms):
        """Green for fast, red for slow."""
        if ms is None:
            return "#ccc"
        ratio = min((ms - global_min) / max(global_max - global_min, 0.001), 1.0)
        if ratio < 0.2:
            return "#4CAF50"  # Fast - green
        elif ratio < 0.5:
            return "#8BC34A"
        elif ratio < 0.7:
            return "#FFC107"
        elif ratio < 0.9:
            return "#FF9800"
        else:
            return "#F44336"  # Slow - red

    # Build operator rows
    op_rows = []
    for op_name in operators:
        op_results = by_op.get(op_name, [])
        if not op_results:
            op_rows.append(f"""
            <tr>
                <td class="op-name">{op_name}</td>
                <td colspan="4" style="color:#999">no data</td>
            </tr>""")
            continue

        # Get shapes
        shape_rows = []
        for r in sorted(op_results, key=lambda x: x["shape_id"]):
            median = r.get("median_ms")
            p95 = r.get("p95_ms")
            success = r.get("success", False)
            if success and median is not None:
                color = speed_color(median)
                shape_rows.append(
                    f'<span class="shape-badge" style="background:{color}">'
                    f'{r["shape_id"]}:{median:.4f}ms</span> '
                )
            else:
                shape_rows.append(
                    f'<span class="shape-badge" style="background:#ccc">'
                    f'{r["shape_id"]}:FAIL</span> '
                )

        # Overall stats for operator
        op_medians = [r["median_ms"] for r in op_results if r.get("success") and r["median_ms"] is not None]
        avg_ms = statistics.mean(op_medians) if op_medians else None
        min_ms = min(op_medians) if op_medians else None
        max_ms = max(op_medians) if op_medians else None

        op_rows.append(f"""
            <tr>
                <td class="op-name">{op_name}</td>
                <td>{len(op_results)}</td>
                <td>{f'{avg_ms:.4f}' if avg_ms else 'N/A'}</td>
                <td>{f'{min_ms:.4f}' if min_ms else 'N/A'}</td>
                <td>{f'{max_ms:.4f}' if max_ms else 'N/A'}</td>
            </tr>""")

    # Build shape detail rows
    detail_rows = []
    for op_name in operators:
        op_results = by_op.get(op_name, [])
        if not op_results:
            detail_rows.append(f"""
            <tr>
                <td class="op-name">{op_name}</td>
                <td colspan="5" style="color:#888">no benchmark data available</td>
            </tr>""")
            continue

        for r in sorted(op_results, key=lambda x: x["shape_id"]):
            median = r.get("median_ms")
            p95 = r.get("p95_ms")
            success = r.get("success", False)
            if success and median is not None:
                color = speed_color(median)
                detail_rows.append(f"""
                <tr>
                    <td class="op-name">{r['op_name']}</td>
                    <td>{r['shape_id']}</td>
                    <td><span class="tier-tag">{r.get('shape_tier', '')}</span></td>
                    <td>{r.get('dtype', '')}</td>
                    <td style="background:{color};color:white;font-weight:bold">{median:.6f}</td>
                    <td>{p95:.6f}</td>
                </tr>""")
            else:
                detail_rows.append(f"""
                <tr>
                    <td class="op-name">{r['op_name']}</td>
                    <td>{r['shape_id']}</td>
                    <td><span class="tier-tag">{r.get('shape_tier', '')}</span></td>
                    <td>{r.get('dtype', '')}</td>
                    <td style="background:#f44336;color:white">FAILED</td>
                    <td>-</td>
                </tr>""")

    # Build error section
    error_rows = ""
    if errors:
        error_rows = "<h2>Errors</h2><table><tr><th>Operator</th><th>Shape</th><th>Error</th></tr>"
        for e in errors:
            error_rows += f"<tr><td>{e.get('op', '')}</td><td>{e.get('shape', '')}</td><td style='color:red'>{e.get('error', '')[:200]}</td></tr>"
        error_rows += "</table>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CANNBench-Lite AscendC Origin Baseline Performance Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f5f7fa; color: #333; padding: 20px; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ text-align: center; color: #1a237e; margin-bottom: 8px; font-size: 24px; }}
    .subtitle {{ text-align: center; color: #666; margin-bottom: 24px; font-size: 14px; }}
    .summary-cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
    .card {{ flex: 1; min-width: 180px; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
    .card .value {{ font-size: 32px; font-weight: 700; color: #1a237e; }}
    .card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
    .card.success .value {{ color: #2e7d32; }}
    .card.fail .value {{ color: #c62828; }}
    .card.perf .value {{ color: #e65100; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
    th {{ background: #1a237e; color: white; padding: 12px 16px; text-align: left; font-size: 13px; font-weight: 600; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #eee; font-size: 13px; }}
    tr:hover {{ background: #f8f9ff; }}
    .op-name {{ font-weight: 600; color: #1a237e; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; }}
    .shape-badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; color: white; font-size: 11px; margin: 1px; font-family: monospace; }}
    .tier-tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; background: #e3f2fd; color: #1565c0; }}
    h2 {{ color: #1a237e; margin: 24px 0 12px 0; font-size: 18px; border-bottom: 2px solid #1a237e; padding-bottom: 6px; }}
    .legend {{ display: flex; gap: 12px; align-items: center; margin-bottom: 16px; font-size: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 4px; }}
    .legend-color {{ width: 16px; height: 16px; border-radius: 3px; }}
    .totals-row {{ background: #e8eaf6; font-weight: 700; }}
    .note {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 12px 16px; border-radius: 4px; margin: 16px 0; font-size: 13px; color: #e65100; }}
    .nav {{ display: flex; gap: 8px; margin-bottom: 16px; }}
    .nav a {{ padding: 8px 16px; background: #1a237e; color: white; text-decoration: none; border-radius: 6px; font-size: 13px; }}
    .nav a:hover {{ background: #283593; }}
</style>
</head>
<body>
<div class="container">
    <h1>🚀 CANNBench-Lite AscendC Origin Baseline Performance Report</h1>
    <p class="subtitle">
        Hardware: Ascend 910B4 (npu:{NPU_DEVICE}) &nbsp;|&nbsp;
        Warmup: {WARMUP} &nbsp;|&nbsp; Repeat: {REPEAT} &nbsp;|&nbsp;
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{len(operators)}</div>
            <div class="label">Total Operators</div>
        </div>
        <div class="card">
            <div class="value">{total_cases}</div>
            <div class="label">Total Shape Cases</div>
        </div>
        <div class="card success">
            <div class="value">{success_count}</div>
            <div class="label">Passed</div>
        </div>
        <div class="card fail">
            <div class="value">{fail_count}</div>
            <div class="label">Failed</div>
        </div>
        <div class="card perf">
            <div class="value">{global_min:.4f}</div>
            <div class="label">Fastest (ms)</div>
        </div>
        <div class="card perf">
            <div class="value">{global_max:.4f}</div>
            <div class="label">Slowest (ms)</div>
        </div>
    </div>

    <div class="legend">
        <span>Speed:</span>
        <div class="legend-item"><div class="legend-color" style="background:#4CAF50"></div>Fast (&lt;{global_min + (global_max-global_min)*0.2:.1f}ms)</div>
        <div class="legend-item"><div class="legend-color" style="background:#8BC34A"></div></div>
        <div class="legend-item"><div class="legend-color" style="background:#FFC107"></div></div>
        <div class="legend-item"><div class="legend-color" style="background:#FF9800"></div></div>
        <div class="legend-item"><div class="legend-color" style="background:#F44336"></div>Slow (&gt;{global_min + (global_max-global_min)*0.9:.1f}ms)</div>
    </div>

    <h2>📊 Operator Performance Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Operator</th>
                <th>Shapes</th>
                <th>Avg (ms)</th>
                <th>Min (ms)</th>
                <th>Max (ms)</th>
            </tr>
        </thead>
        <tbody>
            {"".join(op_rows)}
        </tbody>
    </table>

    <h2>📋 Detailed Shape-Level Performance</h2>
    <table>
        <thead>
            <tr>
                <th>Operator</th>
                <th>Shape</th>
                <th>Tier</th>
                <th>Dtype</th>
                <th>Median (ms)</th>
                <th>P95 (ms)</th>
            </tr>
        </thead>
        <tbody>
            {"" .join(detail_rows)}
        </tbody>
    </table>

    {error_rows}

    <div class="note">
        <strong>💡 Note:</strong> This report captures <em>origin AscendC baseline</em> performance —
        the reference PyTorch operator running on NPU ({torch_npu.__version__ if hasattr(torch_npu, '__version__') else 'unknown'}).
        These values serve as the performance baseline against which optimized AscendC kernel implementations are compared.
        Shorter execution time (lower ms) = faster = better.
    </div>
</div>
</body>
</html>"""

    report_path = BASELINE_LITE / "baseline_performance_report.html"
    report_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
