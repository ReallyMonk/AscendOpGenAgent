#!/usr/bin/env python3
"""Unified benchmark CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import List

from cannbench.scripts.baseline_runner import build_remote_case_payload, persist_run_output, run_baseline_case_remote
from cannbench.scripts.case_expander import expand_case
from cannbench.scripts.compare_results import write_compare_csv
from cannbench.scripts.custom_remote_runner import persist_custom_output, run_custom_case
from cannbench.scripts.registry_utils import BenchmarkRegistry, RunnerTemplateRegistry, ServerRegistry
from cannbench.scripts.result_writer import append_csv, write_json
from cannbench.remote.remote_client import RemoteWorkerClient


def _get_default_registry_path() -> str:
    """Resolve default registry path from package data."""
    return str(pkg_files("cannbench.data.registry").joinpath("lightweight_benchmark_registry_v2.json"))


def _get_default_templates_path() -> str:
    """Resolve default runner templates path from package data."""
    return str(pkg_files("cannbench.data.registry").joinpath("runner_templates_v1.json"))


def _get_default_servers_path() -> str:
    """Resolve default servers config path from package data."""
    return str(pkg_files("cannbench.data.registry").joinpath("servers.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cannbench-lite",
        description="Unified NPU benchmark CLI",
    )
    parser.add_argument("--registry", default=_get_default_registry_path(), help="Benchmark registry path")
    parser.add_argument("--runner-templates", default=_get_default_templates_path(), help="Runner templates path")
    parser.add_argument("--servers", default=_get_default_servers_path(), help="Server config path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List benchmark operators and shapes")
    list_parser.add_argument("--set", dest="set_name", help="Filter by benchmark set name")
    list_parser.add_argument("--layer", help="Filter by benchmark layer")
    list_parser.add_argument("--role", help="Filter by benchmark role")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--op", dest="op_name", help="Filter by operator name")
    list_parser.add_argument("--shape", dest="shape_id", help="Only show a specific shape id")
    list_parser.add_argument("--server", help="Show resolved server info for this server name")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    baseline_parser = subparsers.add_parser("run-baseline", help="Run benchmark baseline via torch_npu/PyTorch API")
    baseline_parser.add_argument("--set", dest="set_name", help="Filter by benchmark set name")
    baseline_parser.add_argument("--layer", help="Filter by benchmark layer")
    baseline_parser.add_argument("--role", help="Filter by benchmark role")
    baseline_parser.add_argument("--category", help="Filter by category")
    baseline_parser.add_argument("--op", dest="op_name", help="Filter by operator name")
    baseline_parser.add_argument("--shape", dest="shape_id", help="Only run a specific shape id")
    baseline_parser.add_argument("--all-shapes", action="store_true", help="Run all shapes for matched operators")
    baseline_parser.add_argument("--server", help="Server name defined in servers.json")
    baseline_parser.add_argument("--device", help="Override device id; defaults to server default_device")
    baseline_parser.add_argument("--dtype", help="Override dtype; defaults to the operator's first dtype")
    baseline_parser.add_argument("--warmup", type=int, default=20)
    baseline_parser.add_argument("--repeat", type=int, default=100)
    baseline_parser.add_argument("--profiling", action="store_true")
    baseline_parser.add_argument("--refresh-baseline", action="store_true")
    baseline_parser.add_argument("--run-id", help="Optional run id; defaults to timestamp")
    baseline_parser.add_argument(
        "--results-root",
        default="results",
        help="Root results directory containing baseline_cache/ and runs/",
    )

    custom_parser = subparsers.add_parser("run-custom", help="Run custom implementation on remote server")
    custom_parser.add_argument("--set", dest="set_name", help="Filter by benchmark set name")
    custom_parser.add_argument("--layer", help="Filter by benchmark layer")
    custom_parser.add_argument("--role", help="Filter by benchmark role")
    custom_parser.add_argument("--category", help="Filter by category")
    custom_parser.add_argument("--op", dest="op_name", required=True, help="Operator name")
    custom_parser.add_argument("--shape", dest="shape_id", required=True, help="Shape id")
    custom_parser.add_argument("--custom-dir", required=True, help="Custom implementation directory")
    custom_parser.add_argument("--server", help="Server name defined in servers.json")
    custom_parser.add_argument("--device", help="Override device id; defaults to server default_device")
    custom_parser.add_argument("--dtype", help="Override dtype; defaults to the operator's first dtype")
    custom_parser.add_argument("--profiling", action="store_true")
    custom_parser.add_argument("--run-id", help="Optional run id; defaults to timestamp")
    custom_parser.add_argument(
        "--results-root",
        default="results",
        help="Root results directory containing baseline_cache/ and runs/",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare baseline and custom results")
    compare_parser.add_argument("--results-dir", help="Single run directory containing both baseline and custom results")
    compare_parser.add_argument("--baseline-results-dir", help="Baseline run directory")
    compare_parser.add_argument("--custom-results-dir", help="Custom run directory")
    compare_parser.add_argument("--out-csv", help="Optional output CSV path; defaults to <results-dir>/compare.csv")

    return parser


def cmd_list(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.from_file(args.registry)
    templates = RunnerTemplateRegistry.from_file(args.runner_templates)
    servers = ServerRegistry.from_file(args.servers)
    server_info = servers.get(args.server)

    rows = []
    for op_name, operator in registry.iter_operators(
        set_name=args.set_name,
        layer=args.layer,
        role=args.role,
        category=args.category,
        op_name=args.op_name,
    ):
        if not templates.has(op_name):
            rows.append(
                {
                    "op_name": op_name,
                    "status": "missing_runner_template",
                    "benchmark_layer": operator["benchmark_layer"],
                    "benchmark_role": operator["benchmark_role"],
                    "category": operator["category"],
                }
            )
            continue

        template = templates.get(op_name)
        for shape in operator["shapes"]:
            if args.shape_id and shape["id"] != args.shape_id:
                continue
            expanded = expand_case(op_name, operator, shape, template)
            rows.append(
                {
                    "op_name": op_name,
                    "shape_id": expanded.shape_id,
                    "shape_tier": shape["shape_tier"],
                    "benchmark_layer": expanded.benchmark_layer,
                    "benchmark_role": expanded.benchmark_role,
                    "category": expanded.category,
                    "runner_type": expanded.runner_type,
                    "api": expanded.api,
                    "inputs": [{"name": inp.name, "shape": inp.shape} for inp in expanded.inputs],
                    "server_name": server_info.name,
                    "server_worker_url": server_info.worker_url,
                    "default_device": server_info.default_device,
                }
            )

    if args.json:
        print(json.dumps({"server": server_info.__dict__, "rows": rows}, indent=2, ensure_ascii=False))
        return 0

    print(f"server={server_info.name} worker_url={server_info.worker_url} default_device={server_info.default_device}")
    print(f"matched_cases={len(rows)}")
    for row in rows:
        if row.get("status") == "missing_runner_template":
            print(
                f"- {row['op_name']} [{row['benchmark_layer']}/{row['benchmark_role']}/{row['category']}] "
                f"status={row['status']}"
            )
            continue
        input_desc = ", ".join(f"{inp['name']}={inp['shape']}" for inp in row["inputs"])
        print(
            f"- {row['op_name']}:{row['shape_id']} "
            f"[{row['benchmark_layer']}/{row['benchmark_role']}/{row['category']}] "
            f"runner={row['runner_type']} api={row['api']} inputs=({input_desc})"
        )
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.from_file(args.registry)
    templates = RunnerTemplateRegistry.from_file(args.runner_templates)
    servers = ServerRegistry.from_file(args.servers)
    server_info = servers.get(args.server)
    device_id = str(args.device or server_info.default_device)

    matched_cases = []
    for op_name, operator in registry.iter_operators(
        set_name=args.set_name,
        layer=args.layer,
        role=args.role,
        category=args.category,
        op_name=args.op_name,
    ):
        if not templates.has(op_name):
            continue
        template = templates.get(op_name)
        for shape in operator["shapes"]:
            if args.shape_id and shape["id"] != args.shape_id:
                continue
            if not args.all_shapes and not args.shape_id and len(operator["shapes"]) > 1:
                if shape.get("shape_tier") != "main":
                    continue
            matched_cases.append(expand_case(op_name, operator, shape, template))

    if not matched_cases:
        raise SystemExit("no benchmark cases matched the provided filters")

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_root = Path(args.results_root)
    run_root = results_root / "runs" / run_id
    cache_root = results_root / "baseline_cache"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "registry_path": str(args.registry),
        "runner_template_path": str(args.runner_templates),
        "server_config_path": str(args.servers),
        "server_name": server_info.name,
        "worker_url": server_info.worker_url,
        "device": device_id,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "profiling": args.profiling,
        "created_at": datetime.now().isoformat(),
    }
    write_json(run_root / "manifest.json", manifest)

    print(
        f"run_id={run_id} server={server_info.name} worker_url={server_info.worker_url} "
        f"device={device_id} matched_cases={len(matched_cases)}"
    )

    for case in matched_cases:
        dtype = args.dtype or case.raw_operator["dtypes"][0]
        print(f"[baseline] {case.op_name}:{case.shape_id} dtype={dtype}")
        try:
            client = RemoteWorkerClient(server_info.worker_url, server_info.client_id)
            remote_result = asyncio.run(
                client.run_baseline_case(
                    build_remote_case_payload(case, dtype),
                    server_name=server_info.name,
                    device_id=int(device_id) if str(device_id).isdigit() else int(str(device_id).split(":")[-1]),
                    warmup=args.warmup,
                    repeat=args.repeat,
                    profiling=args.profiling,
                )
            )
            output = run_baseline_case_remote(
                case,
                server_name=server_info.name,
                hardware=server_info.hardware or "unknown",
                worker_url=server_info.worker_url,
                client=type("_ResolvedClient", (), {"run_baseline_case": lambda *a, **k: remote_result})(),
                device_id=device_id,
                dtype=dtype,
                warmup=args.warmup,
                repeat=args.repeat,
                profiling=args.profiling,
                cache_root=cache_root,
                refresh_baseline=args.refresh_baseline,
            )
            persist_run_output(
                output,
                run_root=run_root,
                run_id=run_id,
                server_name=server_info.name,
                worker_url=server_info.worker_url,
                op_name=case.op_name,
                shape_id=case.shape_id,
            )
            print(
                f"  status={output.result['status']} median_ms={output.result['e2e_median_ms']} "
                f"cache_hit={output.cache_hit}"
            )
        except Exception as exc:
            case_dir = run_root / "baseline" / case.op_name / case.shape_id
            write_json(
                case_dir / "result.json",
                {
                    "mode": "baseline",
                    "status": "failed",
                    "op_name": case.op_name,
                    "shape_id": case.shape_id,
                    "dtype": dtype,
                    "server_name": server_info.name,
                    "device": device_id,
                    "error": str(exc),
                    "created_at": datetime.now().isoformat(),
                },
            )
            append_csv(
                run_root / "summary.csv",
                [
                    {
                        "run_id": run_id,
                        "mode": "baseline",
                        "op_name": case.op_name,
                        "shape_id": case.shape_id,
                        "benchmark_layer": case.benchmark_layer,
                        "benchmark_role": case.benchmark_role,
                        "category": case.category,
                        "dtype": dtype,
                        "server_name": server_info.name,
                        "device": device_id,
                        "backend": case.runner_type,
                        "status": "failed",
                        "correctness": "",
                        "e2e_median_ms": "",
                        "e2e_p95_ms": "",
                        "profiling_available": False,
                        "profiling_path": "",
                        "custom_dir": "",
                        "worker_url": server_info.worker_url,
                        "created_at": datetime.now().isoformat(),
                        "source": "",
                        "cache_key": "",
                    }
                ],
            )
            print(f"  status=failed error={exc}")
            return 1
    return 0


def cmd_run_custom(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.from_file(args.registry)
    templates = RunnerTemplateRegistry.from_file(args.runner_templates)
    servers = ServerRegistry.from_file(args.servers)
    server_info = servers.get(args.server)
    device_id = str(args.device or server_info.default_device)

    operator = registry.get_operator(args.op_name)
    if not templates.has(args.op_name):
        raise SystemExit(f"missing runner template for op: {args.op_name}")
    template = templates.get(args.op_name)
    shape = registry.get_shape(args.op_name, args.shape_id)
    case = expand_case(args.op_name, operator, shape, template)
    dtype = args.dtype or operator["dtypes"][0]

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_root = Path(args.results_root)
    run_root = results_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "registry_path": str(args.registry),
        "runner_template_path": str(args.runner_templates),
        "server_config_path": str(args.servers),
        "server_name": server_info.name,
        "worker_url": server_info.worker_url,
        "device": device_id,
        "profiling": args.profiling,
        "created_at": datetime.now().isoformat(),
        "mode": "custom",
    }
    write_json(run_root / "manifest.json", manifest)

    custom_dir = Path(args.custom_dir)
    print(
        f"run_id={run_id} server={server_info.name} worker_url={server_info.worker_url} "
        f"device={device_id} op={case.op_name} shape={case.shape_id} custom_dir={custom_dir}"
    )
    try:
        output = run_custom_case(
            case,
            server_name=server_info.name,
            worker_url=server_info.worker_url,
            client_id=server_info.client_id,
            device_id=device_id,
            dtype=dtype,
            custom_dir=custom_dir,
            profiling=args.profiling,
        )
        persist_custom_output(
            output,
            run_root=run_root,
            run_id=run_id,
            op_name=case.op_name,
            shape_id=case.shape_id,
            impl_name=output.result["impl_name"],
        )
        print(f"  status={output.result['status']} returncode={output.result['returncode']}")
        if output.result["status"] != "success":
            return 1
        return 0
    except Exception as exc:
        case_dir = run_root / "custom" / case.op_name / case.shape_id / custom_dir.name
        write_json(
            case_dir / "result.json",
            {
                "mode": "custom",
                "status": "failed",
                "op_name": case.op_name,
                "shape_id": case.shape_id,
                "dtype": dtype,
                "server_name": server_info.name,
                "device": device_id,
                "worker_url": server_info.worker_url,
                "custom_dir": str(custom_dir),
                "error": str(exc),
                "created_at": datetime.now().isoformat(),
            },
        )
        append_csv(
            run_root / "summary.csv",
            [
                {
                    "run_id": run_id,
                    "mode": "custom",
                    "op_name": case.op_name,
                    "shape_id": case.shape_id,
                    "benchmark_layer": case.benchmark_layer,
                    "benchmark_role": case.benchmark_role,
                    "category": case.category,
                    "dtype": dtype,
                    "server_name": server_info.name,
                    "device": device_id,
                    "backend": "remote_custom",
                    "status": "failed",
                    "correctness": "",
                    "e2e_median_ms": "",
                    "e2e_p95_ms": "",
                    "profiling_available": False,
                    "profiling_path": "",
                    "custom_dir": str(custom_dir),
                    "worker_url": server_info.worker_url,
                    "created_at": datetime.now().isoformat(),
                    "source": "measured",
                    "cache_key": "",
                }
            ],
        )
        print(f"  status=failed error={exc}")
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    out_csv = Path(args.out_csv) if args.out_csv else None
    if args.baseline_results_dir or args.custom_results_dir:
        if not (args.baseline_results_dir and args.custom_results_dir):
            raise SystemExit("--baseline-results-dir and --custom-results-dir must be provided together")
        target = write_compare_csv(
            out_csv=out_csv,
            baseline_results_dir=Path(args.baseline_results_dir),
            custom_results_dir=Path(args.custom_results_dir),
        )
    else:
        if not args.results_dir:
            raise SystemExit("either --results-dir or both --baseline-results-dir/--custom-results-dir are required")
        target = write_compare_csv(Path(args.results_dir), out_csv)
    print(f"compare_csv={target}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return cmd_list(args)
    if args.command == "run-baseline":
        return cmd_run_baseline(args)
    if args.command == "run-custom":
        return cmd_run_custom(args)
    if args.command == "compare":
        return cmd_compare(args)

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
