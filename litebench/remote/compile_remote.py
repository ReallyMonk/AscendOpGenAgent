#!/usr/bin/env python3
"""
Remote Compilation Helper Script

This script helps the lingxi_code agent perform remote compilation.
It generates all AscendC code on the client side and sends it to the remote worker.

Supports two modes:
  --run-evaluation  Compile + evaluate (correctness + timing)
  --run-profiling   Compile + profile (timing + hardware counters via acl.prof)

These two flags are mutually exclusive.
"""

import sys
import json
import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path

# Add local remote directory to path
sys.path.insert(0, str(Path(__file__).parent))

from remote_client import RemoteWorkerClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _prepare_eval_files(output_path: Path, op_name: str):
    """Prepare evaluation-related files (shared by both evaluation and profiling modes)."""
    eval_script = None
    generate_pybind_script = None
    additional_files = {}

    benchmark_root = Path(__file__).resolve().parents[1]
    repo_root = benchmark_root.parent
    cwd_root = Path.cwd()
    asset_root = benchmark_root / "assets"

    _eval_candidates = [
        asset_root / "evaluate.py",
        cwd_root / "benchmark/assets/evaluate.py",
        repo_root / "benchmark/assets/evaluate.py",
        cwd_root / ".claude/skills/op_evaluation/ascendc_evaluation/scripts/evaluate.py",
        repo_root / ".claude/skills/op_evaluation/ascendc_evaluation/scripts/evaluate.py",
        cwd_root / ".opencode/skills/ascendc_evaluation/scripts/evaluate.py",
        repo_root / ".opencode/skills/ascendc_evaluation/scripts/evaluate.py",
    ]
    eval_script = next((p for p in _eval_candidates if p.exists()), None)
    if eval_script is None:
        logger.warning(f"Evaluation script not found in any candidate path")

    # Look for generate_pybind script
    _pybind_candidates = [
        asset_root / "generate_pybind.py",
        cwd_root / "benchmark/assets/generate_pybind.py",
        repo_root / "benchmark/assets/generate_pybind.py",
        cwd_root / ".claude/skills/op_evaluation/ascendc_evaluation/scripts/generate_pybind.py",
        repo_root / ".claude/skills/op_evaluation/ascendc_evaluation/scripts/generate_pybind.py",
        cwd_root / ".opencode/skills/ascendc_evaluation/scripts/generate_pybind.py",
        repo_root / ".opencode/skills/ascendc_evaluation/scripts/generate_pybind.py",
    ]
    generate_pybind_script = next((p for p in _pybind_candidates if p.exists()), None)
    if generate_pybind_script is None:
        logger.warning(f"Generate PyBind script not found in any candidate path")

    # Add reference and custom Python files for evaluation
    reference_file = output_path / op_name / f"{op_name}_reference.py"
    custom_file = output_path / op_name / f"{op_name}_custom.py"
    cpp_file = output_path / op_name / f"{op_name}.cpp"

    if reference_file.exists():
        additional_files[f"output/{op_name}/{op_name}_reference.py"] = reference_file
        logger.info(f"Adding reference file for evaluation: {reference_file}")
    else:
        logger.warning(f"Reference file not found: {reference_file}")

    if custom_file.exists():
        additional_files[f"output/{op_name}/{op_name}_custom.py"] = custom_file
        logger.info(f"Adding custom file for evaluation: {custom_file}")
    else:
        logger.warning(f"Custom file not found: {custom_file}")

    if cpp_file.exists():
        additional_files[f"output/{op_name}/{op_name}.cpp"] = cpp_file
        logger.info(f"Adding cpp file for PyBind: {cpp_file}")
    else:
        logger.warning(f"CPP file not found: {cpp_file}")

    # Add generate_pybind script and its template directory
    if generate_pybind_script and generate_pybind_script.exists():
        additional_files["generate_pybind.py"] = generate_pybind_script
        logger.info(f"Adding generate_pybind script: {generate_pybind_script}")

        # Add template directory
        template_dir = generate_pybind_script.parent / "template"
        if template_dir.exists():
            # Add all files in template directory recursively
            for template_file in template_dir.rglob("*"):
                if template_file.is_file():
                    rel_path = template_file.relative_to(generate_pybind_script.parent)
                    additional_files[str(rel_path)] = template_file
            logger.info(f"Adding template directory: {template_dir}")

    return eval_script, additional_files


async def compile_remote(
    worker_url: str,
    client_id: str,
    op_name: str,
    output_dir: str,
    device_id: int | None = None,
    run_evaluation: bool = False,
    run_profiling: bool = False,
    num_trials: int = 10,
    timeout: int = 1800,
):
    """
    Compile operator on remote worker.

    Args:
        worker_url: Remote worker URL (e.g., http://worker-server:9001)
        client_id: Client identifier
        op_name: Operator name
        output_dir: Output directory containing generated files
        run_evaluation: Whether to run evaluation after compilation
        run_profiling: Whether to run profiling after compilation
        num_trials: Number of profiling trials (only for --run-profiling)
        timeout: Request timeout in seconds
    """

    output_path = Path(output_dir)
    if not output_path.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return False

    # Initialize client
    client = RemoteWorkerClient(worker_url, client_id)

    # Check worker health
    logger.info("Checking remote worker health...")
    is_healthy, health_msg = await client.check_health()
    if not is_healthy:
        logger.error(f"Remote worker is not healthy: {health_msg}")
        return False

    logger.info(f"Remote worker is healthy: {health_msg}")

    # Get worker status
    status = await client.get_status()
    logger.info(f"Worker status: {status}")

    # Prepare paths - try both lowercase and capitalized versions
    project_dir = output_path / op_name / f"{op_name}Custom"
    if not project_dir.exists():
        # Try capitalized version (e.g., MoeRouterCustom)
        op_name_capitalized = ''.join(word.capitalize() for word in op_name.split('_'))
        project_dir = output_path / op_name / f"{op_name_capitalized}Custom"

    dsl_file = output_path / op_name / f"{op_name}_dsl.py"

    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        return False

    if not dsl_file.exists():
        logger.error(f"DSL file not found: {dsl_file}")
        return False

    # Prepare evaluation/profiling files
    eval_script = None
    additional_files = {}
    needs_eval_files = run_evaluation or run_profiling

    if needs_eval_files:
        eval_script, additional_files = _prepare_eval_files(output_path, op_name)

    # Create package
    logger.info("Creating package...")
    try:
        package_data = client.create_package(
            op_name=op_name,
            project_dir=project_dir,
            dsl_file=dsl_file,
            run_evaluation=run_evaluation,
            eval_script=eval_script,
            additional_files=additional_files if additional_files else None
        )
    except Exception as e:
        logger.error(f"Failed to create package: {e}")
        return False

    # Send request based on mode
    if run_profiling:
        logger.info(f"Sending profiling request to {worker_url}...")
        result = await client.profile_operator(
            package_data=package_data,
            op_name=op_name,
            device_id=device_id,
            num_trials=num_trials,
            timeout=timeout,
        )
    else:
        logger.info(f"Sending compilation request to {worker_url}...")
        result = await client.compile_operator(
            package_data=package_data,
            op_name=op_name,
            device_id=device_id,
            timeout=timeout,
        )

    # Process result
    success = result.get("success", False)
    stage = result.get("stage", "unknown")
    log_dir = result.get("log_dir")

    if run_profiling:
        logger.info(f"Profiling result: success={success}, stage={stage}")
    else:
        logger.info(f"Compilation result: success={success}, stage={stage}")

    if success:
        if run_profiling:
            logger.info("Remote profiling succeeded!")

            # Print timing results
            timing = result.get("timing", {})
            if timing:
                logger.info(f"\n{'='*60}")
                logger.info(f"TIMING RESULTS:")
                logger.info(f"  Reference:  {timing.get('ref_median_ms', 'N/A')} ms")
                logger.info(f"  Custom:     {timing.get('custom_median_ms', 'N/A')} ms")
                logger.info(f"  Speedup:    {timing.get('speedup', 'N/A')}x")
                logger.info(f"{'='*60}")

            # Print profiling results
            profiling = result.get("profiling", {})
            if profiling:
                logger.info(f"\nPROFILING METRICS:")
                for key, value in profiling.items():
                    if key != "raw_rows":
                        logger.info(f"  {key}: {value}")

            # Save profiling JSON to output directory
            profiling_output_dir = output_path / op_name
            profiling_output_dir.mkdir(parents=True, exist_ok=True)

            # Find next round number (use max+1 to handle gaps from deleted files)
            existing = list(profiling_output_dir.glob("profiling_round_*.json"))
            if existing:
                round_nums = []
                for p in existing:
                    try:
                        round_nums.append(int(p.stem.split('_')[-1]))
                    except ValueError:
                        pass
                round_num = max(round_nums) + 1 if round_nums else 0
            else:
                round_num = 0
            profiling_json_path = profiling_output_dir / f"profiling_round_{round_num}.json"

            # Compute content hashes for traceability
            dsl_hash = ""
            if dsl_file.exists():
                dsl_hash = hashlib.sha256(dsl_file.read_bytes()).hexdigest()[:8]

            kernel_file = project_dir / "op_kernel" / f"{op_name}_custom.cpp"
            kernel_hash = ""
            if kernel_file.exists():
                kernel_hash = hashlib.sha256(kernel_file.read_bytes()).hexdigest()[:8]

            profiling_data = {
                "round": round_num,
                "dsl_hash": dsl_hash,
                "kernel_hash": kernel_hash,
                "timestamp": datetime.now().isoformat(),
                "timing": timing,
                "profiling": profiling,
            }
            profiling_json_path.write_text(
                json.dumps(profiling_data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(f"Profiling data saved to: {profiling_json_path}")

            # Print profile log
            profile_log = result.get("profile_log", "")
            if profile_log:
                logger.info(f"\n=== Profile Log ===\n{profile_log}")

        else:
            logger.info("Remote compilation succeeded!")
            if log_dir:
                logger.info(f"Server logs stored at: {log_dir}")

            # Print build log
            build_log = result.get("build_log", "")
            if build_log:
                logger.info("=== Build Log ===")
                print(build_log)

            # Print evaluation log if available
            eval_log = result.get("eval_log")
            if eval_log:
                logger.info("=== Evaluation Results ===")
                print(eval_log)

                # Extract and highlight key evaluation metrics
                if "STDERR" in eval_log:
                    stderr_section = eval_log.split("=== STDERR ===")[1] if "=== STDERR ===" in eval_log else ""
                    if "Evaluation correctness:" in stderr_section:
                        for line in stderr_section.split('\n'):
                            if "Evaluation correctness:" in line:
                                logger.info(f"\n{'='*60}")
                                logger.info(f"CORRECTNESS: {line.split('Evaluation correctness:')[1].strip()}")
                            elif "Match rate:" in line:
                                logger.info(f"  {line.strip()}")
                            elif "Example mismatch" in line:
                                logger.info(f"  {line.strip()}")
                            elif "Evaluation performance:" in line:
                                logger.info(f"PERFORMANCE: {line.split('Evaluation performance:')[1].strip()}")
                        logger.info(f"{'='*60}\n")

    else:
        logger.error("Remote operation failed!")
        if log_dir:
            logger.error(f"Server logs stored at: {log_dir}")

        # Print error log
        error_log = result.get("log", result.get("build_log", ""))
        if error_log:
            logger.error("=== Error Log ===")
            print(error_log)

    return success


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Remote Compilation Helper")
    parser.add_argument("--worker-url", required=True, help="Remote worker URL")
    parser.add_argument("--client-id", default="default_client", help="Client ID")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--device-id", type=int, help="Optional target NPU device id on the remote worker")
    parser.add_argument("--timeout", type=int, default=1800, help="Request timeout in seconds")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--run-evaluation", action="store_true", help="Run evaluation after compilation")
    mode_group.add_argument("--run-profiling", action="store_true", help="Run profiling after compilation (hardware counters)")

    parser.add_argument("--num-trials", type=int, default=10, help="Number of profiling trials (only with --run-profiling)")

    args = parser.parse_args()

    # Run async function
    success = asyncio.run(compile_remote(
        worker_url=args.worker_url,
        client_id=args.client_id,
        op_name=args.op_name,
        output_dir=args.output_dir,
        device_id=args.device_id,
        run_evaluation=args.run_evaluation,
        run_profiling=args.run_profiling,
        num_trials=args.num_trials,
        timeout=args.timeout,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
