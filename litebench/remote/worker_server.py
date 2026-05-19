#!/usr/bin/env python3
"""
OpenOps Remote Worker Server

Provides remote compilation and evaluation services for OpenOps clients.
No LLM required on server side - all code generation happens on client.

Usage:
    python worker_server.py --port 9001 --log-dir /path/to/logs

Environment Variables:
    ASCEND_HOME_PATH: Path to CANN installation (required)
"""

import os
import sys
import json
import logging
import tempfile
import tarfile
import shutil
import asyncio
import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global configuration
SERVER_CONFIG = {
    "log_dir": None,
    "max_workers": 4,
    "executor": None,
    "request_queue": None,
    "device_pool": None,
}


def sanitize_json_value(value):
    """Convert NaN/Inf recursively so FastAPI JSONResponse can serialize results."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(v) for v in value]
    return value


class DevicePool:
    """Device pool for managing NPU devices"""

    def __init__(self, device_list: List[int]):
        self.device_list = device_list
        self.available_devices = asyncio.Queue()
        self.condition = asyncio.Condition()

        for device_id in device_list:
            self.available_devices.put_nowait(device_id)

        logger.info(f"Initialized device pool with devices: {device_list}")

    async def acquire_device(self) -> int:
        """Acquire an available device"""
        async with self.condition:
            while self.available_devices.empty():
                await self.condition.wait()
            device_id = await self.available_devices.get()
            logger.info(f"Acquired device: {device_id}")
            return device_id

    async def acquire_specific_device(self, requested_device: int) -> int:
        """Acquire a specific device, waiting until it becomes available."""
        async with self.condition:
            if requested_device not in self.device_list:
                raise ValueError(f"Requested device {requested_device} not in device pool {self.device_list}")
            while True:
                drained = []
                found = None
                while not self.available_devices.empty():
                    device_id = await self.available_devices.get()
                    if device_id == requested_device and found is None:
                        found = device_id
                        break
                    drained.append(device_id)
                for device_id in drained:
                    await self.available_devices.put(device_id)
                if found is not None:
                    logger.info(f"Acquired requested device: {found}")
                    return found
                await self.condition.wait()

    async def release_device(self, device_id: int):
        """Release device back to pool"""
        async with self.condition:
            await self.available_devices.put(device_id)
            self.condition.notify()
            logger.info(f"Released device: {device_id}")


class RequestQueue:
    """Queue for handling multiple concurrent requests"""

    def __init__(self, max_workers: int = 4):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active_requests = {}

    async def acquire(self, request_id: str):
        """Acquire a slot for processing"""
        await self.semaphore.acquire()
        self.active_requests[request_id] = datetime.now()
        logger.info(
            f"[{request_id}] Acquired processing slot. Active requests: {len(self.active_requests)}"
        )

    def release(self, request_id: str):
        """Release a processing slot"""
        self.semaphore.release()
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        logger.info(
            f"[{request_id}] Released processing slot. Active requests: {len(self.active_requests)}"
        )


def check_cann_environment():
    """Check if CANN environment is properly configured"""
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        logger.error("ASCEND_HOME_PATH environment variable not set")
        return False

    if not Path(ascend_home).exists():
        logger.error(f"ASCEND_HOME_PATH does not exist: {ascend_home}")
        return False

    logger.info(f"CANN environment found: {ascend_home}")
    return True


def create_log_directory(client_id: str, op_name: str) -> Path:
    """Create a unique log directory for this request"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir_name = f"{client_id}_{op_name}_{timestamp}"
    log_dir = Path(SERVER_CONFIG["log_dir"]) / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def run_build_script(project_dir: Path, log_file: Path) -> tuple[bool, str]:
    """Run build.sh script and capture output"""
    import subprocess

    build_script = project_dir / "build.sh"
    if not build_script.exists():
        return False, f"build.sh not found in {project_dir}"

    try:
        # Make build.sh executable
        os.chmod(build_script, 0o755)

        # Run build script
        result = subprocess.run(
            ["bash", str(build_script)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
        )

        # Write logs
        log_content = f"=== Build Script Output ===\n"
        log_content += f"Return code: {result.returncode}\n\n"
        log_content += f"=== STDOUT ===\n{result.stdout}\n\n"
        log_content += f"=== STDERR ===\n{result.stderr}\n"

        log_file.write_text(log_content, encoding="utf-8")

        success = result.returncode == 0
        return success, log_content

    except subprocess.TimeoutExpired:
        error_msg = "Build script timed out after 10 minutes"
        log_file.write_text(error_msg, encoding="utf-8")
        return False, error_msg
    except Exception as e:
        error_msg = f"Build script failed: {str(e)}"
        log_file.write_text(error_msg, encoding="utf-8")
        return False, error_msg


def run_evaluation_script(
    eval_script: Path, op_name: str, install_dir: Path, log_file: Path, device_id: int
) -> tuple[bool, str]:
    """Run evaluation script and capture output"""
    import subprocess

    if not eval_script.exists():
        return False, f"Evaluation script not found: {eval_script}"

    try:
        # Set environment variables for evaluation
        env = os.environ.copy()
        # Point to the vendors/customize directory in the install location
        custom_opp_path = install_dir / "vendors" / "customize"
        if not custom_opp_path.exists():
            return False, f"ASCEND custom OPP directory not found: {custom_opp_path}"

        env["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
        # Set device ID for NPU
        env["ASCEND_DEVICE_ID"] = str(device_id)

        # Run evaluation script from the extract directory (where output/ folder is)
        # The evaluation script expects to find output/{op_name}/ directory
        cwd = (
            eval_script.parent
        )  # eval_script is in extract_dir, so parent is extract_dir

        # Use OpenOps venv Python if available (has torch_npu 2.9.0 instead of 2.8.0)
        # Try to find venv relative to the server script location
        python_executable = sys.executable

        # Look for venv in the OpenOps project root (parent of server directory)
        server_dir = Path(__file__).parent
        project_root = server_dir.parent
        venv_python = project_root / ".venv" / "bin" / "python3"

        if venv_python.exists():
            python_executable = str(venv_python)
            logger.info(f"Using OpenOps venv Python: {python_executable}")
        else:
            logger.info(f"Using system Python: {python_executable}")

        # Run evaluation script
        result = subprocess.run(
            [python_executable, str(eval_script), op_name],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
            env=env,
        )

        # Write logs with clear success/failure indication
        log_content = f"=== Evaluation Script Output ===\n"
        log_content += f"Device ID: {device_id}\n"
        log_content += f"Return code: {result.returncode}\n"
        log_content += (
            f"Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}\n\n"
        )
        log_content += f"=== STDOUT ===\n{result.stdout}\n\n"
        log_content += f"=== STDERR ===\n{result.stderr}\n"

        log_file.write_text(log_content, encoding="utf-8")

        success = result.returncode == 0

        # Log clear result
        if success:
            logger.info(
                f"Evaluation completed successfully. Output: {result.stdout[:200] if result.stdout else 'No output'}"
            )
        else:
            logger.error(
                f"Evaluation failed. Error: {result.stderr[:200] if result.stderr else 'No error message'}"
            )

        return success, log_content

    except subprocess.TimeoutExpired:
        error_msg = "Evaluation script timed out after 10 minutes"
        log_file.write_text(error_msg, encoding="utf-8")
        return False, error_msg
    except Exception as e:
        error_msg = f"Evaluation script failed: {str(e)}"
        log_file.write_text(error_msg, encoding="utf-8")
        return False, error_msg


async def process_compilation_request(
    package_data: bytes, client_id: str, op_name: str, request_id: str, requested_device: Optional[int] = None
) -> Dict[str, Any]:
    """Process a compilation request"""
    import subprocess

    log_dir = create_log_directory(client_id, op_name)
    logger.info(f"[{request_id}] Created log directory: {log_dir}")

    # Acquire device from pool
    device_pool = SERVER_CONFIG["device_pool"]
    if requested_device is not None:
        device_id = await device_pool.acquire_specific_device(requested_device)
    else:
        device_id = await device_pool.acquire_device()
    logger.info(f"[{request_id}] Using device: {device_id}")

    # Create temporary workspace in server output directory
    server_tmp_dir = Path(SERVER_CONFIG["log_dir"]) / "tmp"
    server_tmp_dir.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"openops_{op_name}_", dir=str(server_tmp_dir))

    try:
        # Extract package
        tar_path = Path(temp_dir) / "package.tar"
        tar_path.write_bytes(package_data)

        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_dir)

        logger.info(f"[{request_id}] Extracted package to {extract_dir}")

        # Read manifest
        manifest_path = extract_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("manifest.json not found in package")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logger.info(f"[{request_id}] Loaded manifest: {manifest.keys()}")

        # Find project directory - look for *Custom directory
        project_dirs = list(extract_dir.glob("*Custom"))
        if not project_dirs:
            raise ValueError(f"No *Custom directory found in package")

        project_dir = project_dirs[0]
        logger.info(f"[{request_id}] Found project directory: {project_dir.name}")

        # Run build script
        build_log = log_dir / "build.log"
        logger.info(f"[{request_id}] Running build script...")
        logger.info(f"[{request_id}] project_dir absolute: {project_dir.resolve()}")
        build_success, build_output = run_build_script(project_dir.resolve(), build_log)

        if not build_success:
            logger.error(f"[{request_id}] Build failed")
            return {
                "success": False,
                "stage": "compilation",
                "log": build_output,
                "log_dir": str(log_dir),
            }

        logger.info(f"[{request_id}] Build succeeded")

        # Copy build output to log directory (ignore errors for missing files)
        build_out_dir = project_dir / "build_out"
        if build_out_dir.exists():
            try:
                shutil.copytree(
                    build_out_dir,
                    log_dir / "build_out",
                    dirs_exist_ok=True,
                    ignore_dangling_symlinks=True,
                    ignore=lambda dir, files: [
                        f
                        for f in files
                        if os.path.islink(os.path.join(dir, f))
                        and not os.path.exists(os.path.join(dir, f))
                    ],
                )
                logger.info(f"[{request_id}] Copied build output to log directory")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to copy some build files: {e}")
                # Continue anyway - the important files (.run) should be there

        # Run evaluation if requested
        eval_success = True
        eval_output = "Evaluation not requested"

        if manifest.get("run_evaluation", False):
            eval_script_path = extract_dir / "evaluate.py"
            if eval_script_path.exists():
                # First, install the operator .run file
                run_file = build_out_dir / "custom_opp_ubuntu_aarch64.run"
                if run_file.exists():
                    # Install to a temporary location
                    install_dir = Path(temp_dir) / "install"
                    install_dir.mkdir(exist_ok=True)

                    logger.info(
                        f"[{request_id}] Installing operator to {install_dir}..."
                    )
                    install_result = subprocess.run(
                        ["bash", str(run_file), f"--install-path={install_dir}"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                    if install_result.returncode != 0:
                        logger.warning(
                            f"[{request_id}] Operator installation failed: {install_result.stderr}"
                        )
                        eval_success = False
                        eval_output = f"Installation failed: {install_result.stderr}"
                    else:
                        # Copy vendors directory to where evaluation script expects it
                        # The evaluation script looks for output/moe_router/vendors/customize
                        output_op_dir = extract_dir / "output" / op_name
                        output_op_dir.mkdir(parents=True, exist_ok=True)

                        # Copy the installed vendors directory
                        vendors_dest = output_op_dir / "vendors"
                        vendors_source = install_dir / "vendors"

                        if vendors_source.exists():
                            if vendors_dest.exists():
                                shutil.rmtree(vendors_dest)  # Remove existing directory
                            shutil.copytree(vendors_source, vendors_dest)
                            logger.info(
                                f"[{request_id}] Copied vendors directory: {vendors_source} -> {vendors_dest}"
                            )

                            # Verify the customize directory exists
                            customize_dir = vendors_dest / "customize"
                            if customize_dir.exists():
                                logger.info(
                                    f"[{request_id}] Verified customize directory exists: {customize_dir}"
                                )
                            else:
                                logger.warning(
                                    f"[{request_id}] Customize directory not found: {customize_dir}"
                                )
                        else:
                            logger.warning(
                                f"[{request_id}] Vendors source directory not found: {vendors_source}"
                            )

                        # Generate PyBind bindings before evaluation
                        generate_pybind_script = extract_dir / "generate_pybind.py"
                        if generate_pybind_script.exists():
                            logger.info(f"[{request_id}] Generating PyBind bindings...")
                            pybind_log = log_dir / "pybind_generation.log"

                            # Use OpenOps venv Python if available (has torch_npu 2.9.0 instead of 2.8.0)
                            # Try to find venv relative to the server script location
                            python_executable = sys.executable

                            # Look for venv in the OpenOps project root (parent of server directory)
                            server_dir = Path(__file__).parent
                            project_root = server_dir.parent
                            venv_python = project_root / ".venv" / "bin" / "python3"

                            if venv_python.exists():
                                python_executable = str(venv_python)
                                logger.info(
                                    f"[{request_id}] Using OpenOps venv Python for PyBind: {python_executable}"
                                )
                            else:
                                logger.info(
                                    f"[{request_id}] Using system Python for PyBind: {python_executable}"
                                )

                            pybind_result = subprocess.run(
                                [
                                    python_executable,
                                    str(generate_pybind_script),
                                    op_name,
                                ],
                                cwd=str(extract_dir),
                                capture_output=True,
                                text=True,
                                timeout=300,
                            )

                            # Write PyBind generation log
                            pybind_log_content = f"=== PyBind Generation Output ===\n"
                            pybind_log_content += (
                                f"Return code: {pybind_result.returncode}\n\n"
                            )
                            pybind_log_content += (
                                f"=== STDOUT ===\n{pybind_result.stdout}\n\n"
                            )
                            pybind_log_content += (
                                f"=== STDERR ===\n{pybind_result.stderr}\n"
                            )
                            pybind_log.write_text(pybind_log_content, encoding="utf-8")

                            if pybind_result.returncode != 0:
                                logger.warning(
                                    f"[{request_id}] PyBind generation failed: {pybind_result.stderr[:200]}"
                                )
                                eval_success = False
                                eval_output = (
                                    f"PyBind generation failed: {pybind_result.stderr}"
                                )
                            else:
                                logger.info(
                                    f"[{request_id}] PyBind bindings generated successfully"
                                )
                                # Now run evaluation
                                eval_log = log_dir / "evaluation.log"
                                logger.info(
                                    f"[{request_id}] Running evaluation on device {device_id}..."
                                )
                                eval_success, eval_output = run_evaluation_script(
                                    eval_script_path,
                                    op_name,
                                    install_dir,
                                    eval_log,
                                    device_id,
                                )

                                if eval_success:
                                    logger.info(f"[{request_id}] Evaluation succeeded")
                                else:
                                    logger.error(f"[{request_id}] Evaluation failed")
                        else:
                            logger.warning(
                                f"[{request_id}] PyBind generation script not found: {generate_pybind_script}"
                            )
                            # Try to run evaluation anyway
                            eval_log = log_dir / "evaluation.log"
                            logger.info(
                                f"[{request_id}] Running evaluation on device {device_id}..."
                            )
                            eval_success, eval_output = run_evaluation_script(
                                eval_script_path,
                                op_name,
                                install_dir,
                                eval_log,
                                device_id,
                            )
                else:
                    logger.warning(f"[{request_id}] .run file not found: {run_file}")
                    eval_success = False
                    eval_output = f".run file not found: {run_file}"

        # Prepare response
        return {
            "success": build_success and eval_success,
            "stage": "evaluation" if manifest.get("run_evaluation") else "compilation",
            "build_log": build_output,
            "eval_log": eval_output if manifest.get("run_evaluation") else None,
            "log_dir": str(log_dir),
            "device_id": device_id,
        }

    except Exception as e:
        logger.error(f"[{request_id}] Processing failed: {e}", exc_info=True)
        error_log = log_dir / "error.log"
        error_log.write_text(
            f"Error: {str(e)}\n{traceback.format_exc()}", encoding="utf-8"
        )

        return {
            "success": False,
            "stage": "error",
            "log": str(e),
            "log_dir": str(log_dir),
        }
    finally:
        # Release device back to pool
        await device_pool.release_device(device_id)
        logger.info(f"[{request_id}] Released device: {device_id}")

        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to cleanup temp dir: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize server resources on startup"""

    # Check CANN environment
    if not check_cann_environment():
        logger.error("CANN environment check failed. Server cannot start.")
        sys.exit(1)

    # Initialize device pool
    device_list = SERVER_CONFIG.get("device_list", [0])
    SERVER_CONFIG["device_pool"] = DevicePool(device_list)

    # Initialize request queue
    SERVER_CONFIG["request_queue"] = RequestQueue(
        max_workers=SERVER_CONFIG["max_workers"]
    )

    # Initialize thread pool executor
    SERVER_CONFIG["executor"] = ThreadPoolExecutor(
        max_workers=SERVER_CONFIG["max_workers"]
    )

    logger.info(f"OpenOps Worker Server initialized")
    logger.info(f"Log directory: {SERVER_CONFIG['log_dir']}")
    logger.info(f"Max concurrent workers: {SERVER_CONFIG['max_workers']}")
    logger.info(f"Available devices: {device_list}")

    yield

    # Cleanup
    if SERVER_CONFIG["executor"]:
        SERVER_CONFIG["executor"].shutdown(wait=True)
    logger.info("OpenOps Worker Server shutting down")


app = FastAPI(title="OpenOps Worker Service", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    device_pool = SERVER_CONFIG.get("device_pool")
    available_devices = device_pool.available_devices.qsize() if device_pool else 0

    return {
        "status": "healthy",
        "cann_available": check_cann_environment(),
        "active_requests": len(SERVER_CONFIG["request_queue"].active_requests)
        if SERVER_CONFIG["request_queue"]
        else 0,
        "available_devices": available_devices,
        "total_devices": len(SERVER_CONFIG.get("device_list", [])),
    }


@app.post("/api/v1/compile")
async def compile_operator(
    package: UploadFile = File(...),
    client_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(1800),
    device_id: Optional[int] = Form(None),
):
    """
    Compile and optionally evaluate an operator.

    Package should contain:
    - manifest.json: Metadata including op_name, run_evaluation flag
    - {op_name}Custom/: AscendC project directory with build.sh
    - evaluate.py: (optional) Evaluation script
    - Other necessary files

    Returns:
    - success: bool
    - stage: str (compilation/evaluation/error)
    - build_log: str
    - eval_log: str (if evaluation was run)
    - log_dir: str (path to log directory on server)
    """

    request_id = f"{client_id}_{op_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    queue = SERVER_CONFIG["request_queue"]

    try:
        # Acquire processing slot
        await queue.acquire(request_id)

        # Read package data
        package_data = await package.read()
        logger.info(f"[{request_id}] Received package: {len(package_data)} bytes")

        # Process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            SERVER_CONFIG["executor"],
            lambda: asyncio.run(
                process_compilation_request(
                    package_data, client_id, op_name, request_id, device_id
                )
            ),
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        queue.release(request_id)


async def process_profiling_request(
    package_data: bytes,
    client_id: str,
    op_name: str,
    request_id: str,
    num_trials: int = 10,
    requested_device: Optional[int] = None,
) -> Dict[str, Any]:
    """Process a profiling request — build, install, pybind, then profile with acl.prof."""
    import subprocess

    log_dir = create_log_directory(client_id, op_name)
    logger.info(f"[{request_id}] Created log directory: {log_dir}")

    device_pool = SERVER_CONFIG["device_pool"]
    if requested_device is not None:
        device_id = await device_pool.acquire_specific_device(requested_device)
    else:
        device_id = await device_pool.acquire_device()
    logger.info(f"[{request_id}] Using device: {device_id}")

    server_tmp_dir = Path(SERVER_CONFIG["log_dir"]) / "tmp"
    server_tmp_dir.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"openops_prof_{op_name}_", dir=str(server_tmp_dir))

    try:
        # Extract package (same as /compile)
        tar_path = Path(temp_dir) / "package.tar"
        tar_path.write_bytes(package_data)

        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_dir)

        logger.info(f"[{request_id}] Extracted package to {extract_dir}")

        # Read manifest
        manifest_path = extract_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("manifest.json not found in package")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logger.info(f"[{request_id}] Loaded manifest: {manifest.keys()}")

        # Find project directory
        project_dirs = list(extract_dir.glob("*Custom"))
        if not project_dirs:
            raise ValueError("No *Custom directory found in package")

        project_dir = project_dirs[0]
        logger.info(f"[{request_id}] Found project directory: {project_dir.name}")

        # Build
        build_log_file = log_dir / "build.log"
        logger.info(f"[{request_id}] Running build script...")
        logger.info(f"[{request_id}] project_dir absolute: {project_dir.resolve()}")
        build_success, build_output = run_build_script(project_dir.resolve(), build_log_file)

        if not build_success:
            logger.error(f"[{request_id}] Build failed")
            return {
                "success": False,
                "stage": "compilation",
                "build_log": build_output,
                "log_dir": str(log_dir),
            }

        logger.info(f"[{request_id}] Build succeeded")

        # Install .run file
        build_out_dir = project_dir / "build_out"
        run_file = build_out_dir / "custom_opp_ubuntu_aarch64.run"
        if not run_file.exists():
            return {
                "success": False,
                "stage": "installation",
                "build_log": build_output,
                "log_dir": str(log_dir),
                "log": f".run file not found: {run_file}",
            }

        install_dir = Path(temp_dir) / "install"
        install_dir.mkdir(exist_ok=True)

        logger.info(f"[{request_id}] Installing operator to {install_dir}...")
        install_result = subprocess.run(
            ["bash", str(run_file), f"--install-path={install_dir}"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if install_result.returncode != 0:
            return {
                "success": False,
                "stage": "installation",
                "build_log": build_output,
                "log_dir": str(log_dir),
                "log": f"Installation failed: {install_result.stderr}",
            }

        # Copy vendors directory for evaluation
        output_op_dir = extract_dir / "output" / op_name
        output_op_dir.mkdir(parents=True, exist_ok=True)
        vendors_dest = output_op_dir / "vendors"
        vendors_source = install_dir / "vendors"

        if vendors_source.exists():
            if vendors_dest.exists():
                shutil.rmtree(vendors_dest)
            shutil.copytree(vendors_source, vendors_dest)
            logger.info(f"[{request_id}] Copied vendors directory")

        # Generate PyBind bindings
        generate_pybind_script = extract_dir / "generate_pybind.py"
        if generate_pybind_script.exists():
            logger.info(f"[{request_id}] Generating PyBind bindings...")
            pybind_log = log_dir / "pybind_generation.log"

            python_executable = sys.executable
            server_dir = Path(__file__).parent
            project_root = server_dir.parent
            venv_python = project_root / ".venv" / "bin" / "python3"
            if venv_python.exists():
                python_executable = str(venv_python)

            pybind_result = subprocess.run(
                [python_executable, str(generate_pybind_script), op_name],
                cwd=str(extract_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )

            pybind_log_content = (
                f"=== PyBind Generation Output ===\n"
                f"Return code: {pybind_result.returncode}\n\n"
                f"=== STDOUT ===\n{pybind_result.stdout}\n\n"
                f"=== STDERR ===\n{pybind_result.stderr}\n"
            )
            pybind_log.write_text(pybind_log_content, encoding="utf-8")

            if pybind_result.returncode != 0:
                return {
                    "success": False,
                    "stage": "pybind_generation",
                    "build_log": build_output,
                    "log_dir": str(log_dir),
                    "log": f"PyBind generation failed: {pybind_result.stderr}",
                }

            logger.info(f"[{request_id}] PyBind bindings generated successfully")
        else:
            logger.warning(f"[{request_id}] PyBind script not found, proceeding anyway")

        # Run profiling as subprocess (NPU driver requires fresh process for ACL init)
        logger.info(f"[{request_id}] Running profiling on device {device_id}...")

        profiling_runner_script = Path(__file__).parent / "profiling_runner.py"
        result_file = log_dir / "profiling_result.json"
        profiling_log_file = log_dir / "profiling.log"

        # Set up environment for profiling subprocess.
        # NOTE: Do NOT set ASCEND_CUSTOM_OPP_PATH here — profiling_runner.py delays
        # that until AFTER reference model timing to avoid CANN dispatch interference.
        # profiling_runner.py handles both LD_LIBRARY_PATH and ASCEND_CUSTOM_OPP_PATH.
        prof_env = os.environ.copy()
        prof_env.pop("ASCEND_CUSTOM_OPP_PATH", None)  # ensure not inherited
        prof_env["ASCEND_DEVICE_ID"] = str(device_id)

        # Use venv Python if available
        python_executable = sys.executable
        server_dir = Path(__file__).parent
        project_root = server_dir.parent
        venv_python = project_root / ".venv" / "bin" / "python3"
        if venv_python.exists():
            python_executable = str(venv_python)

        prof_result = subprocess.run(
            [
                python_executable,
                str(profiling_runner_script),
                op_name,
                str(device_id),
                str(log_dir),
                str(extract_dir),
                "--num-trials", str(num_trials),
                "--result-file", str(result_file),
            ],
            cwd=str(extract_dir),
            capture_output=True,
            text=True,
            timeout=1800,
            env=prof_env,
        )

        # Write profiling log
        prof_log_content = (
            f"=== Profiling Runner Output ===\n"
            f"Return code: {prof_result.returncode}\n\n"
            f"=== STDOUT ===\n{prof_result.stdout}\n\n"
            f"=== STDERR ===\n{prof_result.stderr}\n"
        )
        profiling_log_file.write_text(prof_log_content, encoding="utf-8")

        if prof_result.returncode != 0:
            logger.error(f"[{request_id}] Profiling subprocess failed: {prof_result.stderr[:500]}")
            return {
                "success": False,
                "stage": "profiling",
                "build_log": build_output,
                "log": prof_log_content,
                "log_dir": str(log_dir),
            }

        # Read profiling result JSON
        if not result_file.exists():
            return {
                "success": False,
                "stage": "profiling",
                "build_log": build_output,
                "log": "Profiling result file not created",
                "log_dir": str(log_dir),
            }

        profiling_result = json.loads(result_file.read_text(encoding="utf-8"))
        logger.info(f"[{request_id}] Profiling completed")

        return {
            "success": True,
            "stage": "profiling",
            "timing": profiling_result.get("timing", {}),
            "profiling": profiling_result.get("profiling", {}),
            "build_log": build_output,
            "profile_log": profiling_result.get("profile_log", ""),
            "log_dir": str(log_dir),
            "device_id": device_id,
        }

    except Exception as e:
        logger.error(f"[{request_id}] Profiling request failed: {e}", exc_info=True)
        error_log = log_dir / "error.log"
        error_log.write_text(
            f"Error: {str(e)}\n{traceback.format_exc()}", encoding="utf-8"
        )
        return {
            "success": False,
            "stage": "error",
            "log": str(e),
            "log_dir": str(log_dir),
        }
    finally:
        await device_pool.release_device(device_id)
        logger.info(f"[{request_id}] Released device: {device_id}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to cleanup temp dir: {e}")


@app.post("/api/v1/profile")
async def profile_operator(
    package: UploadFile = File(...),
    client_id: str = Form(...),
    op_name: str = Form(...),
    num_trials: int = Form(10),
    timeout: int = Form(1800),
    device_id: Optional[int] = Form(None),
):
    """
    Profile an operator with hardware counters (acl.prof + msprof).

    Package should contain same files as /compile, plus evaluation files.
    Returns timing results AND profiling metrics (op_summary data).
    """
    request_id = f"{client_id}_{op_name}_prof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    queue = SERVER_CONFIG["request_queue"]

    try:
        await queue.acquire(request_id)

        package_data = await package.read()
        logger.info(f"[{request_id}] Received profiling package: {len(package_data)} bytes")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            SERVER_CONFIG["executor"],
            lambda: asyncio.run(
                process_profiling_request(
                    package_data, client_id, op_name, request_id, num_trials, device_id
                )
            ),
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"[{request_id}] Profiling request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        queue.release(request_id)


@app.get("/api/v1/status")
async def get_status():
    """Get server status"""
    queue = SERVER_CONFIG["request_queue"]
    device_pool = SERVER_CONFIG.get("device_pool")
    available_devices = device_pool.available_devices.qsize() if device_pool else 0

    return {
        "active_requests": len(queue.active_requests),
        "max_workers": SERVER_CONFIG["max_workers"],
        "log_dir": SERVER_CONFIG["log_dir"],
        "available_devices": available_devices,
        "total_devices": len(SERVER_CONFIG.get("device_list", [])),
        "device_list": SERVER_CONFIG.get("device_list", []),
    }


@app.post("/api/v1/run_baseline")
async def run_baseline(
    case_json: UploadFile = File(...),
    server_name: str = Form("default"),
    device_id: int = Form(0),
    warmup: int = Form(20),
    repeat: int = Form(100),
    profiling: str = Form("false"),
):
    """Run a generic torch/torch_npu baseline case on the worker."""
    import subprocess

    request_id = f"baseline_{server_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    queue = SERVER_CONFIG["request_queue"]
    await queue.acquire(request_id)

    try:
        log_dir = create_log_directory(server_name, "baseline")
        payload = await case_json.read()
        case_path = log_dir / "baseline_case.json"
        result_path = log_dir / "baseline_result.json"
        case_path.write_bytes(payload)

        python_executable = sys.executable
        server_dir = Path(__file__).parent
        project_root = server_dir.parent
        venv_python = project_root / ".venv" / "bin" / "python3"
        if venv_python.exists():
            python_executable = str(venv_python)

        runner_script = Path(__file__).parent / "baseline_benchmark_runner.py"
        proc = subprocess.run(
            [
                python_executable,
                str(runner_script),
                "--case-json",
                str(case_path),
                "--device-id",
                str(device_id),
                "--warmup",
                str(warmup),
                "--repeat",
                str(repeat),
                "--result-file",
                str(result_path),
                *(["--profiling"] if profiling.lower() == "true" else []),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if proc.returncode != 0:
            return JSONResponse(
                content={
                    "success": False,
                    "stage": "baseline_runner",
                    "log": proc.stderr or proc.stdout,
                    "log_dir": str(log_dir),
                    "device_id": device_id,
                }
            )

        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["log_dir"] = str(log_dir)
        result["device_id"] = device_id
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        return JSONResponse(content=sanitize_json_value(result))
    except Exception as e:
        logger.error(f"[{request_id}] Baseline request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        queue.release(request_id)


@app.post("/api/v1/generate_project")
async def generate_project(
    project_json: UploadFile = File(...),
    op_name: str = Form(...),
    client_id: str = Form("default_client"),
    force: str = Form("false"),
):
    """
    Generate AscendC project structure using msopgen.

    This endpoint allows clients without CANN to generate project structure.

    Args:
        project_json: JSON file defining the operator
        op_name: Operator name (e.g., 'relu', 'fast_gelu')
        client_id: Client identifier for logging
        force: If "true", overwrite existing project on server

    Returns:
        TAR archive containing the generated project structure
    """
    import subprocess
    from fastapi.responses import StreamingResponse
    import io

    request_id = f"{client_id}_{op_name}_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    force_overwrite = force.lower() == "true"
    logger.info(
        f"[{request_id}] Generating project for {op_name}, force={force_overwrite}"
    )

    # Create temporary workspace
    server_tmp_dir = Path(SERVER_CONFIG["log_dir"]) / "tmp"
    server_tmp_dir.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"gen_{op_name}_", dir=str(server_tmp_dir))

    try:
        # Save project JSON
        json_content = await project_json.read()
        json_path = Path(temp_dir) / f"{op_name}_project.json"
        json_path.write_bytes(json_content)
        logger.info(f"[{request_id}] Saved project JSON: {json_path}")

        # Import gen_project functions
        # Priority: 1) same directory as worker_server.py (for standalone deployment)
        #           2) development tree path (for local dev)
        server_dir = Path(__file__).parent
        dev_path = server_dir.parent / ".opencode/skills/ascend_call_generation/scripts"
        for candidate in [str(server_dir), str(dev_path)]:
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        from gen_project import prepare_ascend_project

        # Use per-request isolated output directory inside temp_dir
        output_base_dir = Path(temp_dir) / "output"
        output_dir = output_base_dir / op_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy JSON to output directory
        output_json = output_dir / f"{op_name}_project.json"
        shutil.copy2(json_path, output_json)

        # Generate project into the isolated temp directory
        logger.info(f"[{request_id}] Running msopgen in isolated workspace: {output_base_dir}")
        project_path = prepare_ascend_project(op_name, output_json, output_base_dir=output_base_dir)
        logger.info(f"[{request_id}] Project generated at: {project_path}")

        # Create TAR archive of the project
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            # Add the project directory
            tar.add(project_path, arcname=project_path.name)

        tar_buffer.seek(0)
        logger.info(
            f"[{request_id}] Created TAR archive ({len(tar_buffer.getvalue())} bytes)"
        )

        # Return TAR as streaming response
        return StreamingResponse(
            io.BytesIO(tar_buffer.getvalue()),
            media_type="application/x-tar",
            headers={
                "Content-Disposition": f"attachment; filename={project_path.name}.tar"
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Project generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Project generation failed: {str(e)}"
        )
    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to cleanup temp dir: {e}")


def main():
    parser = argparse.ArgumentParser(description="OpenOps Remote Worker Server")
    parser.add_argument("--port", type=int, default=9001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Directory for storing logs"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum concurrent workers"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated list of NPU device IDs (e.g., '0,1,2,3')",
    )

    args = parser.parse_args()

    # Parse device list
    try:
        device_list = [int(d.strip()) for d in args.devices.split(",")]
    except ValueError:
        logger.error(f"Invalid device list: {args.devices}")
        sys.exit(1)

    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    SERVER_CONFIG["log_dir"] = str(log_dir)
    SERVER_CONFIG["max_workers"] = args.max_workers
    SERVER_CONFIG["device_list"] = device_list

    # Start server
    logger.info(f"Starting OpenOps Worker Server on {args.host}:{args.port}")
    logger.info(f"Using NPU devices: {device_list}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    import traceback

    main()
