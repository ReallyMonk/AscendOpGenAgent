#!/usr/bin/env python3
"""
OpenOps Remote Client

Client-side module for sending compilation/evaluation requests to remote worker server.
"""

import httpx
import logging
import tarfile
import json
import io
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 3.0  # seconds, adjusted for SSH tunnel recovery
DEFAULT_MAX_RETRY_WAIT = 30.0  # max backoff cap in seconds

# Health pre-check configuration
HEALTH_CHECK_TIMEOUT = 10.0  # seconds
HEALTH_RETRY_DELAY = 10.0    # wait for tunnel recovery
HEALTH_MAX_RETRIES = 3       # health check retries before giving up


class RemoteWorkerClient:
    """Client for communicating with OpenOps remote worker server"""

    def __init__(self, worker_url: str, client_id: str = "default_client"):
        self.worker_url = worker_url.rstrip('/')
        self.client_id = client_id
        logger.info(f"Initialized RemoteWorkerClient: {self.worker_url}, client_id={self.client_id}")

    async def _ensure_healthy(self) -> bool:
        """
        Pre-check that the remote worker is reachable before sending a request.
        Retries with delay to allow SSH tunnel recovery.

        Returns:
            True if healthy, False if all retries exhausted.
        """
        for attempt in range(HEALTH_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                    response = await client.get(f"{self.worker_url}/health")
                    response.raise_for_status()
                    result = response.json()
                    if result.get("status") == "healthy":
                        return True
                    logger.warning(f"Worker unhealthy: {result}")
            except Exception as e:
                logger.warning(f"Health pre-check failed (attempt {attempt + 1}/{HEALTH_MAX_RETRIES}): {e}")

            if attempt < HEALTH_MAX_RETRIES - 1:
                logger.info(f"Waiting {HEALTH_RETRY_DELAY}s for tunnel recovery...")
                await asyncio.sleep(HEALTH_RETRY_DELAY)

        logger.error("Health pre-check exhausted — worker unreachable")
        return False

    def create_package(
        self,
        op_name: str,
        project_dir: Path,
        dsl_file: Path,
        run_evaluation: bool = False,
        eval_script: Optional[Path] = None,
        additional_files: Optional[Dict[str, Path]] = None
    ) -> bytes:
        """
        Create a TAR package for remote compilation.

        Args:
            op_name: Operator name
            project_dir: Path to {op_name}Custom directory
            dsl_file: Path to DSL file
            run_evaluation: Whether to run evaluation after compilation
            eval_script: Path to evaluation script (if run_evaluation=True)
            additional_files: Additional files to include {tar_path: file_path}

        Returns:
            bytes: TAR package data
        """
        tar_buffer = io.BytesIO()

        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            # Add manifest
            manifest = {
                "op_name": op_name,
                "client_id": self.client_id,
                "run_evaluation": run_evaluation
            }
            manifest_data = json.dumps(manifest, indent=2).encode('utf-8')
            manifest_info = tarfile.TarInfo(name="manifest.json")
            manifest_info.size = len(manifest_data)
            tar.addfile(manifest_info, io.BytesIO(manifest_data))

            # Add project directory - preserve the actual directory name
            if project_dir.exists():
                # Use the actual directory name from the path
                actual_dir_name = project_dir.name
                tar.add(project_dir, arcname=actual_dir_name)
                logger.info(f"Added project directory: {project_dir}")
            else:
                raise FileNotFoundError(f"Project directory not found: {project_dir}")

            # Add DSL file
            if dsl_file.exists():
                tar.add(dsl_file, arcname=f"{op_name}_dsl.py")
                logger.info(f"Added DSL file: {dsl_file}")

            # Add evaluation script if requested
            if run_evaluation and eval_script:
                if eval_script.exists():
                    tar.add(eval_script, arcname="evaluate.py")
                    logger.info(f"Added evaluation script: {eval_script}")
                else:
                    logger.warning(f"Evaluation script not found: {eval_script}")

            # Add additional files
            if additional_files:
                for tar_path, file_path in additional_files.items():
                    if file_path.exists():
                        tar.add(file_path, arcname=tar_path)
                        logger.info(f"Added additional file: {tar_path}")

        package_data = tar_buffer.getvalue()
        logger.info(f"Created package: {len(package_data)} bytes")
        return package_data

    async def compile_operator(
        self,
        package_data: bytes,
        op_name: str,
        device_id: Optional[int] = None,
        timeout: int = 1800,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF
    ) -> Dict[str, Any]:
        """
        Send compilation request to remote worker with automatic retry.

        Args:
            package_data: TAR package bytes
            op_name: Operator name
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for network errors
            retry_backoff: Initial backoff delay in seconds (doubles each retry)

        Returns:
            Dict with keys: success, stage, build_log, eval_log, log_dir
        """
        # Health pre-check before sending request
        if not await self._ensure_healthy():
            return {
                "success": False,
                "stage": "health_check",
                "log": "Worker unreachable after health pre-check retries",
                "log_dir": None
            }

        compile_url = f"{self.worker_url}/api/v1/compile"
        last_error = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait_time = min(retry_backoff * (2 ** (attempt - 1)), DEFAULT_MAX_RETRY_WAIT)
                logger.warning(f"Retry {attempt}/{max_retries} after {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Re-check health before retry (tunnel may have recovered)
                if not await self._ensure_healthy():
                    last_error = "Worker unreachable during retry health check"
                    continue

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout + 30, connect=30.0)
                ) as client:
                    files = {'package': ('package.tar', package_data, 'application/x-tar')}
                    data = {
                        'client_id': self.client_id,
                        'op_name': op_name,
                        'timeout': str(timeout)
                    }
                    if device_id is not None:
                        data['device_id'] = str(device_id)

                    logger.info(f"Sending compilation request to {compile_url} (attempt {attempt + 1}/{max_retries + 1})")
                    logger.info(f"Package size: {len(package_data)} bytes, timeout: {timeout}s")

                    response = await client.post(compile_url, files=files, data=data)
                    response.raise_for_status()

                    result = response.json()
                    logger.info(f"Compilation request completed: success={result.get('success')}, stage={result.get('stage')}")

                    return result

            except httpx.TimeoutException as e:
                last_error = f"Request timed out after {timeout}s: {e}"
                logger.error(last_error)
                # Timeouts are retryable (SSH tunnel may have recovered)
                continue

            except (httpx.RequestError, httpx.RemoteProtocolError) as e:
                last_error = f"Network error communicating with worker at {self.worker_url}: {e}"
                logger.error(last_error)
                # Network errors are retryable (SSH tunnel instability)
                continue

            except httpx.HTTPStatusError as e:
                # HTTP errors (4xx, 5xx) are NOT retryable — server received the request
                error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "stage": "http_error",
                    "log": error_msg,
                    "log_dir": None
                }

            except Exception as e:
                last_error = f"Remote compilation failed: {e}"
                logger.error(last_error, exc_info=True)
                # Unknown errors: retry conservatively
                continue

        # All retries exhausted
        error_msg = f"All {max_retries + 1} attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        return {
            "success": False,
            "stage": "network_error",
            "log": error_msg,
            "log_dir": None
        }

    async def profile_operator(
        self,
        package_data: bytes,
        op_name: str,
        device_id: Optional[int] = None,
        num_trials: int = 10,
        timeout: int = 1800,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ) -> Dict[str, Any]:
        """
        Send profiling request to remote worker with automatic retry.

        Same as compile_operator but hits /api/v1/profile and returns
        hardware profiling metrics in addition to timing data.

        Args:
            package_data: TAR package bytes
            op_name: Operator name
            num_trials: Number of profiling trials
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for network errors
            retry_backoff: Initial backoff delay in seconds (doubles each retry)

        Returns:
            Dict with keys: success, timing, profiling, build_log, profile_log, log_dir
        """
        # Health pre-check before sending request
        if not await self._ensure_healthy():
            return {
                "success": False,
                "stage": "health_check",
                "log": "Worker unreachable after health pre-check retries",
                "log_dir": None,
            }

        profile_url = f"{self.worker_url}/api/v1/profile"
        last_error = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait_time = min(retry_backoff * (2 ** (attempt - 1)), DEFAULT_MAX_RETRY_WAIT)
                logger.warning(f"Retry {attempt}/{max_retries} after {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Re-check health before retry
                if not await self._ensure_healthy():
                    last_error = "Worker unreachable during retry health check"
                    continue

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout + 30, connect=30.0)
                ) as client:
                    files = {'package': ('package.tar', package_data, 'application/x-tar')}
                    data = {
                        'client_id': self.client_id,
                        'op_name': op_name,
                        'num_trials': str(num_trials),
                        'timeout': str(timeout),
                    }
                    if device_id is not None:
                        data['device_id'] = str(device_id)

                    logger.info(f"Sending profiling request to {profile_url} (attempt {attempt + 1}/{max_retries + 1})")
                    logger.info(f"Package size: {len(package_data)} bytes, num_trials: {num_trials}, timeout: {timeout}s")

                    response = await client.post(profile_url, files=files, data=data)
                    response.raise_for_status()

                    result = response.json()
                    logger.info(f"Profiling request completed: success={result.get('success')}, stage={result.get('stage')}")

                    return result

            except httpx.TimeoutException as e:
                last_error = f"Request timed out after {timeout}s: {e}"
                logger.error(last_error)
                continue

            except (httpx.RequestError, httpx.RemoteProtocolError) as e:
                last_error = f"Network error communicating with worker at {self.worker_url}: {e}"
                logger.error(last_error)
                continue

            except httpx.HTTPStatusError as e:
                error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "stage": "http_error",
                    "log": error_msg,
                    "log_dir": None,
                }

            except Exception as e:
                last_error = f"Remote profiling failed: {e}"
                logger.error(last_error, exc_info=True)
                continue

        error_msg = f"All {max_retries + 1} attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        return {
            "success": False,
            "stage": "network_error",
            "log": error_msg,
            "log_dir": None,
        }

    async def check_health(self) -> Tuple[bool, str]:
        """
        Check if remote worker is healthy.

        Returns:
            Tuple[bool, str]: (is_healthy, message)
        """
        health_url = f"{self.worker_url}/health"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(health_url)
                response.raise_for_status()

                result = response.json()
                status = result.get("status")
                cann_available = result.get("cann_available")

                if status == "healthy" and cann_available:
                    return True, "Worker is healthy and CANN is available"
                elif status == "healthy":
                    return False, "Worker is healthy but CANN is not available"
                else:
                    return False, f"Worker status: {status}"

        except Exception as e:
            return False, f"Health check failed: {e}"

    async def generate_project(
        self,
        op_name: str,
        project_json_path: Path,
        output_dir: Path,
        force: bool = False,
        timeout: int = 300,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF
    ) -> Tuple[bool, str]:
        """
        Generate AscendC project structure on remote server.

        This allows clients without CANN to generate project structure.

        Args:
            op_name: Operator name (e.g., 'relu', 'fast_gelu')
            project_json_path: Path to project JSON file
            output_dir: Directory to save the generated project
            force: If True, overwrite existing project on server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for network errors
            retry_backoff: Initial backoff delay in seconds (doubles each retry)

        Returns:
            Tuple[bool, str]: (success, message/error)
        """
        gen_url = f"{self.worker_url}/api/v1/generate_project"

        if not project_json_path.exists():
            return False, f"Project JSON not found: {project_json_path}"

        # Health pre-check before sending request
        if not await self._ensure_healthy():
            return False, "Worker unreachable after health pre-check retries"

        # Read project JSON once
        with open(project_json_path, 'rb') as f:
            json_data = f.read()

        last_error = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait_time = min(retry_backoff * (2 ** (attempt - 1)), DEFAULT_MAX_RETRY_WAIT)
                logger.warning(f"Retry {attempt}/{max_retries} after {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Re-check health before retry
                if not await self._ensure_healthy():
                    last_error = "Worker unreachable during retry health check"
                    continue

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout + 10, connect=30.0)
                ) as client:
                    files = {'project_json': (project_json_path.name, json_data, 'application/json')}
                    data = {
                        'op_name': op_name,
                        'client_id': self.client_id,
                        'force': str(force).lower()
                    }

                    logger.info(f"Sending project generation request to {gen_url} (attempt {attempt + 1}/{max_retries + 1})")
                    logger.info(f"Operator: {op_name}, JSON size: {len(json_data)} bytes, force: {force}")

                    response = await client.post(gen_url, files=files, data=data)
                    response.raise_for_status()

                    # Save TAR response
                    tar_data = response.content
                    logger.info(f"Received project TAR: {len(tar_data)} bytes")

                    # Extract TAR to output directory
                    output_dir.mkdir(parents=True, exist_ok=True)
                    tar_buffer = io.BytesIO(tar_data)

                    with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                        tar.extractall(output_dir)

                    logger.info(f"Extracted project to: {output_dir}")
                    return True, f"Project generated successfully at {output_dir}"

            except httpx.TimeoutException as e:
                last_error = f"Request timed out after {timeout}s: {e}"
                logger.error(last_error)
                continue

            except (httpx.RequestError, httpx.RemoteProtocolError) as e:
                last_error = f"Network error: {e}"
                logger.error(last_error)
                continue

            except httpx.HTTPStatusError as e:
                error_msg = f"Server error: {e.response.status_code} - {e.response.text}"
                logger.error(error_msg)
                return False, error_msg

            except Exception as e:
                last_error = f"Project generation failed: {e}"
                logger.error(last_error, exc_info=True)
                continue

        error_msg = f"All {max_retries + 1} attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        return False, error_msg

    async def run_baseline_case(
        self,
        case_payload: Dict[str, Any],
        *,
        server_name: str,
        device_id: int,
        warmup: int,
        repeat: int,
        profiling: bool = False,
        timeout: int = 1800,
    ) -> Dict[str, Any]:
        """Run a generic baseline benchmark case on the remote worker."""
        if not await self._ensure_healthy():
            return {
                "success": False,
                "stage": "health_check",
                "log": "Worker unreachable after health pre-check retries",
                "log_dir": None,
            }

        baseline_url = f"{self.worker_url}/api/v1/run_baseline"
        case_bytes = json.dumps(case_payload).encode("utf-8")
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout + 30, connect=30.0)
        ) as client:
            files = {"case_json": ("baseline_case.json", case_bytes, "application/json")}
            data = {
                "server_name": server_name,
                "device_id": str(device_id),
                "warmup": str(warmup),
                "repeat": str(repeat),
                "profiling": "true" if profiling else "false",
            }
            response = await client.post(baseline_url, files=files, data=data)
            response.raise_for_status()
            return response.json()

    async def get_status(self) -> Dict[str, Any]:
        """
        Get worker status.

        Returns:
            Dict with keys: active_requests, max_workers, log_dir
        """
        status_url = f"{self.worker_url}/api/v1/status"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(status_url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get worker status: {e}")
            return {"error": str(e)}
