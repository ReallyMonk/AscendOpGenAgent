"""CANNBench-Lite MCP Server

将 CANNBench-Lite 的算子基准测试能力封装为 MCP 工具，供 AI Agent 直接调用。

工具列表：
    list_operators        列出 registry 中所有算子及 shape
    get_operator_info     查询单个算子详情（shapes、dtypes、bottleneck_tags）
    list_baseline_operators  列出有 baseline 源码快照的算子
    get_baseline_reference   获取算子 baseline AscendC 实现 + 参考 shape
    local_run_custom      本地直接编译执行自定义实现（无需 worker）
    remote_run_custom     通过远程 worker 编译执行自定义实现
    get_session_token_usage  估算会话 token 使用量（与 performance-report schema 对齐）

启动方式：
    python -m cannbench.mcp_server

环境变量：
    CANNBENCH_DATA_DIR  自定义数据目录（默认使用包内数据）
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# ── 数据目录 ─────────────────────────────────────────────────────────────────

_env_data_dir = os.environ.get("CANNBENCH_DATA_DIR")


def _data_path(*parts: str) -> Path:
    if _env_data_dir:
        return Path(_env_data_dir).joinpath(*parts)
    return Path(str(pkg_files("cannbench.data").joinpath(*parts)))


def _registry_path() -> Path:
    return _data_path("registry", "lightweight_benchmark_registry_v2.json")


def _runner_templates_path() -> Path:
    return _data_path("registry", "runner_templates_v1.json")


def _servers_path() -> Path:
    return _data_path("registry", "servers.json")


# ── 数据加载 ─────────────────────────────────────────────────────────────────


def _load_registry() -> Dict[str, Any]:
    return json.loads(_registry_path().read_text(encoding="utf-8"))


def _load_baseline_ref(op_name: str) -> Dict[str, Any]:
    p = _data_path("baseline_reference", f"{op_name}.json")
    if not p.exists():
        raise FileNotFoundError(f"baseline_reference/{op_name}.json 不存在")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_baseline_summary() -> Dict[str, Any]:
    p = _data_path("baseline_reference", "summary.json")
    if not p.exists():
        return {"operators": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="cannbench-lite",
    instructions=(
        "CANNBench-Lite 是 NPU 算子基准测试框架。"
        "提供算子注册表查询、baseline AscendC 源码快照获取、"
        "本地和远程自定义实现编译执行能力。"
        "在算子优化前，可用 get_baseline_reference 获取参考实现和测试 shape；"
        "优化完成后可用 local_run_custom 或 remote_run_custom 验证正确性与性能。"
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# 只读工具
# ══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def list_operators(category: Optional[str] = None, layer: Optional[str] = None) -> str:
    """列出 registry 中所有算子，可按 category 或 benchmark_layer 过滤。

    Args:
        category: 按类别过滤，如 activation、normalization、matmul、pooling、embedding
        layer: 按 benchmark_layer 过滤，如 foundation、composite
    """
    data = _load_registry()
    operators = data.get("operators", {})
    result = []
    for name, op in operators.items():
        if category and op.get("category") != category:
            continue
        if layer and op.get("benchmark_layer") != layer:
            continue
        result.append({
            "name": name,
            "category": op.get("category", ""),
            "benchmark_role": op.get("benchmark_role", ""),
            "benchmark_layer": op.get("benchmark_layer", ""),
            "dtypes": op.get("dtypes", []),
            "shape_count": len(op.get("shapes", [])),
        })
    return _json({"total": len(result), "operators": result})


@mcp.tool()
def get_operator_info(op_name: str) -> str:
    """查询单个算子的完整信息，包含 shapes、dtypes、description、bottleneck_tags。

    Args:
        op_name: 算子名称，如 gelu、mat_mul_v3、rms_norm
    """
    data = _load_registry()
    operators = data.get("operators", {})
    if op_name not in operators:
        available = sorted(operators.keys())
        return _json({"error": f"算子 '{op_name}' 不存在", "available_operators": available})
    return _json(operators[op_name])


@mcp.tool()
def list_baseline_operators() -> str:
    """列出所有有 baseline 源码快照的算子。"""
    summary = _load_baseline_summary()
    return _json(summary)


@mcp.tool()
def get_baseline_reference(op_name: str, file_name: Optional[str] = None) -> str:
    """获取算子的 baseline AscendC 实现源码及参考测试 shape。

    不传 file_name 时返回全部文件（含 content 源码字段）；
    传 file_name 时只返回该单个文件。

    Args:
        op_name: 算子名称，如 gelu、rms_norm、mat_mul_v3
        file_name: 指定单个文件名，如 gelu_kernel.h；不传则返回全部
    """
    ref = _load_baseline_ref(op_name)
    files = ref.get("files", {})

    # 从 registry 获取参考 shape 和 dtypes
    registry = _load_registry()
    op_info = registry.get("operators", {}).get(op_name, {})
    ref_shapes = op_info.get("shapes", [])
    dtypes = op_info.get("dtypes", [])

    if file_name:
        if file_name not in files:
            return _json({
                "error": f"文件 '{file_name}' 不存在",
                "available_files": sorted(files.keys()),
            })
        return _json({
            "operator_name": op_name,
            "dtypes": dtypes,
            "reference_shapes": ref_shapes,
            "file": files[file_name],
        })

    return _json({
        "operator_name": op_name,
        "dtypes": dtypes,
        "reference_shapes": ref_shapes,
        "total_files": len(files),
        "file_list": sorted(files.keys()),
        "files": files,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 本地执行工具
# ══════════════════════════════════════════════════════════════════════════════


def _find_build_dir(op_root: Path) -> Optional[Path]:
    """在算子根目录下查找构建目录。

    优先级: kernel/ > {op_root.name}_custom/ > *Custom/
    兼容 agent 产出结构（kernel/）和旧 CANNBench 结构（*Custom/）。
    """
    kernel = op_root / "kernel"
    if kernel.is_dir() and (kernel / "build.sh").exists():
        return kernel
    direct = op_root / f"{op_root.name}_custom"
    if direct.is_dir() and (direct / "build.sh").exists():
        return direct
    for candidate in op_root.glob("*Custom"):
        if candidate.is_dir() and (candidate / "build.sh").exists():
            return candidate
    return None


def _resolve_eval_files(op_root: Path, op_name: str) -> Dict[str, Optional[Path]]:
    """探测评测所需的 reference 和 custom Python 文件。

    支持两种命名约定:
      - agent 结构: model.py / model_new_ascendc.py
      - CANNBench 结构: {op_name}_reference.py / {op_name}_custom.py

    Returns:
        {"reference": Path|None, "custom": Path|None}
    """
    result: Dict[str, Optional[Path]] = {"reference": None, "custom": None}

    # CANNBench 命名
    ref_cannbench = op_root / f"{op_name}_reference.py"
    cust_cannbench = op_root / f"{op_name}_custom.py"
    if ref_cannbench.exists():
        result["reference"] = ref_cannbench
    if cust_cannbench.exists():
        result["custom"] = cust_cannbench

    # agent 命名 (fallback)
    if result["reference"] is None:
        ref_agent = op_root / "model.py"
        if ref_agent.exists():
            result["reference"] = ref_agent
    if result["custom"] is None:
        cust_agent = op_root / "model_new_ascendc.py"
        if cust_agent.exists():
            result["custom"] = cust_agent

    return result


def _ensure_eval_symlinks(op_root: Path, op_name: str, eval_files: Dict[str, Optional[Path]]) -> list[str]:
    """确保 evaluate.py 期望的 {op}_reference.py / {op}_custom.py 存在。

    如果文件使用了 agent 命名（model.py / model_new_ascendc.py），创建软链接。
    Returns: 创建的软链接路径列表（用于日志）。
    """
    created: list[str] = []
    mapping = {
        "reference": f"{op_name}_reference.py",
        "custom": f"{op_name}_custom.py",
    }
    for role, expected_name in mapping.items():
        src = eval_files.get(role)
        if src is None:
            continue
        target = op_root / expected_name
        if target == src:
            continue  # 已是期望的文件名
        if target.exists() or target.is_symlink():
            continue  # 已存在（不覆盖）
        target.symlink_to(src.name)
        created.append(f"{target.name} -> {src.name}")
    return created


def _ensure_vendors(op_root: Path, build_dir: Optional[Path]) -> Optional[str]:
    """确保 vendors/customize/op_api/lib/ 目录存在且含 .so 文件。

    evaluate.py 的 setup_ascend_runtime_environment() 强制要求此路径。
    若 vendors/ 缺失但 build_dir 下有 .so，自动复制过去。

    Returns: 操作说明字符串，或 None（无需操作）。
    """
    vendors_lib = op_root / "vendors" / "customize" / "op_api" / "lib"
    # 已就绪
    if vendors_lib.is_dir() and any(vendors_lib.glob("*.so")):
        return None

    # 从 build_dir 查找 .so
    if build_dir is None:
        return f"vendors 缺失且无构建目录: {vendors_lib}"

    so_files = list(build_dir.glob("build/*.so")) or list(build_dir.glob("*.so"))
    if not so_files:
        return f"vendors 缺失且构建目录下无 .so: {build_dir}"

    # 创建目录并复制
    vendors_lib.mkdir(parents=True, exist_ok=True)
    for so in so_files:
        dest = vendors_lib / so.name
        if not dest.exists():
            shutil.copy2(str(so), str(dest))
    return f"已从 {build_dir.name}/ 复制 {len(so_files)} 个 .so 到 vendors/customize/op_api/lib/"


def _run_build(project_dir: Path) -> subprocess.CompletedProcess:
    build_script = project_dir / "build.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"build.sh 不存在: {build_script}")
    return subprocess.run(
        ["bash", "build.sh"],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _run_eval(output_path: Path, op_name: str, device_id: str) -> subprocess.CompletedProcess:
    eval_script = _data_path("assets", "evaluate.py")
    if not eval_script.exists():
        raise FileNotFoundError(f"evaluate.py 不存在: {eval_script}")
    return subprocess.run(
        [sys.executable, str(eval_script), op_name, "--output-path", str(output_path), "--device", f"npu:{device_id}"],
        capture_output=True,
        text=True,
        timeout=600,
    )


def _parse_timing(log: str) -> Dict[str, Any]:
    timing: Dict[str, Any] = {}
    import re as _re
    for line in log.splitlines():
        s = line.strip()
        # 格式 1: "Evaluation performance: ref=0.115ms, custom=0.309ms, speedup=0.37x"
        m = _re.search(r'ref=([0-9.]+)\s*ms', s)
        if m:
            timing["ref_median_ms"] = float(m.group(1))
        m = _re.search(r'custom=([0-9.]+)\s*ms', s)
        if m:
            timing["custom_median_ms"] = float(m.group(1))
        m = _re.search(r'speedup=([0-9.]+)', s)
        if m:
            timing["speedup"] = float(m.group(1))
        # 格式 2 (兼容旧版): "Reference: 0.115 ms" / "Custom: 0.309 ms" / "Speedup: 0.37x"
        if "Reference:" in s and "INFO" in s and "ref_median_ms" not in timing:
            val = s.split("Reference:")[1].replace("ms", "").strip()
            try:
                timing["ref_median_ms"] = float(val)
            except ValueError:
                pass
        elif "Custom:" in s and "INFO" in s and "custom_median_ms" not in timing:
            val = s.split("Custom:")[1].replace("ms", "").strip()
            try:
                timing["custom_median_ms"] = float(val)
            except ValueError:
                pass
        elif "Speedup:" in s and "INFO" in s and "speedup" not in timing:
            val = s.split("Speedup:")[1].strip().rstrip("x")
            try:
                timing["speedup"] = float(val)
            except ValueError:
                pass
        # 正确性状态
        if "[PASS]" in s:
            timing["correctness"] = "pass"
        elif "[FAIL]" in s:
            timing["correctness"] = "fail"
    return timing


def _parse_profiling(log: str) -> Dict[str, Any]:
    profiling: Dict[str, Any] = {}
    in_metrics = False
    for line in log.splitlines():
        s = line.strip()
        if "PROFILING METRICS:" in s:
            in_metrics = True
            continue
        if in_metrics and "INFO:" in s and ":" in s.split("INFO:")[-1]:
            payload = s.split("INFO:")[-1].strip() if "INFO:" in s else ""
            if not payload:
                continue
            key, _, value = payload.partition(":")
            key, value = key.strip(), value.strip()
            if key == "raw_rows":
                continue
            try:
                profiling[key] = float(value)
            except ValueError:
                pass
    return profiling


@mcp.tool()
def local_run_custom(
    op_name: str,
    custom_dir: str,
    shape_id: Optional[str] = None,
    device: str = "0",
    dtype: Optional[str] = None,
    profiling: bool = False,
    skip_build: bool = False,
) -> str:
    """在本地 NPU 编译并执行自定义实现，验证正确性并测量性能。无需远程 worker。

    自动适配两种目录结构:
      - agent 结构: custom_dir/kernel/build.sh + model.py + model_new_ascendc.py
      - CANNBench 结构: custom_dir/<op>/<Op>Custom/build.sh + <op>_reference.py + <op>_custom.py

    Args:
        op_name: 算子名称
        custom_dir: 自定义实现目录的绝对路径（算子根目录或其父目录）
        shape_id: shape ID（可选，仅用于 registry 校验；不传时跳过校验）
        device: NPU 设备 ID，默认 0
        dtype: 数据类型，如 float16、bfloat16；不传则使用算子默认 dtype
        profiling: 是否采集 profiling 数据（耗时更长）
        skip_build: 跳过编译步骤（已有 .so 编译产物时设为 True）
    """
    from cannbench.scripts.registry_utils import BenchmarkRegistry

    custom_path = Path(custom_dir).resolve()
    if not custom_path.exists():
        return _json({"status": "failed", "error": f"目录不存在: {custom_dir}"})

    # ── 定位算子根目录 ──────────────────────────────────────────────────
    op_root = custom_path / op_name if (custom_path / op_name).is_dir() else custom_path

    # ── 定位构建目录 ────────────────────────────────────────────────────
    build_dir = _find_build_dir(op_root)
    if build_dir is None and not skip_build:
        return _json({
            "status": "failed",
            "error": f"在 {op_root} 下未找到构建目录（kernel/ 或 *Custom/，需含 build.sh）",
            "contents": sorted([p.name for p in op_root.iterdir() if p.is_dir()]),
        })

    # ── 定位评测文件 ────────────────────────────────────────────────────
    eval_files = _resolve_eval_files(op_root, op_name)
    if eval_files["reference"] is None or eval_files["custom"] is None:
        missing = []
        if eval_files["reference"] is None:
            missing.append(f"{op_name}_reference.py 或 model.py")
        if eval_files["custom"] is None:
            missing.append(f"{op_name}_custom.py 或 model_new_ascendc.py")
        return _json({
            "status": "failed",
            "error": f"在 {op_root} 下缺少评测文件: {', '.join(missing)}",
            "found_files": sorted([p.name for p in op_root.glob("*.py")]),
        })

    # 确保 evaluate.py 期望的文件名存在（自动创建软链接）
    symlinks = _ensure_eval_symlinks(op_root, op_name, eval_files)

    # ── Registry 校验（可选）────────────────────────────────────────────
    resolved_dtype = dtype
    if shape_id or not dtype:
        try:
            registry = BenchmarkRegistry.from_file(str(_registry_path()))
            operator = registry.get_operator(op_name)
            if not resolved_dtype:
                resolved_dtype = operator["dtypes"][0]
            if shape_id:
                try:
                    registry.get_shape(op_name, shape_id)
                except KeyError:
                    available = [s["id"] for s in operator.get("shapes", [])]
                    return _json({"status": "failed", "error": f"shape_id '{shape_id}' 不存在", "available_shapes": available})
        except (KeyError, FileNotFoundError):
            if not resolved_dtype:
                resolved_dtype = "float16"

    # ── Step 1: 编译 ────────────────────────────────────────────────────
    build_log = ""
    if not skip_build and build_dir is not None:
        build_result = _run_build(build_dir)
        build_log = (build_result.stdout or "") + (build_result.stderr or "")
        if build_result.returncode != 0:
            return _json({
                "status": "failed",
                "step": "build",
                "returncode": build_result.returncode,
                "build_log": build_log[-3000:],
                "op_name": op_name,
                "shape_id": shape_id or "",
                "build_dir": str(build_dir),
            })

    # ── Step 1.5: 确保 vendors/ 就绪 ────────────────────────────────────
    vendors_msg = _ensure_vendors(op_root, build_dir)

    # ── Step 2: 评估 ────────────────────────────────────────────────────
    eval_result = _run_eval(op_root, op_name, device)
    eval_log = (eval_result.stdout or "") + (eval_result.stderr or "")
    success = eval_result.returncode == 0

    timing = _parse_timing(eval_log)
    prof = _parse_profiling(eval_log) if profiling else {}

    return _json({
        "status": "success" if success else "failed",
        "op_name": op_name,
        "shape_id": shape_id or "",
        "dtype": resolved_dtype,
        "device": device,
        "timing": timing,
        "profiling": prof if prof else None,
        "symlinks_created": symlinks if symlinks else None,
        "vendors_setup": vendors_msg,
        "build_skipped": skip_build,
        "build_log": build_log[-1500:] if build_log and not success else "",
        "eval_log": eval_log[-3000:],
        "returncode": eval_result.returncode,
        "created_at": datetime.now().isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 远程执行工具
# ══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def remote_run_custom(
    op_name: str,
    shape_id: str,
    custom_dir: str,
    server: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    profiling: bool = False,
) -> str:
    """通过远程 worker 编译并执行自定义实现。需要 cannbench-worker 已在目标机器上运行。

    Args:
        op_name: 算子名称
        shape_id: shape ID，如 S1、S2、S3
        custom_dir: 自定义实现目录的绝对路径
        server: 服务器名称（对应 servers.json 中的 key）；不传则使用默认服务器
        device: NPU 设备 ID；不传则使用服务器默认设备
        dtype: 数据类型；不传则使用算子默认 dtype
        profiling: 是否采集 profiling 数据
    """
    from cannbench.scripts.case_expander import expand_case
    from cannbench.scripts.custom_remote_runner import run_custom_case, persist_custom_output
    from cannbench.scripts.registry_utils import BenchmarkRegistry, RunnerTemplateRegistry, ServerRegistry

    registry = BenchmarkRegistry.from_file(str(_registry_path()))
    templates = RunnerTemplateRegistry.from_file(str(_runner_templates_path()))
    servers = ServerRegistry.from_file(str(_servers_path()))

    try:
        operator = registry.get_operator(op_name)
    except KeyError:
        return _json({"status": "failed", "error": f"算子 '{op_name}' 不在 registry 中"})

    if not templates.has(op_name):
        return _json({"status": "failed", "error": f"算子 '{op_name}' 缺少 runner template"})

    server_info = servers.get(server)
    device_id = str(device or server_info.default_device)
    template = templates.get(op_name)

    try:
        shape = registry.get_shape(op_name, shape_id)
    except KeyError:
        available = [s["id"] for s in operator.get("shapes", [])]
        return _json({"status": "failed", "error": f"shape_id '{shape_id}' 不存在", "available_shapes": available})

    case = expand_case(op_name, operator, shape, template)
    resolved_dtype = dtype or operator["dtypes"][0]

    try:
        output = run_custom_case(
            case,
            server_name=server_info.name,
            worker_url=server_info.worker_url,
            client_id=server_info.client_id,
            device_id=device_id,
            dtype=resolved_dtype,
            custom_dir=Path(custom_dir),
            profiling=profiling,
        )
    except Exception as exc:
        return _json({"status": "failed", "error": str(exc)})

    # 提取关键字段，避免返回过多原始日志
    result = output.result
    summary = {
        "status": result["status"],
        "op_name": op_name,
        "shape_id": shape_id,
        "dtype": resolved_dtype,
        "device": device_id,
        "server": server_info.name,
        "timing": result.get("timing", {}),
        "returncode": result.get("returncode", -1),
        "impl_name": result.get("impl_name", ""),
        "created_at": result.get("created_at", ""),
    }
    if result["status"] != "success":
        summary["error"] = result.get("error", "")
    return _json(summary)


# ══════════════════════════════════════════════════════════════════════════════
# Token 统计工具
# ══════════════════════════════════════════════════════════════════════════════

_QODER_CACHE_ROOT = Path.home() / ".qoder" / "cache" / "projects"

# 字符级 token 估算的默认比率（混合代码/中英文场景）
_DEFAULT_CHARS_PER_TOKEN = 3.5


def _find_session_jsonl(session_id: Optional[str] = None) -> Optional[Path]:
    """定位会话 jsonl 文件。

    - 传 session_id: 在所有 project 目录下搜索 {session_id}/{session_id}.jsonl
    - 不传: 返回最近修改的 jsonl 文件
    """
    if not _QODER_CACHE_ROOT.exists():
        return None

    if session_id:
        pattern = str(_QODER_CACHE_ROOT / "*" / "conversation-history" / session_id / f"{session_id}.jsonl")
        matches = glob.glob(pattern)
        return Path(matches[0]) if matches else None

    # 找最近修改的 jsonl
    pattern = str(_QODER_CACHE_ROOT / "*" / "conversation-history" / "*" / "*.jsonl")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return Path(max(candidates, key=lambda p: os.path.getmtime(p)))


def _extract_text_from_content(content: Any) -> str:
    """从 message.content 中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(text)
                # tool_result 的 content 可能嵌套
                inner = block.get("content", "")
                if isinstance(inner, str) and inner:
                    parts.append(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and "text" in sub:
                            parts.append(sub["text"])
        return "\n".join(parts)
    return ""


def _estimate_tokens(char_count: int, chars_per_token: float) -> int:
    """字符级 token 估算。"""
    if char_count <= 0:
        return 0
    return max(1, math.ceil(char_count / chars_per_token))


def _estimate_cache_hit_rate(user_texts: list[str]) -> float:
    """基于 user 消息中重复前缀比例估算 cache hit rate。

    启发式: 多轮对话中 system prompt / 上下文会被重复发送,
    通过比较相邻 user 消息的公共前缀长度占比来近似缓存命中率。
    """
    if len(user_texts) < 2:
        return 0.0

    total_ratio = 0.0
    comparisons = 0
    for i in range(1, len(user_texts)):
        prev, curr = user_texts[i - 1], user_texts[i]
        if not curr:
            continue
        # 计算公共前缀长度
        prefix_len = 0
        for a, b in zip(prev, curr):
            if a != b:
                break
            prefix_len += 1
        ratio = prefix_len / len(curr) if curr else 0.0
        total_ratio += ratio
        comparisons += 1

    if comparisons == 0:
        return 0.0
    # 取平均并限制到 [0, 1]
    return min(1.0, total_ratio / comparisons)


@mcp.tool()
def get_session_token_usage(
    session_id: Optional[str] = None,
    chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
) -> str:
    """估算指定 Qoder 会话的 token 使用量，输出与 performance-report token_usage schema 对齐。

    通过读取 ~/.qoder/cache/projects/*/conversation-history/{session_id}/{session_id}.jsonl
    进行字符级 token 估算。不依赖外部 proxy，适合在 agent 生成报告阶段按需调用。

    Args:
        session_id: Qoder 会话 ID（如 da33f497）；不传则自动选取最近修改的会话
        chars_per_token: 字符到 token 的换算比率，默认 3.5（混合代码/中英文场景）
    """
    jsonl_path = _find_session_jsonl(session_id)
    if jsonl_path is None:
        if session_id:
            return _json({"error": f"未找到会话 {session_id} 的 jsonl 文件"})
        return _json({"error": "未找到任何会话历史文件"})

    actual_session_id = jsonl_path.stem

    input_chars = 0
    output_chars = 0
    user_texts: list[str] = []
    msg_count = {"user": 0, "assistant": 0, "other": 0}

    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = record.get("role", "")
                content = record.get("message", {}).get("content", [])
                text = _extract_text_from_content(content)

                if role == "user":
                    input_chars += len(text)
                    user_texts.append(text)
                    msg_count["user"] += 1
                elif role == "assistant":
                    output_chars += len(text)
                    msg_count["assistant"] += 1
                else:
                    msg_count["other"] += 1
    except OSError as exc:
        return _json({"error": f"读取 jsonl 失败: {exc}"})

    input_tokens = _estimate_tokens(input_chars, chars_per_token)
    output_tokens = _estimate_tokens(output_chars, chars_per_token)
    total_tokens = input_tokens + output_tokens
    cache_hit_rate = _estimate_cache_hit_rate(user_texts)

    return _json({
        "session_id": actual_session_id,
        "jsonl_path": str(jsonl_path),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_hit_rate": round(cache_hit_rate, 4),
        },
        "detail": {
            "input_chars": input_chars,
            "output_chars": output_chars,
            "chars_per_token": chars_per_token,
            "message_count": msg_count,
        },
        "note": "基于字符级估算，非精确 token 计数。cache_hit_rate 由多轮 user 消息公共前缀比例推算。",
    })


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
