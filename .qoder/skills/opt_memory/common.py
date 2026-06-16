"""Kope-Mem 公共工具：ReMeLight 初始化、路径解析、schema 加载。

提供：
1. 懒加载 ReMeLight 实例 — compact / search / summary 命令共用
2. opt_memory/ 下各目录的路径工具函数
3. ThoughtKernelCase schema 缓存
"""

import json
from pathlib import Path
from typing import Optional


# ── ReMeLight 懒加载 ─────────────────────────────────────────────

_reme: Optional[object] = None  # ReMeLight | None


def get_reme(working_dir: str = "."):
    """获取 ReMeLight 单例，首次调用时初始化。

    ReMe 需要 LLM 做摘要生成和 embedding，配置从环境变量读取
    （LLM_API_KEY, EMBEDDING_API_KEY 等）。
    如果 reme-ai 未安装，提示用户安装 optional 依赖。
    """
    global _reme
    if _reme is not None:
        return _reme
    try:
        from reme_light import ReMeLight
    except ImportError:
        raise ImportError(
            "ReMe 未安装。请运行: uv sync --extra reme"
        )
    _reme = ReMeLight(
        working_dir=working_dir,
        memory_dir="opt_memory",
        enable_load_env=True,
    )
    return _reme


# ── Schema 缓存 ───────────────────────────────────────────────────

_schema: Optional[dict] = None
_schema_path: Optional[Path] = None


def get_schema() -> dict:
    """加载 thoughts_case_schema.json。

    从 kope-mem 项目根目录查找 schema 文件并缓存。
    """
    global _schema, _schema_path
    if _schema is not None:
        return _schema

    # 向上查找 schema 文件
    current = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = current / "thoughts_case_schema.json"
        if candidate.exists():
            _schema_path = candidate
            with open(candidate) as f:
                _schema = json.load(f)
            return _schema
        if (current / "pyproject.toml").exists():
            break
        current = current.parent

    raise FileNotFoundError(
        "找不到 thoughts_case_schema.json。"
        "请确保文件在 kope-mem 项目根目录。"
    )


# ── 路径工具 ──────────────────────────────────────────────────────

def get_opt_dir(working_dir: str = ".") -> Path:
    """返回 opt_memory/ 根目录路径。"""
    return Path(working_dir).resolve() / "opt_memory"


def get_operator_dir(operator: str, working_dir: str = ".") -> Path:
    """返回 opt_memory/operators/{operator}/ 目录路径。"""
    return get_opt_dir(working_dir) / "operators" / operator


def get_case_dir(operator: str, working_dir: str = ".") -> Path:
    """返回 opt_memory/operators/{operator}/cases/ 目录路径。"""
    return get_operator_dir(operator, working_dir) / "cases"


def get_case_index_path(operator: str, working_dir: str = ".") -> Path:
    """返回 opt_memory/operators/{operator}/cases/index.json 路径。"""
    return get_case_dir(operator, working_dir) / "index.json"


def ensure_dir(path: Path) -> Path:
    """确保目录存在，返回该目录。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
