"""Qoder CLI 会话 Token 统计 MCP Server

读取 Qoder CLI 会话历史（JSONL），估算 token 用量（无需 tiktoken，基于字符比例估算）。

工具列表：
    list_sessions          列出所有会话（按时间倒序）
    get_session_usage      获取指定会话的 token 用量统计
    get_latest_usage       获取最近一次会话的 token 用量统计

启动方式：
    python utils/session_stats_server.py

环境变量：
    QODER_CACHE_DIR  Qoder 缓存目录（默认 ~/.qoder/cache）
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# ── 配置 ──────────────────────────────────────────────────────────────────────

QODER_CACHE_DIR = Path(os.environ.get("QODER_CACHE_DIR", Path.home() / ".qoder" / "cache"))
HISTORY_BASE = QODER_CACHE_DIR / "projects"

# ── Token 估算 ────────────────────────────────────────────────────────────────

# 经验值：
# - CJK 字符：约 1.5 token/字
# - ASCII/拉丁字母代码：约 0.25 token/字符（4 字符 ≈ 1 token）
# - 其他 Unicode：约 0.5 token/字符
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x3000, 0x303F),    # CJK Symbols
    (0xFF00, 0xFFEF),    # Fullwidth Forms
    (0xAC00, 0xD7AF),    # Hangul
    (0x3040, 0x30FF),    # Hiragana + Katakana
]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数，无需外部依赖。"""
    if not text:
        return 0
    cjk_chars = 0
    ascii_chars = 0
    other_chars = 0
    for ch in text:
        if _is_cjk(ch):
            cjk_chars += 1
        elif ord(ch) < 128:
            ascii_chars += 1
        else:
            other_chars += 1
    # CJK: 1.5 token/字, ASCII: 0.25 token/字符, 其他: 0.5 token/字符
    return int(cjk_chars * 1.5 + ascii_chars * 0.25 + other_chars * 0.5) + 1


def _extract_text(content: Any) -> str:
    """从 message.content（可能是 list 或 str）提取全部文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content else ""


# ── 会话文件解析 ──────────────────────────────────────────────────────────────


def _find_session_files() -> List[Dict[str, Any]]:
    """扫描所有会话历史 JSONL 文件，返回元数据列表（按修改时间倒序）。"""
    sessions: List[Dict[str, Any]] = []
    if not HISTORY_BASE.exists():
        return sessions

    for project_dir in HISTORY_BASE.iterdir():
        if not project_dir.is_dir():
            continue
        project_id = project_dir.name
        history_dir = project_dir / "conversation-history"
        if not history_dir.exists():
            continue
        for session_dir in history_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            jsonl_path = session_dir / f"{session_id}.jsonl"
            if not jsonl_path.exists():
                continue
            try:
                mtime = jsonl_path.stat().st_mtime
                line_count = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
                sessions.append({
                    "session_id": session_id,
                    "project_id": project_id,
                    "path": str(jsonl_path),
                    "message_count": line_count,
                    "modified_at": datetime.fromtimestamp(mtime).isoformat(),
                    "mtime": mtime,
                })
            except Exception:
                continue

    sessions.sort(key=lambda x: x["mtime"], reverse=True)
    return sessions


def _compute_usage(jsonl_path: Path) -> Dict[str, Any]:
    """解析单个 JSONL 会话文件，统计 token 用量。"""
    input_tokens = 0
    output_tokens = 0
    message_count = 0
    user_messages = 0
    assistant_messages = 0
    tool_calls = 0
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = entry.get("role", "")
            msg = entry.get("message", {})
            content = msg.get("content", "")
            text = _extract_text(content)
            tokens = estimate_tokens(text)

            message_count += 1

            if role == "user":
                input_tokens += tokens
                user_messages += 1
            elif role == "assistant":
                output_tokens += tokens
                assistant_messages += 1
                # 统计 tool call 次数（assistant 消息里的 tool_use 块）
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in ("tool_use", "function_call"):
                            tool_calls += 1
            elif role == "tool":
                # tool 返回内容也算输入（下一轮的 context）
                input_tokens += tokens

            # 尝试提取时间戳（如果存在）
            ts = entry.get("timestamp") or entry.get("created_at")
            if ts:
                if first_ts is None:
                    first_ts = str(ts)
                last_ts = str(ts)

    total_tokens = input_tokens + output_tokens
    # 估算缓存命中率：无法从字符数据推算，标记为不可用
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "message_count": message_count,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "tool_calls": tool_calls,
        "cache_hit_rate": None,
        "first_message_at": first_ts,
        "last_message_at": last_ts,
        "estimation_note": "基于字符比例估算（CJK×1.5, ASCII×0.25, 其他×0.5），非精确值",
    }


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="session-stats",
    instructions=(
        "Qoder CLI 会话 Token 统计工具。"
        "用于在生成性能报告时获取本次优化会话的 token 用量（估算值）。"
        "在调用 performance-report skill 前，用 get_latest_usage 或 get_session_usage 获取统计数据，"
        "将结果填入 perf_data.json 的 global_meta.token_usage 字段。"
    ),
)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


@mcp.tool()
def list_sessions(limit: int = 20) -> str:
    """列出最近 N 个 Qoder CLI 会话（按修改时间倒序）。

    Args:
        limit: 最多返回多少条，默认 20
    """
    sessions = _find_session_files()[:limit]
    result = []
    for s in sessions:
        result.append({
            "session_id": s["session_id"],
            "project_id": s["project_id"],
            "message_count": s["message_count"],
            "modified_at": s["modified_at"],
        })
    return _json({"total_found": len(_find_session_files()), "sessions": result})


@mcp.tool()
def get_session_usage(session_id: str) -> str:
    """获取指定会话的 token 用量估算值。

    Args:
        session_id: 会话 ID（32 位 hex，可从 list_sessions 获取）
    """
    # 在所有 project 目录下搜索该 session_id
    if not HISTORY_BASE.exists():
        return _json({"error": f"Qoder 缓存目录不存在: {HISTORY_BASE}"})

    for project_dir in HISTORY_BASE.iterdir():
        if not project_dir.is_dir():
            continue
        jsonl_path = project_dir / "conversation-history" / session_id / f"{session_id}.jsonl"
        if jsonl_path.exists():
            usage = _compute_usage(jsonl_path)
            usage["session_id"] = session_id
            return _json(usage)

    return _json({"error": f"会话 '{session_id}' 不存在", "hint": "使用 list_sessions 查看可用会话"})


@mcp.tool()
def get_latest_usage() -> str:
    """获取最近一次 Qoder CLI 会话的 token 用量估算值。

    通常用于在生成性能报告时快速拿到本次优化会话的 token 统计。
    """
    sessions = _find_session_files()
    if not sessions:
        return _json({"error": "没有找到任何会话历史"})

    latest = sessions[0]
    usage = _compute_usage(Path(latest["path"]))
    usage["session_id"] = latest["session_id"]
    usage["project_id"] = latest["project_id"]
    usage["modified_at"] = latest["modified_at"]
    return _json(usage)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
