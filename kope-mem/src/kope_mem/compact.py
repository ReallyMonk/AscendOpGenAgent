"""ReMe 封装：对话上下文压缩。

调用 ReMeLight.compact_memory，使用 AscendC 优化定制摘要模板。
"""

from pathlib import Path

from .common import ensure_dir, get_reme, get_opt_dir


def compact(
    *,
    operator: str,
    dialog: str | None = None,
    previous_summary: str = "",
    working_dir: str = ".",
    max_input_length: int = 128000,
    compact_ratio: float = 0.6,
) -> dict:
    """压缩优化对话上下文。

    Args:
        operator: 算子名称，用于定位对话文件和写入压缩结果。
        dialog: 对话 JSONL 路径，不指定则从 opt_memory/dialog/ 取最近的。
        previous_summary: 之前的摘要（增量压缩时注入）。
        max_input_length: 最大输入 token 数。
        compact_ratio: 压缩比（保留最近对话的比例）。

    Returns:
        dict: {"summary": str, "path": str} 或 {"error": str}
    """
    reme = get_reme(working_dir)
    opt_dir = get_opt_dir(working_dir)

    # 定位对话文件
    if dialog:
        dialog_path = Path(dialog)
    else:
        dialog_dir = opt_dir / "dialog"
        if not dialog_dir.exists():
            return {"error": "暂无对话文件可压缩"}
        jsonl_files = sorted(dialog_dir.glob("*.jsonl"), reverse=True)
        if not jsonl_files:
            return {"error": "暂无对话文件可压缩"}
        dialog_path = jsonl_files[0]

    if not dialog_path.exists():
        return {"error": f"对话文件不存在: {dialog_path}"}

    # 读取对话消息
    import json
    messages = []
    with open(dialog_path) as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    if not messages:
        return {"error": "对话文件为空"}

    try:
        summary = reme.compact_memory(
            messages=messages,
            previous_summary=previous_summary,
            max_input_length=max_input_length,
            compact_ratio=compact_ratio,
            language="zh",
        )

        # 持久化压缩结果到算子目录
        from datetime import date
        op_dir = ensure_dir(opt_dir / "operators" / operator)
        today = date.today().isoformat()
        summary_path = op_dir / f"compacted_{today}.md"
        with open(summary_path, "w") as f:
            f.write(f"# 优化摘要 — {operator} {today}\n\n{summary}")

        return {"summary": summary, "path": str(summary_path)}

    except Exception as e:
        return {"error": f"压缩失败: {e}"}
