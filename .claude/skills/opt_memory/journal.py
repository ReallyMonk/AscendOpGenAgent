"""Journal 轨：算子优化叙事日志。

管理 opt_memory/operators/{op}/YYYY-MM-DD.md 文件。
与 Case 轨并列，独立演进。

Journal 格式：

## Iter {N}: {策略名}
- **改动**：{关键代码变更}
- **性能**：{TFLOPS} TFLOPS，利用率 {X}%
- **状态**：✅通过 / ❌回退 / ⚠️部分改善
- **洞察**：{为什么有效/无效，一句话}
"""

from datetime import date
from pathlib import Path

from .common import ensure_dir, get_operator_dir

STATUS_ICONS = {
    "pass": "✅通过",
    "fail": "❌回退",
    "partial": "⚠️部分改善",
}


def add(
    *,
    operator: str,
    iteration: int,
    strategy: str,
    change: str,
    tflops: float,
    utilization: float,
    status: str,
    insight: str = "",
    working_dir: str = ".",
) -> dict:
    """追加一条迭代记录到当日 Journal。

    Returns:
        dict: {"path": str, "date": str} 写入的文件路径和日期。
    """
    status_icon = STATUS_ICONS.get(status, status)
    today = date.today().isoformat()

    entry = (
        f"## Iter {iteration}: {strategy}\n\n"
        f"- **改动**：{change}\n"
        f"- **性能**：{tflops} TFLOPS，利用率 {utilization}%\n"
        f"- **状态**：{status_icon}\n"
        f"- **洞察**：{insight or '—'}\n\n"
        "---\n\n"
    )

    op_dir = ensure_dir(get_operator_dir(operator, working_dir))
    journal_path = op_dir / f"{today}.md"

    with open(journal_path, "a") as f:
        f.write(entry)

    return {"path": str(journal_path), "date": today}


def show(
    *,
    operator: str,
    date_filter: str | None = None,
    latest: int = 10,
    working_dir: str = ".",
) -> dict:
    """查看指定算子的 Journal 内容。

    Args:
        operator: 算子名称。
        date_filter: 指定日期（YYYY-MM-DD），不指定则返回最近的条目。
        latest: 返回最近 N 条迭代记录（仅在 date_filter 为 None 时生效）。

    Returns:
        dict: {"entries": list[str], "source": str} 或 {"error": str}。
    """
    op_dir = get_operator_dir(operator, working_dir)
    if not op_dir.exists():
        return {"entries": [], "source": str(op_dir), "error": "算子目录不存在"}

    if date_filter:
        journal_path = op_dir / f"{date_filter}.md"
        if not journal_path.exists():
            return {"entries": [], "source": str(journal_path), "error": "该日期无日志"}
        content = journal_path.read_text()
        return {"entries": [content], "source": str(journal_path)}

    # 收集最近的条目
    md_files = sorted(op_dir.glob("*.md"), reverse=True)
    entries: list[str] = []
    count = 0
    for md_file in md_files:
        lines = md_file.read_text().strip().split("---")
        for block in reversed(lines):
            block = block.strip()
            if block.startswith("## Iter"):
                entries.append(block)
                count += 1
                if count >= latest:
                    break
        if count >= latest:
            break

    return {"entries": entries, "source": str(op_dir)}
