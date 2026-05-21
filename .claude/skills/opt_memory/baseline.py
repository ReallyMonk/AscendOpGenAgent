"""Baseline 管理：性能基线的记录、趋势展示和对比。

基线数据从两个来源聚合：
1. Journal 轨 — 迭代记录中的 tflops/utilization
2. Case 轨 — Case.effect 中的 step_improvement 和 metric_changes

存储于 opt_memory/baselines/index.json：
{
    "operator_name": [
        {"date": "YYYY-MM-DD", "tflops": ..., "utilization": ..., "source": "journal|case", "case_id": "..."},
        ...
    ]
}
"""

import json
from datetime import date
from pathlib import Path

from .common import get_opt_dir, get_operator_dir


BASELINE_INDEX = "index.json"


def _load_baselines(working_dir: str = ".") -> dict:
    """加载全局基线索引。"""
    baseline_dir = get_opt_dir(working_dir) / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    idx_path = baseline_dir / BASELINE_INDEX
    if idx_path.exists():
        with open(idx_path) as f:
            return json.load(f)
    return {}


def _save_baselines(data: dict, working_dir: str = ".") -> None:
    """保存全局基线索引。"""
    baseline_dir = get_opt_dir(working_dir) / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    idx_path = baseline_dir / BASELINE_INDEX
    with open(idx_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add(
    *,
    operator: str,
    tflops: float,
    utilization: float,
    working_dir: str = ".",
) -> dict:
    """记录一条性能基线。

    Returns:
        dict: {"operator": str, "date": str, "tflops": float, "utilization": float}
    """
    today = date.today().isoformat()
    entry = {
        "date": today,
        "tflops": tflops,
        "utilization": utilization,
        "source": "manual",
    }

    baselines = _load_baselines(working_dir)
    if operator not in baselines:
        baselines[operator] = []
    baselines[operator].append(entry)
    _save_baselines(baselines, working_dir)

    return {"operator": operator, **entry}


def show(*, operator: str, working_dir: str = ".") -> dict:
    """查看指定算子的历史基线趋势。

    Returns:
        dict: {"operator": str, "entries": list[dict]}
    """
    baselines = _load_baselines(working_dir)
    entries = baselines.get(operator, [])
    entries.sort(key=lambda e: e.get("date", ""))

    # 计算趋势
    if len(entries) >= 2:
        first = entries[0]
        last = entries[-1]
        entries.append({
            "_trend": "summary",
            "from_date": first["date"],
            "to_date": last["date"],
            "tflops_change": (
                (last["tflops"] - first["tflops"]) / first["tflops"] * 100
                if first["tflops"] > 0 else 0
            ),
            "utilization_change": last["utilization"] - first["utilization"],
        })

    return {"operator": operator, "entries": entries}


def compare(
    *,
    operator: str,
    date_a: str | None = None,
    date_b: str | None = None,
    working_dir: str = ".",
) -> dict:
    """对比两条基线。

    Args:
        date_a: 基准日期，不指定则取最早。
        date_b: 对比日期，不指定则取最新。

    Returns:
        dict: {"operator": str, "baseline": dict, "target": dict, "delta": dict}
    """
    baselines = _load_baselines(working_dir)
    entries = sorted(baselines.get(operator, []), key=lambda e: e.get("date", ""))

    if not entries:
        return {"error": f"算子 {operator} 无基线数据"}

    baseline_entry = entries[0]
    target_entry = entries[-1]

    if date_a:
        for e in entries:
            if e["date"] == date_a:
                baseline_entry = e
                break
    if date_b:
        for e in entries:
            if e["date"] == date_b:
                target_entry = e
                break

    delta_tflops = target_entry["tflops"] - baseline_entry["tflops"]
    delta_util = target_entry["utilization"] - baseline_entry["utilization"]

    return {
        "operator": operator,
        "baseline": baseline_entry,
        "target": target_entry,
        "delta": {
            "tflops": delta_tflops,
            "tflops_pct": (
                delta_tflops / baseline_entry["tflops"] * 100
                if baseline_entry["tflops"] > 0 else 0
            ),
            "utilization": delta_util,
        },
    }
