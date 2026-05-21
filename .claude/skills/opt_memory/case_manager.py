"""Case 轨：ThoughtKernelCase 结构化记录。

管理 opt_memory/operators/{op}/cases/ 下的 JSON 文件。
使用 thoughts_case_schema.json 做校验。

每算子维护一个 index.json 加速 case_id → 文件路径查找：
{
    "case_id_1": {"path": "case_xxx.json", "thought_id": "...", "label": "...", ...},
    ...
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from .common import (
    ensure_dir,
    get_case_dir,
    get_case_index_path,
    get_schema,
)


# ── 校验 ──────────────────────────────────────────────────────────

def validate(data: dict) -> list[str]:
    """校验 Case JSON 是否符合 schema。

    Returns:
        list[str]: 错误信息列表，空列表表示通过。
    """
    schema = get_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = [e.message for e in validator.iter_errors(data)]
    return errors


# ── 索引 ──────────────────────────────────────────────────────────

def _load_index(operator: str, working_dir: str = ".") -> dict:
    """加载算子 Case 索引，不存在则返回空字典。"""
    idx_path = get_case_index_path(operator, working_dir)
    if idx_path.exists():
        with open(idx_path) as f:
            return json.load(f)
    return {}


def _save_index(operator: str, index: dict, working_dir: str = ".") -> None:
    """保存算子 Case 索引。"""
    idx_path = get_case_index_path(operator, working_dir)
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ── CRUD ──────────────────────────────────────────────────────────

def add_from_json(json_path: str, working_dir: str = ".") -> dict:
    """从 JSON 文件导入一条 Case。

    Returns:
        dict: {"case_id": str, "path": str} 或 {"error": str}。
    """
    with open(json_path) as f:
        data = json.load(f)

    errors = validate(data)
    if errors:
        return {"error": f"校验失败: {'; '.join(errors)}"}

    case_id = data["case_id"]
    operator = data["kernel"]["name"]
    case_dir = ensure_dir(get_case_dir(operator, working_dir))

    # 写入 Case 文件
    case_path = case_dir / f"{case_id}.json"
    with open(case_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 更新索引
    index = _load_index(operator, working_dir)
    index[case_id] = {
        "path": case_path.name,
        "thought_id": data["thought"]["id"],
        "thought_title": data["thought"]["title"],
        "label": data["effect"]["label"],
        "session_id": data["chain"]["session_id"],
        "step_index": data["chain"]["step_index"],
        "created_at": data.get("runtime", {}).get("created_at", ""),
    }
    _save_index(operator, index, working_dir)

    return {"case_id": case_id, "path": str(case_path)}


def append(
    *,
    operator: str,
    thought_id: str,
    thought_title: str,
    thought_desc: str,
    thought_reason: str,
    opt_layer: str,
    step_improvement: float,
    label: str,
    session_id: str,
    step_index: int,
    dtype: str,
    shape_signature: str,
    category: str = "other",
    compute_type: str = "unknown",
    target_bottleneck: str = "unknown",
    apply_scenario: str = "performance",
    apply_stage: str = "optimization",
    confidence: float = 0.8,
    thought_match: str = "high",
    device: str = "Ascend910B",
    agent: str = "kope",
    iteration: int | None = None,
    dsl_before: str = "",
    dsl_after: str = "",
    profiling_path: str = "",
    kernel_path: str = "",
    data_tags: str = "",
    task_tags: str = "",
    working_dir: str = ".",
) -> dict:
    """扁平参数模式：组装最小合法 Case，校验后写入。

    自动填充 case_id、prev_case_id、full_sequence_hint 和 created_at。
    适合 Agent 在迭代后快速记录。

    Returns:
        dict: {"case_id": str, "path": str} 或 {"error": str}。
    """
    # 生成 case_id: case_{operator}_sess_{YYYYMMDD}_{seq}_{step_index}
    # session_id 格式: sess_{op_name}_{YYYYMMDD}_{seq}
    # 从 session_id 中提取 YYYYMMDD 和 seq 组装 case_id
    parts = session_id.split("_")
    if len(parts) >= 4 and len(parts[2]) == 8:
        date_str = parts[2]
        seq = parts[3]
    else:
        # 兜底：用当前日期和 session_id
        from datetime import date
        date_str = date.today().strftime("%Y%m%d")
        seq = session_id.split("_")[-1] if "_" in session_id else "00"
    case_id = f"case_{operator}_sess_{date_str}_{seq}_{step_index}"

    # 查询上一步 case_id
    index = _load_index(operator, working_dir)
    prev_case_id = None
    full_sequence: list[str] = []
    if step_index > 0:
        # 从索引中查找同 session 的 step_index-1
        for cid, meta in index.items():
            if (
                meta.get("session_id") == session_id
                and meta.get("step_index") == step_index - 1
            ):
                prev_case_id = cid
                break

    # 构建 full_sequence_hint
    for cid, meta in sorted(index.items(), key=lambda x: x[1].get("step_index", 0)):
        if meta.get("session_id") == session_id:
            full_sequence.append(meta["thought_id"])
    full_sequence.append(thought_id)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    iter_num = iteration if iteration is not None else step_index

    # 解析 tags
    data_tags_list = [t.strip() for t in data_tags.split(",") if t.strip()]
    task_tags_list = [t.strip() for t in task_tags.split(",") if t.strip()]

    data: dict = {
        "schema_version": "0.2.0",
        "case_id": case_id,
        "kernel": {
            "name": operator,
            "category": category,
            "compute_type": compute_type,
            "path": kernel_path or "",
        },
        "thought": {
            "id": thought_id,
            "title": thought_title,
            "description": thought_desc,
            "reason": thought_reason,
            "opt_layer": opt_layer,
            "target_bottleneck": target_bottleneck,
            "apply_scenario": apply_scenario,
            "apply_stage": apply_stage,
        },
        "scene": {
            "shape_signature": shape_signature,
            "dtype": dtype,
            "data_tags": data_tags_list or [],
            "task_tags": task_tags_list or [],
        },
        "effect": {
            "label": label,
            "confidence": confidence,
            "thought_match": thought_match,
            "step_improvement": step_improvement,
        },
        "chain": {
            "session_id": session_id,
            "step_index": step_index,
            "prev_thought_id": "baseline" if step_index == 0 else "",
            "prev_case_id": prev_case_id,
            "full_sequence_hint": full_sequence,
        },
        "runtime": {
            "device": device,
            "agent": agent,
            "iteration": iter_num,
            "dsl_before": dsl_before,
            "dsl_after": dsl_after,
            "profiling_path": profiling_path,
            "created_at": now,
        },
    }

    # 设置 prev_thought_id
    if step_index > 0 and prev_case_id:
        prev_meta = index.get(prev_case_id, {})
        data["chain"]["prev_thought_id"] = prev_meta.get("thought_id", "")

    errors = validate(data)
    if errors:
        return {"error": f"校验失败: {'; '.join(errors)}"}

    case_dir = ensure_dir(get_case_dir(operator, working_dir))
    case_path = case_dir / f"{case_id}.json"
    with open(case_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 更新索引
    index[case_id] = {
        "path": case_path.name,
        "thought_id": thought_id,
        "thought_title": thought_title,
        "label": label,
        "session_id": session_id,
        "step_index": step_index,
        "created_at": now,
    }
    _save_index(operator, index, working_dir)

    return {"case_id": case_id, "path": str(case_path)}


def get(case_id: str, working_dir: str = ".") -> dict | None:
    """按 case_id 查找 Case。

    Returns:
        dict | None: Case 数据，不存在时返回 None。
    """
    # 搜索所有算子目录
    operators_root = Path(working_dir).resolve() / "opt_memory" / "operators"
    if not operators_root.exists():
        return None

    # 先查索引
    for op_dir in operators_root.iterdir():
        if not op_dir.is_dir():
            continue
        idx_path = op_dir / "cases" / "index.json"
        if not idx_path.exists():
            continue
        with open(idx_path) as f:
            index = json.load(f)
        if case_id in index:
            case_path = op_dir / "cases" / index[case_id]["path"]
            if case_path.exists():
                with open(case_path) as f:
                    return json.load(f)

    return None


def list_cases(
    *,
    operator: str | None = None,
    thought_id: str | None = None,
    label: str | None = None,
    session_id: str | None = None,
    opt_layer: str | None = None,
    working_dir: str = ".",
) -> list[dict]:
    """按条件过滤 Case 列表。

    Args:
        operator: 算子名称过滤。
        thought_id: 经验 ID 过滤。
        label: 效果标签过滤（effective / partial / ineffective / negative）。
        session_id: 会话 ID 过滤。
        opt_layer: 优化层级过滤。

    Returns:
        list[dict]: 匹配的 Case 摘要列表（不含完整 effect/metric_changes 细节）。
    """
    results: list[dict] = []
    operators_root = Path(working_dir).resolve() / "opt_memory" / "operators"
    if not operators_root.exists():
        return results

    search_dirs = (
        [operators_root / operator]
        if operator
        else [d for d in operators_root.iterdir() if d.is_dir()]
    )

    for op_dir in search_dirs:
        if not op_dir.exists() or not op_dir.is_dir():
            continue
        idx_path = op_dir / "cases" / "index.json"
        if not idx_path.exists():
            continue
        with open(idx_path) as f:
            index = json.load(f)

        for case_id, meta in index.items():
            if thought_id and meta.get("thought_id") != thought_id:
                continue
            if label and meta.get("label") != label:
                continue
            if session_id and meta.get("session_id") != session_id:
                continue
            if opt_layer:
                # 需要读 Case 文件取 opt_layer
                case_path = op_dir / "cases" / meta["path"]
                if case_path.exists():
                    with open(case_path) as f:
                        case_data = json.load(f)
                    if case_data.get("thought", {}).get("opt_layer") != opt_layer:
                        continue
                else:
                    continue

            results.append({
                "case_id": case_id,
                "operator": op_dir.name,
                "thought_id": meta.get("thought_id"),
                "thought_title": meta.get("thought_title"),
                "label": meta.get("label"),
                "session_id": meta.get("session_id"),
                "step_index": meta.get("step_index"),
                "created_at": meta.get("created_at"),
            })

    # 按 session_id + step_index 排序
    results.sort(key=lambda r: (r.get("session_id", ""), r.get("step_index", 0)))
    return results


def rebuild_chain(session_id: str, working_dir: str = ".") -> list[dict]:
    """按 step_index 排序重建一条优化链路。

    Returns:
        list[dict]: 有序 Case 完整数据列表。
    """
    chain: list[dict] = []
    operators_root = Path(working_dir).resolve() / "opt_memory" / "operators"
    if not operators_root.exists():
        return chain

    for op_dir in operators_root.iterdir():
        if not op_dir.is_dir():
            continue
        idx_path = op_dir / "cases" / "index.json"
        if not idx_path.exists():
            continue
        with open(idx_path) as f:
            index = json.load(f)

        for case_id, meta in index.items():
            if meta.get("session_id") == session_id:
                case_path = op_dir / "cases" / meta["path"]
                if case_path.exists():
                    with open(case_path) as f:
                        chain.append(json.load(f))

    chain.sort(key=lambda c: c.get("chain", {}).get("step_index", 0))
    return chain
