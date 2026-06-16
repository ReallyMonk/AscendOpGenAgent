"""Kope-Mem MCP Server — AscendC 算子优化记忆框架的 MCP 接口。

通过 Model Context Protocol (MCP) 将 kope-mem 的全部能力暴露为结构化工具，
供 AI Agent 在优化算子时直接调用，替代 bash heredoc 手动写文件的方式。

启动方式：
    uv run kope-mem-mcp                          # 默认 stdio 传输
    KOPE_MEM_WORKING_DIR=/path/to/project uv run kope-mem-mcp  # 指定工作目录

环境变量：
    KOPE_MEM_WORKING_DIR  工作目录，默认当前目录
"""

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .journal import add as _journal_add, show as _journal_show
from .case_manager import (
    append as _case_append,
    get as _case_get,
    list_cases as _list_cases,
    rebuild_chain as _rebuild_chain,
)
from .baseline import (
    add as _baseline_add,
    show as _baseline_show,
    compare as _baseline_compare,
)
from .compact import compact as _do_compact
from .search import search as _do_search
from .extract import add_pattern as _add_pattern, list_patterns as _list_patterns

# ── 工作目录 ──────────────────────────────────────────────────────
WORKING_DIR = os.environ.get("KOPE_MEM_WORKING_DIR", ".")

# ── MCP Server 实例 ───────────────────────────────────────────────
mcp = FastMCP(
    name="kope-mem",
    instructions=(
        "kope-mem 是 AscendC 算子优化记忆框架。"
        "提供四轨记忆能力：Journal（叙事日志）、Case（结构化经验）、"
        "Baseline（性能基线）、Pattern（可复用优化模式）。"
        "在算子优化开始前，建议先调用 search 查询历史经验；"
        "每次迭代后，用 journal_add 记录进展，用 case_append 沉淀经验。"
    ),
)


def _json(obj: dict | list) -> str:
    """序列化为 JSON 字符串。"""
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Journal 工具
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def journal_add(
    operator: str,
    iteration: int,
    strategy: str,
    change: str,
    tflops: float,
    utilization: float,
    status: str,
    insight: str = "",
) -> str:
    """追加一条迭代记录到当日 Journal（叙事日志轨）。

    在每次算子优化迭代完成后调用，记录本次迭代的策略、改动、性能和洞察。

    Args:
        operator: 算子名称，如 "matmul"、"layernorm"
        iteration: 迭代序号（从 1 开始）
        strategy: 本次使用的优化策略名称
        change: 关键代码变更描述
        tflops: 本次测量的 TFLOPS 值
        utilization: 利用率百分比（0-100）
        status: 状态，取值为 "pass"、"fail"、"partial"
        insight: 一句话洞察（为什么有效或无效）
    """
    result = _journal_add(
        operator=operator,
        iteration=iteration,
        strategy=strategy,
        change=change,
        tflops=tflops,
        utilization=utilization,
        status=status,
        insight=insight,
        working_dir=WORKING_DIR,
    )
    return _json(result)


@mcp.tool()
def journal_show(
    operator: str,
    date: Optional[str] = None,
    latest: int = 10,
) -> str:
    """查看指定算子的 Journal 历史日志。

    Args:
        operator: 算子名称
        date: 指定日期（格式 YYYY-MM-DD），不指定则返回最近条目
        latest: 返回最近 N 条记录（date 未指定时生效）
    """
    result = _journal_show(operator=operator, date_filter=date, latest=latest, working_dir=WORKING_DIR)
    return _json(result)


# ═══════════════════════════════════════════════════════════════════
# Case 工具
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def case_append(
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
    iteration: Optional[int] = None,
    dsl_before: str = "",
    dsl_after: str = "",
    profiling_path: str = "",
    kernel_path: str = "",
    data_tags: str = "",
    task_tags: str = "",
) -> str:
    """追加一条结构化优化经验（Case 轨）。

    记录一次优化操作的完整上下文：做了什么、为什么、效果如何。
    自动更新索引，使后续的 search/case_list 能检索到该经验。

    opt_layer 取值：core_partition | tiling | pipeline | instruction | none
    label 取值：effective | partial | ineffective | negative
    dtype 取值：float16 | float32 | bfloat16 | int8 | int32

    Args:
        operator: 算子名称
        thought_id: 优化经验 ID（自定义标识）
        thought_title: 经验标题
        thought_desc: 做了什么（描述优化操作）
        thought_reason: 为什么这样做（优化动机）
        opt_layer: 优化层级
        step_improvement: 本步性能提升倍数（1.0 表示无变化）
        label: 效果标签
        session_id: 会话 ID（格式：sess_{算子名}_{YYYYMMDD}_{序号}）
        step_index: 本步在链路中的序号（从 0 开始）
        dtype: 数据类型
        shape_signature: 输入 shape 摘要描述
        category: 算子类别（activation/normalization/matmul/conv/elementwise/reduction/attention/other）
        compute_type: 计算类型（memory_bound/compute_bound/pipeline_bound/cube_underutil/unknown）
        target_bottleneck: 目标瓶颈描述
        apply_scenario: 应用场景（performance/correctness/both）
        apply_stage: 应用阶段（optimization/design/translation）
        confidence: 置信度（0-1）
        thought_match: 经验匹配度（high/medium/low）
        device: 设备名称
        agent: 使用的 agent 名称
        iteration: 迭代号（不填则用 step_index）
        dsl_before: 优化前 DSL 代码片段
        dsl_after: 优化后 DSL 代码片段
        profiling_path: profiling 文件路径
        kernel_path: kernel 文件路径
        data_tags: 数据标签（逗号分隔）
        task_tags: 任务标签（逗号分隔）
    """
    result = _case_append(
        operator=operator,
        thought_id=thought_id,
        thought_title=thought_title,
        thought_desc=thought_desc,
        thought_reason=thought_reason,
        opt_layer=opt_layer,
        step_improvement=step_improvement,
        label=label,
        session_id=session_id,
        step_index=step_index,
        dtype=dtype,
        shape_signature=shape_signature,
        category=category,
        compute_type=compute_type,
        target_bottleneck=target_bottleneck,
        apply_scenario=apply_scenario,
        apply_stage=apply_stage,
        confidence=confidence,
        thought_match=thought_match,
        device=device,
        agent=agent,
        iteration=iteration,
        dsl_before=dsl_before,
        dsl_after=dsl_after,
        profiling_path=profiling_path,
        kernel_path=kernel_path,
        data_tags=data_tags,
        task_tags=task_tags,
        working_dir=WORKING_DIR,
    )
    return _json(result)


@mcp.tool()
def case_show(case_id: str) -> str:
    """查看指定 Case 的完整详情。

    Args:
        case_id: Case 唯一标识符
    """
    result = _case_get(case_id, working_dir=WORKING_DIR)
    if result is None:
        return _json({"error": f"Case {case_id} 不存在"})
    return _json(result)


@mcp.tool()
def case_list(
    operator: Optional[str] = None,
    thought_id: Optional[str] = None,
    label: Optional[str] = None,
    session_id: Optional[str] = None,
    opt_layer: Optional[str] = None,
) -> str:
    """按条件过滤查询 Case 列表。

    所有过滤条件均可选，不填则不过滤。

    Args:
        operator: 按算子名过滤
        thought_id: 按经验 ID 过滤
        label: 按效果标签过滤（effective/partial/ineffective/negative）
        session_id: 按会话 ID 过滤
        opt_layer: 按优化层级过滤（core_partition/tiling/pipeline/instruction/none）
    """
    results = _list_cases(
        operator=operator,
        thought_id=thought_id,
        label=label,
        session_id=session_id,
        opt_layer=opt_layer,
        working_dir=WORKING_DIR,
    )
    if not results:
        return _json({"count": 0, "results": []})
    return _json({"count": len(results), "results": results})


@mcp.tool()
def case_chain(session_id: str) -> str:
    """按 session_id 重建完整优化链路。

    将同一 session 下的所有 Case 按 step_index 排序，
    展示从 baseline 到最终优化的完整路径。

    Args:
        session_id: 会话 ID
    """
    chain = _rebuild_chain(session_id, working_dir=WORKING_DIR)
    if not chain:
        return _json({"error": f"未找到 session {session_id} 的链路", "steps": []})
    summary = []
    for step in chain:
        c = step.get("chain", {})
        t = step.get("thought", {})
        e = step.get("effect", {})
        summary.append({
            "step_index": c.get("step_index"),
            "title": t.get("title"),
            "label": e.get("label"),
            "step_improvement": e.get("step_improvement"),
        })
    return _json({"session_id": session_id, "total_steps": len(chain), "steps": summary})


# ═══════════════════════════════════════════════════════════════════
# Baseline 工具
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def baseline_add(operator: str, tflops: float, utilization: float) -> str:
    """记录一条性能基线。

    在算子首次测量或重要里程碑时调用，用于后续对比和趋势分析。

    Args:
        operator: 算子名称
        tflops: TFLOPS 值
        utilization: 利用率百分比（0-100）
    """
    result = _baseline_add(operator=operator, tflops=tflops, utilization=utilization, working_dir=WORKING_DIR)
    return _json(result)


@mcp.tool()
def baseline_show(operator: str) -> str:
    """查看指定算子的历史基线趋势。

    返回所有已记录基线，并自动计算从首条到最新条的趋势变化。

    Args:
        operator: 算子名称
    """
    result = _baseline_show(operator=operator, working_dir=WORKING_DIR)
    return _json(result)


@mcp.tool()
def baseline_compare(
    operator: str,
    date_a: Optional[str] = None,
    date_b: Optional[str] = None,
) -> str:
    """对比两条性能基线。

    不指定日期时，自动对比最早基线与最新基线。

    Args:
        operator: 算子名称
        date_a: 基准日期（格式 YYYY-MM-DD），不指定取最早
        date_b: 对比日期（格式 YYYY-MM-DD），不指定取最新
    """
    result = _baseline_compare(operator=operator, date_a=date_a, date_b=date_b, working_dir=WORKING_DIR)
    return _json(result)


# ═══════════════════════════════════════════════════════════════════
# Pattern 工具
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def pattern_add(
    name: str,
    core_idea: str,
    operator_types: str = "",
    shape_range: str = "",
    hardware_constraints: str = "",
    code_skeleton: str = "",
    limitations: str = "",
    source_operator: str = "",
    source_date: str = "",
    related_cases: str = "",
) -> str:
    """添加一条可复用的优化模式（Pattern 轨）。

    当发现某个优化策略具有跨算子复用价值时调用。

    Args:
        name: 模式名称（英文短横线格式，如 "multi-core-partition"）
        core_idea: 核心优化思路描述
        operator_types: 适用的算子类型
        shape_range: 适用的 shape 范围
        hardware_constraints: 硬件约束说明
        code_skeleton: 代码模板（Python 代码片段）
        limitations: 已知限制
        source_operator: 首次发现该模式的算子
        source_date: 发现日期
        related_cases: 相关 Case ID（逗号分隔）
    """
    result = _add_pattern(
        name=name,
        core_idea=core_idea,
        operator_types=operator_types,
        shape_range=shape_range,
        hardware_constraints=hardware_constraints,
        code_skeleton=code_skeleton,
        limitations=limitations,
        source_operator=source_operator,
        source_date=source_date,
        related_cases=related_cases,
        working_dir=WORKING_DIR,
    )
    return _json(result)


@mcp.tool()
def pattern_list() -> str:
    """列出所有已有的优化模式。"""
    result = _list_patterns(working_dir=WORKING_DIR)
    return _json(result)


# ═══════════════════════════════════════════════════════════════════
# Search 工具
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def search(
    query: str,
    operator: Optional[str] = None,
    thought_id: Optional[str] = None,
    label: Optional[str] = None,
    opt_layer: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """语义检索历史优化经验。

    两阶段检索：先结构化精确过滤，再语义模糊匹配（需 ReMe）。
    建议在开始优化新算子前调用，查询相关历史经验避免重复试错。

    Args:
        query: 自然语言查询（如 "tiling 优化 matmul 的策略"）
        operator: 按算子名过滤
        thought_id: 按经验 ID 过滤
        label: 按效果标签过滤
        opt_layer: 按优化层级过滤
        max_results: 最大返回条数
    """
    result = _do_search(
        query=query,
        operator=operator,
        thought_id=thought_id,
        label=label,
        opt_layer=opt_layer,
        max_results=max_results,
        working_dir=WORKING_DIR,
    )
    return _json(result)


# ═══════════════════════════════════════════════════════════════════
# Compact 工具
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def compact(
    operator: str,
    dialog: Optional[str] = None,
    previous_summary: str = "",
) -> str:
    """压缩过长的优化对话上下文（需 ReMe）。

    当对话历史过长导致上下文紧张时调用，
    将对话压缩为摘要并持久化到算子目录。

    Args:
        operator: 算子名称
        dialog: 对话 JSONL 文件路径，不指定则自动取最近的文件
        previous_summary: 之前的摘要（增量压缩时注入）
    """
    result = _do_compact(
        operator=operator,
        dialog=dialog,
        previous_summary=previous_summary,
        working_dir=WORKING_DIR,
    )
    return _json(result)


# ═══════════════════════════════════════════════════════════════════
# 初始化 Resource（供 Agent 快速了解当前状态）
# ═══════════════════════════════════════════════════════════════════


@mcp.resource("kope-mem://status")
def status_resource() -> str:
    """返回 kope-mem 当前状态概览。"""
    from pathlib import Path
    opt_dir = Path(WORKING_DIR).resolve() / "opt_memory"
    operators_dir = opt_dir / "operators"

    operators = []
    if operators_dir.exists():
        for d in sorted(operators_dir.iterdir()):
            if d.is_dir():
                idx = d / "cases" / "index.json"
                case_count = 0
                if idx.exists():
                    try:
                        with open(idx) as f:
                            case_count = len(json.load(f))
                    except Exception:
                        pass
                operators.append({"name": d.name, "case_count": case_count})

    patterns_dir = opt_dir / "patterns"
    pattern_count = 0
    if patterns_dir.exists():
        pattern_count = len(list(patterns_dir.glob("*.md")))

    return _json({
        "working_dir": str(Path(WORKING_DIR).resolve()),
        "opt_memory_dir": str(opt_dir),
        "operators": operators,
        "pattern_count": pattern_count,
    })


# ── 入口 ──────────────────────────────────────────────────────────


def main():
    """MCP Server 启动入口（stdio 传输）。"""
    mcp.run()


if __name__ == "__main__":
    main()
