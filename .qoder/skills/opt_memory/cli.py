"""Kope-Mem CLI — AscendC 算子优化记忆框架。

统一入口，通过 Cyclopts 命令树暴露 Journal、Case、Baseline、
Compact、Search、Watch、Pattern 等子命令。

用法：
    uv run kope-mem --help
    uv run kope-mem journal add --operator matmul ...
    uv run kope-mem case list --operator matmul --label effective
"""

import json
import sys
from pathlib import Path
from typing import Optional

from cyclopts import App, Parameter

from .journal import add as journal_add, show as journal_show
from .case_manager import (
    add_from_json,
    append,
    get,
    list_cases,
    rebuild_chain,
    validate,
)
from .baseline import add as baseline_add, show as baseline_show, compare as baseline_compare
from .compact import compact as do_compact
from .search import search as do_search
from .watch import watch as do_watch
from .extract import add_pattern, list_patterns

# ── 主 App ────────────────────────────────────────────────────────

app = App(
    name="kope-mem",
    help="AscendC 算子优化记忆框架 CLI",
)

# ── 子 App ────────────────────────────────────────────────────────

journal_app = App(name="journal", help="Journal 轨：Markdown 叙事日志")
case_app = App(name="case", help="Case 轨：ThoughtKernelCase 结构化记录")
baseline_app = App(name="baseline", help="性能基线管理")
pattern_app = App(name="pattern", help="优化模式库管理")

app.command(journal_app)
app.command(case_app)
app.command(baseline_app)
app.command(pattern_app)


# ═══════════════════════════════════════════════════════════════════
# Journal 子命令
# ═══════════════════════════════════════════════════════════════════

@journal_app.command
def journal_add_cmd(
    operator: str,
    iteration: int,
    strategy: str,
    change: str,
    tflops: float,
    utilization: float,
    status: str,
    *,
    insight: str = "",
):
    """追加一条迭代记录到当日 Journal。"""
    result = journal_add(
        operator=operator,
        iteration=iteration,
        strategy=strategy,
        change=change,
        tflops=tflops,
        utilization=utilization,
        status=status,
        insight=insight,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


@journal_app.command
def journal_show_cmd(
    operator: str,
    *,
    date: str | None = None,
    latest: int = 10,
):
    """查看算子 Journal 日志。"""
    result = journal_show(operator=operator, date_filter=date, latest=latest)
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    if not result["entries"]:
        print("暂无日志记录")
        return
    for entry in result["entries"]:
        print(entry)
        print()


# ═══════════════════════════════════════════════════════════════════
# Case 子命令
# ═══════════════════════════════════════════════════════════════════

@case_app.command
def case_add_cmd(
    json_path: Path,
):
    """从 JSON 文件导入一条 ThoughtKernelCase。"""
    result = add_from_json(str(json_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


@case_app.command
def case_append_cmd(
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
    *,
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
):
    """扁平参数快速追加一条 ThoughtKernelCase。"""
    result = append(
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


@case_app.command
def case_show_cmd(
    case_id: str,
):
    """查看指定 Case 详情。"""
    result = get(case_id)
    if result is None:
        print(f"错误: Case {case_id} 不存在")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


@case_app.command
def case_list_cmd(
    *,
    operator: str | None = None,
    thought_id: str | None = None,
    label: str | None = None,
    session_id: str | None = None,
    opt_layer: str | None = None,
):
    """按条件过滤 Case 列表。"""
    results = list_cases(
        operator=operator,
        thought_id=thought_id,
        label=label,
        session_id=session_id,
        opt_layer=opt_layer,
    )
    if not results:
        print("无匹配 Case")
        return
    print(f"共 {len(results)} 条 Case:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


@case_app.command
def case_validate_cmd(
    json_path: Path,
):
    """校验 Case JSON 是否符合 schema。"""
    with open(json_path) as f:
        data = json.load(f)
    errors = validate(data)
    if errors:
        print(f"校验失败 ({len(errors)} 个错误):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ 校验通过")


@case_app.command
def case_chain_cmd(
    session_id: str,
):
    """按 session 重建优化链路。"""
    chain = rebuild_chain(session_id)
    if not chain:
        print(f"错误: 未找到 session {session_id} 的链路")
        return
    print(f"链路 {session_id} 共 {len(chain)} 步:")
    for step in chain:
        c = step.get("chain", {})
        t = step.get("thought", {})
        e = step.get("effect", {})
        print(f"  Step {c.get('step_index')}: "
              f"{t.get('title', '?')} "
              f"({e.get('label', '?')}, "
              f"×{e.get('step_improvement', 1.0):.2f})")


# ═══════════════════════════════════════════════════════════════════
# Baseline 子命令
# ═══════════════════════════════════════════════════════════════════

@baseline_app.command
def baseline_add_cmd(
    operator: str,
    tflops: float,
    utilization: float,
):
    """记录一条性能基线。"""
    result = baseline_add(operator=operator, tflops=tflops, utilization=utilization)
    print(json.dumps(result, ensure_ascii=False, indent=2))


@baseline_app.command
def baseline_show_cmd(
    operator: str,
):
    """查看历史基线趋势。"""
    result = baseline_show(operator=operator)
    print(json.dumps(result, ensure_ascii=False, indent=2))


@baseline_app.command
def baseline_compare_cmd(
    operator: str,
    *,
    date_a: str | None = None,
    date_b: str | None = None,
):
    """对比两条基线。"""
    result = baseline_compare(
        operator=operator, date_a=date_a, date_b=date_b
    )
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    print(f"算子: {result['operator']}")
    print(f"基线 ({result['baseline']['date']}): "
          f"{result['baseline']['tflops']} TFLOPS, "
          f"{result['baseline']['utilization']}%")
    print(f"对比 ({result['target']['date']}): "
          f"{result['target']['tflops']} TFLOPS, "
          f"{result['target']['utilization']}%")
    d = result["delta"]
    direction = "↑" if d["tflops"] > 0 else "↓"
    print(f"变化: {direction}{abs(d['tflops_pct']):.1f}% TFLOPS, "
          f"{d['utilization']:+.1f}% 利用率")


# ═══════════════════════════════════════════════════════════════════
# Pattern 子命令
# ═══════════════════════════════════════════════════════════════════

@pattern_app.command
def pattern_add_cmd(
    name: str,
    core_idea: str,
    *,
    operator_types: str = "",
    shape_range: str = "",
    hardware_constraints: str = "",
    code_skeleton: str = "",
    limitations: str = "",
    source_operator: str = "",
    source_date: str = "",
    related_cases: str = "",
):
    """手动添加一条优化模式。"""
    result = add_pattern(
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


@pattern_app.command
def pattern_list_cmd():
    """列出已有优化模式。"""
    result = list_patterns()
    if not result["patterns"]:
        print("暂无模式")
        return
    print(f"共 {len(result['patterns'])} 个模式:")
    for p in result["patterns"]:
        print(f"  - {p['name']}")


# ═══════════════════════════════════════════════════════════════════
# 顶层命令
# ═══════════════════════════════════════════════════════════════════

@app.command
def compact(
    *,
    operator: str = "unknown",
    dialog: str | None = None,
    previous_summary: str = "",
):
    """压缩对话上下文（需 ReMe）。"""
    result = do_compact(
        operator=operator,
        dialog=dialog,
        previous_summary=previous_summary,
    )
    if "error" in result:
        print(f"错误: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"压缩完成 → {result.get('path', '')}")
    print(result.get("summary", ""))


@app.command
def search(
    *,
    query: str,
    operator: str | None = None,
    thought_id: str | None = None,
    label: str | None = None,
    opt_layer: str | None = None,
    max_results: int = 5,
):
    """语义检索历史优化经验（需 ReMe）。"""
    result = do_search(
        query=query,
        operator=operator,
        thought_id=thought_id,
        label=label,
        opt_layer=opt_layer,
        max_results=max_results,
    )
    if not result["results"]:
        print("无结果")
        return
    print(f"查询: {query}  →  {len(result['results'])} 条结果")
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))


@app.command
def watch(
    *,
    operator: str | None = None,
    threshold_mb: float = 5.0,
    cooldown_seconds: int = 300,
):
    """文件监听守护：自动触发上下文压缩。"""
    do_watch(
        operator=operator,
        threshold_mb=threshold_mb,
        cooldown_seconds=cooldown_seconds,
    )
