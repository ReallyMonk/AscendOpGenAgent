#!/usr/bin/env python3
"""
performance-report render_html.py

读取 kope agent 整理好的 perf_data.json, 生成单文件 HTML 性能报告(浅色主题)。

支持两种 schema:
- 新 schema (多算子): {"operators": [{"op_name": ..., "shapes": [...], "knowledge_cards": [...]}]}
- 旧 schema (单算子, 兼容): {"op_name": ..., "shapes": [...], "knowledge_cards": [...]}

输出: performance_report.html (单文件, 内嵌 CSS + Chart.js CDN)

================================================================
  容错策略
----------------------------------------------------------------
kope agent 输出的 perf_data.json 经常字段不全 / 类型错 / 格式乱
(空 shapes / 字段为 None / 字符串带单位 / operators=None 等), 本渲染器
采用 "轻清洗 + 允许留空" 策略, 任何数据问题都不会让整份报告失败:

  - data 不是 dict              -> 降级为 {"operators": []}
  - operators 不是 list          -> 降级为 []
  - 单个 op 无 shapes            -> 跳过该 op, 继续渲染其他 op
  - 数值字段是 None / str / 异常 -> _safe_float() 降级为 0.0
  - knowledge_cards 不是 list    -> 降级为 []
  - knowledge_card 缺 category   -> 归入 "other"

极端情况 (所有算子都无 shapes) 会输出 "无算子数据" 占位报告,
header + 整体结论卡片仍正常渲染, 调用方不会收到异常。
================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from string import Template

# ---------- 指标计算 ----------

def _geometric_mean(values: list[float]) -> float:
    positives = [v for v in values if v > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def _speedup_class(sp: float) -> str:
    if sp >= 2.0:
        return "speedup-high"
    if sp >= 1.0:
        return "speedup-mid"
    return "speedup-low"


def compute_shape_metrics(shapes) -> dict:
    """给定一个 shape 列表, 返回该列表的汇总指标. 容错: 过滤非 dict 项."""
    shapes = [s for s in _safe_get_list(shapes) if isinstance(s, dict)]
    speedups_opt: list[float] = []
    speedups_pt: list[float] = []
    for s in shapes:
        before = _safe_float(s.get("ascendc_before_ms"))
        after = _safe_float(s.get("ascendc_after_ms"))
        pt = _safe_float(s.get("pytorch_ms"))
        if before > 0 and after > 0:
            speedups_opt.append(before / after)
        if pt > 0 and after > 0:
            speedups_pt.append(pt / after)
    return {
        "n_shapes": len(shapes),
        "geo_mean_opt": _geometric_mean(speedups_opt),
        "max_opt": max(speedups_opt) if speedups_opt else 0.0,
        "min_opt": min(speedups_opt) if speedups_opt else 0.0,
        "geo_mean_pt": _geometric_mean(speedups_pt),
    }


def compute_overall_metrics(operators) -> dict:
    """对所有算子的所有 shape 一起计算整体指标. 容错: 算子/shape 非 dict 时跳过."""
    all_shapes: list[dict] = []
    for op in _safe_get_list(operators):
        if not isinstance(op, dict):
            continue
        for s in _safe_get_list(op.get("shapes")):
            if isinstance(s, dict):
                all_shapes.append(s)
    return compute_shape_metrics(all_shapes)


# ---------- schema 归一化 ----------

def _normalize_data(data: dict) -> dict:
    """将旧 schema 转换为新 schema; 已经是新 schema 则原样返回。"""
    if "operators" in data and data["operators"]:
        return data
    # 旧 schema 兼容
    return {
        "experiment_name": data.get("experiment_name") or data.get("op_name", ""),
        "generated_at": data.get("generated_at", ""),
        "global_meta": {
            "device": data.get("device", ""),
        },
        "operators": [
            {
                "op_name": data.get("op_name", "unknown"),
                "shapes": data.get("shapes", []),
                "knowledge_cards": data.get("knowledge_cards", []),
            }
        ],
        "knowledge_cards": [],
    }


# ---------- 知识卡片分组 ----------

def _group_knowledge_cards(cards: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for c in cards or []:
        cat = (c.get("category") or "other").strip() or "other"
        groups.setdefault(cat, []).append(c)
    return groups


# ---------- HTML 转义 ----------

def _escape(text) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _safe_id(text: str) -> str:
    """把任意字符串变成安全的 HTML id (字母/数字/下划线)。"""
    s = re.sub(r"[^A-Za-z0-9_]", "_", text or "")
    return s or "x"


# ---------- 类型容错 (kope agent 输出经常字段缺失/类型错) ----------

def _safe_float(v, default: float = 0.0) -> float:
    """轻量级容错: 把任意值转 float, 失败时降级为 default。
    - int / float / bool -> 直接返回 float (bool 是 int 子类, True=1.0)
    - str                -> 尝试 float(v), 失败 (如 "0.15 ms") 降级为 default
    - None / list / dict -> 降级为 default
    """
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except (ValueError, TypeError):
            return float(default)
    return float(default)


def _safe_get_list(v) -> list:
    """容错: 不是 list 时返回 []。"""
    return v if isinstance(v, list) else []


def _safe_get_dict(v) -> dict:
    """容错: 不是 dict 时返回 {}。"""
    return v if isinstance(v, dict) else {}


# ---------- 渲染片段 ----------

def _render_table_row(shape) -> str:
    """渲染单行 shape 数据. 容错: shape 非 dict 时返回占位行."""
    if not isinstance(shape, dict):
        shape = {}
    desc = shape.get("description") or ""
    sid = shape.get("shape_id") or ""
    pt = _safe_float(shape.get("pytorch_ms"))
    before = _safe_float(shape.get("ascendc_before_ms"))
    after = _safe_float(shape.get("ascendc_after_ms"))

    if before > 0 and after > 0:
        sp_opt = before / after
        sp_class = _speedup_class(sp_opt)
        sp_opt_str = f"{sp_opt:.2f}x"
    else:
        sp_class = "speedup-low"
        sp_opt_str = "—"

    sp_pt_str = f"{pt / after:.2f}x" if pt > 0 and after > 0 else "—"

    return (
        "<tr>"
        f"<td>{_escape(sid)}</td>"
        f"<td>{_escape(desc)}</td>"
        f"<td>{pt:.4f}</td>"
        f"<td>{before:.4f}</td>"
        f"<td>{after:.4f}</td>"
        f'<td class="{sp_class}">{sp_opt_str}</td>'
        f"<td>{sp_pt_str}</td>"
        "</tr>"
    )


def _render_knowledge_cards_html(cards) -> str:
    """渲染知识卡片. 容错: cards 非 list / 含非 dict 项 -> 全部降级为空."""
    cards = [c for c in _safe_get_list(cards) if isinstance(c, dict)]
    if not cards:
        return '<p class="muted">本次优化未记录使用的知识卡片。</p>'
    groups = _group_knowledge_cards(cards)
    parts: list[str] = []
    for cat in sorted(groups.keys()):
        parts.append('<div class="kc-group">')
        parts.append(f"<h3>{_escape(cat)}</h3>")
        for c in groups[cat]:
            parts.append('<div class="kc-item">')
            parts.append(
                f'<div class="kc-title">{_escape(c.get("title", ""))}</div>'
            )
            card_id = c.get("card_id", "")
            if card_id:
                parts.append(
                    f'<div class="kc-id">{_escape(card_id)}</div>'
                )
            app = c.get("application", "")
            if app:
                parts.append(
                    f'<div class="kc-app">{_escape(app)}</div>'
                )
            parts.append("</div>")
        parts.append("</div>")
    return "\n".join(parts)


# ---------- 单个算子 section ----------

OPERATOR_SECTION_TEMPLATE = Template("""
<div class="op-section">
  <div class="op-header">
    <h2>算子 $idx: $op_name</h2>
    <div class="op-subtitle">$n_shapes 个测试 shape</div>
  </div>

  <div class="cards op-cards">
    <div class="card">
      <div class="label">测试 Shape 数</div>
      <div class="value">$n_shapes</div>
    </div>
    <div class="card highlight">
      <div class="label">加速比</div>
      <div class="value">$geo_opt<span class="unit">x</span></div>
    </div>
    <div class="card">
      <div class="label">最大加速比</div>
      <div class="value">$max_opt<span class="unit">x</span></div>
    </div>
    <div class="card">
      <div class="label">最小加速比</div>
      <div class="value">$min_opt<span class="unit">x</span></div>
    </div>
  </div>

  <div class="section">
    <h3>加速比对比 (AscendC 优化前 vs 优化后)</h3>
    <div class="chart-container">
      <canvas id="$chart_id"></canvas>
    </div>
    <p class="note">注: PyTorch 参考速度仅作对比参考, 主加速比基于 AscendC 优化前后计算。</p>
  </div>

  <div class="section">
    <h3>详细数据</h3>
    <table>
      <thead>
        <tr>
          <th>Shape ID</th>
          <th>Description</th>
          <th>PyTorch (ms)</th>
          <th>AscendC 优化前 (ms)</th>
          <th>AscendC 优化后 (ms)</th>
          <th>优化加速比</th>
          <th>vs PyTorch</th>
        </tr>
      </thead>
      <tbody>
        $table_rows
      </tbody>
    </table>
  </div>

  $kc_block
</div>
""")


def _render_chart_script(chart_id: str, shapes: list[dict]) -> str:
    labels = [
        s.get("description") or s.get("shape_id", f"shape_{i}")
        for i, s in enumerate(shapes)
    ]
    pytorch = [_safe_float(s.get("pytorch_ms")) for s in shapes]
    before = [_safe_float(s.get("ascendc_before_ms")) for s in shapes]
    after = [_safe_float(s.get("ascendc_after_ms")) for s in shapes]
    return f"""
new Chart(document.getElementById('{chart_id}').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(labels, ensure_ascii=False)},
    datasets: [
      {{
        label: 'PyTorch (参考)',
        data: {json.dumps(pytorch)},
        backgroundColor: 'rgba(101, 109, 118, 0.5)',
        borderColor: 'rgba(101, 109, 118, 1)',
        borderWidth: 1
      }},
      {{
        label: 'AscendC 优化前',
        data: {json.dumps(before)},
        backgroundColor: 'rgba(255, 140, 0, 0.6)',
        borderColor: 'rgba(255, 140, 0, 1)',
        borderWidth: 1
      }},
      {{
        label: 'AscendC 优化后',
        data: {json.dumps(after)},
        backgroundColor: 'rgba(30, 111, 255, 0.7)',
        borderColor: 'rgba(30, 111, 255, 1)',
        borderWidth: 1
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ color: '#1f2328' }} }},
      title: {{ display: false }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#1f2328' }}, grid: {{ color: '#eaeef2' }} }},
      y: {{
        ticks: {{ color: '#1f2328' }},
        grid: {{ color: '#eaeef2' }},
        title: {{ display: true, text: '耗时 (ms)', color: '#656d76' }}
      }}
    }}
  }}
}});
"""


def _render_operator_section(op, idx: int) -> tuple[str, str]:
    """渲染单个算子 section. 返回 (HTML 片段, chart JS 脚本).
    容错: op 非 dict / shapes 缺失或空 -> 返回 ("", "")
    """
    op = _safe_get_dict(op)
    op_name = op.get("op_name") or f"op_{idx}"
    shapes = _safe_get_list(op.get("shapes"))
    if not shapes:
        return "", ""
    metrics = compute_shape_metrics(shapes)
    chart_id = f"opChart_{idx}_{_safe_id(op_name)}"
    table_rows = "\n".join(_render_table_row(s) for s in shapes)
    cards = _safe_get_list(op.get("knowledge_cards"))
    kc_block = (
        f'<div class="section"><h3>优化技巧 (来自知识卡片)</h3>'
        f'{_render_knowledge_cards_html(cards)}</div>'
        if cards
        else ""
    )
    html = OPERATOR_SECTION_TEMPLATE.substitute(
        idx=idx,
        op_name=_escape(op_name),
        n_shapes=metrics["n_shapes"],
        geo_opt=f"{metrics['geo_mean_opt']:.2f}",
        max_opt=f"{metrics['max_opt']:.2f}",
        min_opt=f"{metrics['min_opt']:.2f}",
        chart_id=chart_id,
        table_rows=table_rows,
        kc_block=kc_block,
    )
    chart_js = _render_chart_script(chart_id, shapes)
    return html, chart_js


# ---------- 结论 ----------

def _render_overall_conclusion(overall: dict, n_ops: int, n_cards_global: int) -> str:
    sp = overall["geo_mean_opt"]
    n = overall["n_shapes"]
    if sp >= 3.0:
        verdict = "<strong>显著加速</strong>, 优化效果突出"
    elif sp >= 1.5:
        verdict = "<strong>明显加速</strong>, 优化策略有效"
    elif sp >= 1.0:
        verdict = "<strong>轻微加速</strong>, 部分 shape 有改进"
    elif sp > 0:
        verdict = "<strong>回退</strong>, 优化未带来收益, 建议回滚或重新设计"
    else:
        verdict = "<strong>无有效数据</strong>"
    return (
        f"<p>本次实验在 <strong>{n_ops}</strong> 个算子、合计 <strong>{n}</strong> 个 shape 上完成测试。</p>"
        f"<p>整体几何平均优化加速比: <strong>{sp:.2f}x</strong>, {verdict}。</p>"
        f"<p>实验过程中参考了 <strong>{n_cards_global}</strong> 张知识卡片, 为系统性的优化策略提供指导。</p>"
    )


# ---------- 算子总表 ----------

def _render_overall_op_table(operators) -> str:
    """渲染"算子总表", 算子级汇总行 + 各 shape 展开行. 容错: 算子/shape 非 dict 时跳过.

    表头列: 算子 | Shape | 加速比 | PyTorch (ms) | AscendC 优化前 (ms) | AscendC 优化后 (ms)
    - 算子汇总行: 蓝底加粗, 跨 shape 数
    - shape 行: 算子名缩进, 显示该 shape 的加速比 / PyTorch / AscendC 优化前后耗时
    - 数字列均统一为 4 位小数 (纯数字, 不带 x), 整行格式一致
    """
    rows: list[str] = []
    for op in _safe_get_list(operators):
        if not isinstance(op, dict):
            continue
        op_name = op.get("op_name") or "unknown"
        shapes = _safe_get_list(op.get("shapes"))
        if not shapes:
            continue
        metrics = compute_shape_metrics(shapes)
        sp_class = _speedup_class(metrics["geo_mean_opt"])
        # 算子级 PyTorch / Before / After 几何平均
        pt_values = [
            _safe_float(s.get("pytorch_ms"))
            for s in shapes
            if _safe_float(s.get("pytorch_ms")) > 0
        ]
        before_values = [
            _safe_float(s.get("ascendc_before_ms"))
            for s in shapes
            if _safe_float(s.get("ascendc_before_ms")) > 0
        ]
        after_values = [
            _safe_float(s.get("ascendc_after_ms"))
            for s in shapes
            if _safe_float(s.get("ascendc_after_ms")) > 0
        ]
        pt_geo = _geometric_mean(pt_values)
        before_geo = _geometric_mean(before_values)
        after_geo = _geometric_mean(after_values)
        # 算子汇总行 (加速比列与同行其他数字列统一为 4 位小数, 加 x 后缀; 背景均为蓝底, 加速比文字色仍按 speedup 档位)
        # Shape 数从原 colspan=2 的算子名拆出, 单独占据 Shape 列
        rows.append(
            "<tr>"
            f'<td class="op-summary-name">📦 <strong>{_escape(op_name)}</strong></td>'
            f'<td class="op-summary-name op-summary-shape">{metrics["n_shapes"]} shapes</td>'
            f'<td class="op-summary-num {sp_class}">{metrics["geo_mean_opt"]:.4f}x</td>'
            f'<td class="op-summary-num">{pt_geo:.4f}</td>'
            f'<td class="op-summary-num">{before_geo:.4f}</td>'
            f'<td class="op-summary-num">{after_geo:.4f}</td>'
            "</tr>"
        )
        # 每个 shape 一行
        for s in shapes:
            if not isinstance(s, dict):
                s = {}
            desc = s.get("description") or s.get("shape_id") or ""
            pt = _safe_float(s.get("pytorch_ms"))
            before = _safe_float(s.get("ascendc_before_ms"))
            after = _safe_float(s.get("ascendc_after_ms"))
            if before > 0 and after > 0:
                sp = before / after
                row_class = _speedup_class(sp)
                # 4 位小数 + x 后缀, 与算子汇总行加速比格式一致, 等宽数字保证对齐
                sp_str = f"{sp:.4f}x"
            else:
                row_class = "speedup-low"
                sp_str = "—"
            rows.append(
                "<tr>"
                f'<td class="op-name-indent">└ {_escape(op_name)}</td>'
                f"<td>{_escape(desc)}</td>"
                # 加速比列与算子汇总行对齐: 右对齐 + 等宽数字
                f'<td class="{row_class} op-num">{sp_str}</td>'
                f'<td class="op-num">{pt:.4f}</td>'
                f'<td class="op-num">{before:.4f}</td>'
                f'<td class="op-num">{after:.4f}</td>'
                "</tr>"
            )
    return "\n".join(rows)


# ---------- 全局 meta 网格 ----------

META_LABEL_ALIASES = {
    "cann_version": "CANN",
    "cann": "CANN",
    "npu_model": "NPU 型号",
    "npu": "NPU 型号",
    "npu_memory_gb": "NPU 显存 (GB)",
    "soc_version": "SOC",
    "driver_version": "驱动版本",
    "torch_version": "PyTorch",
    "torch_npu_version": "torch_npu",
    "device": "设备",
    "os": "操作系统",
    "python_version": "Python",
    "model": "优化模型",
    "total_tokens": "总 Tokens",
    "input_tokens": "输入 Tokens",
    "output_tokens": "输出 Tokens",
    "cache_hit_rate": "缓存命中率",
    "cache_read_input_tokens": "缓存读取 Tokens",
    "cache_creation_input_tokens": "缓存创建 Tokens",
}

# token_usage 子字段的显示标签
TOKEN_USAGE_LABELS = {
    "input_tokens": "输入 Tokens",
    "output_tokens": "输出 Tokens",
    "total_tokens": "总 Tokens",
    "cache_read_input_tokens": "缓存读取 Tokens",
    "cache_creation_input_tokens": "缓存创建 Tokens",
    "cache_hit_rate": "缓存命中率",
}


def _render_meta_grid(meta: dict) -> str:
    """渲染全局 meta 信息网格，支持嵌套对象（如 token_usage）。"""
    if not meta:
        return ""
    items: list[tuple[str, str]] = []
    for key, value in meta.items():
        if value in (None, "", []):
            continue
        # 处理嵌套对象（如 token_usage）
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_value in (None, "", []):
                    continue
                label = TOKEN_USAGE_LABELS.get(sub_key, META_LABEL_ALIASES.get(sub_key, sub_key))
                # cache_hit_rate 显示为百分比
                if sub_key == "cache_hit_rate" and isinstance(sub_value, (int, float)):
                    items.append((label, f"{sub_value:.1%}" if sub_value <= 1 else f"{sub_value:.1f}%"))
                elif isinstance(sub_value, (int, float)) and sub_key != "cache_hit_rate":
                    items.append((label, f"{sub_value:,}"))
                else:
                    items.append((label, str(sub_value)))
            continue
        label = META_LABEL_ALIASES.get(key, key)
        items.append((label, str(value)))
    if not items:
        return ""
    cells = "".join(
        f'<div class="env-cell"><span class="env-label">{_escape(l)}</span>'
        f'<span class="env-value">{_escape(v)}</span></div>'
        for l, v in items
    )
    return f'<div class="env-grid">{cells}</div>'


# ---------- HTML 模板 (浅色主题) ----------

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>性能优化报告 - $title</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  background: #f6f8fa;
  color: #1f2328;
  line-height: 1.6;
  padding: 24px;
}
.container { max-width: 1240px; margin: 0 auto; }

header {
  background: linear-gradient(135deg, #0969da 0%, #218bff 100%);
  border-radius: 12px;
  padding: 32px;
  margin-bottom: 24px;
  color: #ffffff;
  box-shadow: 0 4px 16px rgba(9, 105, 218, 0.2);
}
header h1 { font-size: 32px; margin-bottom: 12px; font-weight: 700; }
header .meta { display: flex; gap: 12px; flex-wrap: wrap; font-size: 14px; margin-bottom: 16px; }
header .meta span { background: rgba(255, 255, 255, 0.18); padding: 4px 12px; border-radius: 16px; }
header .meta strong { font-weight: 600; }

.env-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px;
}
.env-cell { display: flex; flex-direction: column; padding: 4px 8px; }
.env-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85; }
.env-value { font-size: 14px; font-weight: 600; }

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.card {
  background: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(31, 35, 40, 0.04);
}
.card .label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #656d76;
  margin-bottom: 8px;
}
.card .value { font-size: 28px; font-weight: 700; color: #1f2328; }
.card.highlight { background: linear-gradient(135deg, #ddf4ff 0%, #b6e3ff 100%); border-color: #0969da; }
.card.highlight .value { color: #0969da; }
.card .unit { font-size: 14px; color: #656d76; margin-left: 4px; font-weight: 500; }

.op-section {
  background: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(31, 35, 40, 0.04);
}
.op-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 2px solid #d0d7de;
}
.op-header h2 { font-size: 22px; color: #0969da; }
.op-subtitle { font-size: 13px; color: #656d76; }

.section {
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
}
.section:last-child { margin-bottom: 0; }
.section h3 {
  font-size: 16px;
  margin-bottom: 14px;
  color: #1f2328;
  font-weight: 600;
}
.chart-container { position: relative; height: 360px; margin-bottom: 12px; }
.note { font-size: 13px; color: #656d76; }
.muted { font-size: 14px; color: #656d76; font-style: italic; }

table { width: 100%; border-collapse: collapse; font-size: 14px; background: #ffffff; border-radius: 6px; overflow: hidden; }
th {
  background: #f6f8fa;
  text-align: left;
  padding: 12px;
  color: #656d76;
  font-weight: 600;
  border-bottom: 1px solid #d0d7de;
}
td { padding: 12px; border-bottom: 1px solid #eaeef2; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f6f8fa; }
.speedup-high { color: #1a7f37; font-weight: 700; background: rgba(26, 127, 55, 0.08); }
.speedup-mid { color: #9a6700; font-weight: 700; background: rgba(154, 103, 0, 0.08); }
.speedup-low { color: #cf222e; font-weight: 700; background: rgba(207, 34, 46, 0.08); }

.kc-group { margin-bottom: 16px; }
.kc-group:last-child { margin-bottom: 0; }
.kc-group h3 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #0969da;
  margin-bottom: 10px;
  padding: 6px 12px;
  background: rgba(9, 105, 218, 0.08);
  border-left: 3px solid #0969da;
  border-radius: 0 4px 4px 0;
  font-weight: 700;
}
.kc-item {
  background: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.kc-item:last-child { margin-bottom: 0; }
.kc-item .kc-title { font-size: 15px; font-weight: 600; color: #1f2328; margin-bottom: 2px; }
.kc-item .kc-id { font-size: 11px; color: #656d76; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.kc-item .kc-app { font-size: 14px; color: #1f2328; margin-top: 6px; }

.conclusion {
  background: linear-gradient(135deg, rgba(9, 105, 218, 0.08) 0%, rgba(33, 139, 255, 0.08) 100%);
  border-left: 4px solid #1a7f37;
  padding: 20px;
  border-radius: 6px;
  border: 1px solid rgba(9, 105, 218, 0.2);
  border-left-width: 4px;
}
.conclusion p { margin-bottom: 8px; color: #1f2328; }
.conclusion p:last-child { margin-bottom: 0; }
.conclusion strong { color: #1a7f37; }

.op-table-wrap { overflow-x: auto; }
.op-table-wrap table { table-layout: auto; }
.op-table-wrap th:first-child, .op-table-wrap td:first-child {
  min-width: 140px;
}
.op-table-wrap .op-summary-name {
  background: rgba(9, 105, 218, 0.08);
  color: #0969da;
  font-weight: 600;
  border-top: 2px solid #0969da;
  border-bottom: 1px solid #d0d7de;
}
.op-table-wrap .op-summary-meta { color: #656d76; font-weight: 500; font-size: 12px; }
.op-table-wrap .op-summary-shape {
  color: #656d76;
  font-weight: 600;
  font-size: 13px;
  text-align: left;
}
.op-table-wrap .op-summary-num {
  background: rgba(9, 105, 218, 0.08);
  font-weight: 600;
  border-top: 2px solid #0969da;
  border-bottom: 1px solid #d0d7de;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
/* 算子汇总行的加速比列: 背景与其他列同为蓝底, 文字色仍按 speedup 档位 */
.op-table-wrap .op-summary-num.speedup-high {
  background: rgba(9, 105, 218, 0.08);
  color: #1a7f37;
  font-weight: 700;
}
.op-table-wrap .op-summary-num.speedup-mid {
  background: rgba(9, 105, 218, 0.08);
  color: #9a6700;
  font-weight: 700;
}
.op-table-wrap .op-summary-num.speedup-low {
  background: rgba(9, 105, 218, 0.08);
  color: #cf222e;
  font-weight: 700;
}
.op-table-wrap .op-name-indent {
  font-size: 12px;
  color: #656d76;
  padding-left: 20px;
  white-space: nowrap;
}
.op-table-wrap .op-num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

footer { text-align: center; color: #656d76; font-size: 12px; margin-top: 32px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>📊 性能优化报告</h1>
    <div class="meta">
      <span>实验: <strong>$title</strong></span>
      <span>$generated_at</span>
    </div>
    $env_grid
  </header>

  <div class="cards">
    <div class="card">
      <div class="label">测试算子数</div>
      <div class="value">$n_ops</div>
    </div>
    <div class="card">
      <div class="label">总 Shape 数</div>
      <div class="value">$n_shapes_total</div>
    </div>
    <div class="card highlight">
      <div class="label">整体加速比</div>
      <div class="value">$geo_mean_opt<span class="unit">x</span></div>
    </div>
    <div class="card">
      <div class="label">最大 / 最小加速比</div>
      <div class="value">$max_opt<span class="unit">x</span> / $min_opt<span class="unit">x</span></div>
    </div>
  </div>

  $op_table

  $operator_sections

  $global_kc_block

  <div class="op-section">
    <div class="op-header">
      <h2>整体结论</h2>
      <div class="op-subtitle">综合所有算子</div>
    </div>
    <div class="conclusion">$conclusion</div>
  </div>

  <footer>由 kope agent + performance-report skill 生成</footer>
</div>

<script>
$chart_scripts
</script>
</body>
</html>
""")


# ---------- 入口 ----------

def render_report(data, output_path: str | Path) -> Path:
    """主入口. 极致容错: data 任意错误 -> 输出占位报告, 不抛异常.

    输入 data 期望是 dict, 但 None / list / 其他类型都会被降级为 {"operators": []}.
    算子无 shapes / shapes 为空 时, 跳过该算子继续渲染其他算子.
    """
    # 顶层容错: data 必须是 dict, 否则降级为 {"operators": []}
    if not isinstance(data, dict):
        data = {"operators": []}
    normalized = _normalize_data(data)
    operators = _safe_get_list(normalized.get("operators"))
    # 过滤: 必须是 dict, 且有非空 shapes (空 shapes 的算子直接跳过, 不会中断整份报告)
    operators = [
        op for op in operators
        if isinstance(op, dict) and _safe_get_list(op.get("shapes"))
    ]

    overall = compute_overall_metrics(operators)
    n_ops = len(operators)
    n_shapes_total = overall["n_shapes"]

    # 算子总表 (快速浏览所有算子优化结果)
    op_table_rows = _render_overall_op_table(operators)
    op_table_html = (
        '<div class="section">'
        '<h3>算子总表 (各算子 × 各 shape 优化效果一览)</h3>'
        '<p class="note">每一行对应一个 shape 的实测数据, 算子名前的 “└” 表示该 shape 隶属于上方蓝色背景的算子。'
        '加速比 = AscendC 优化前 / AscendC 优化后。</p>'
        '<div class="op-table-wrap"><table>'
        '<thead><tr>'
        '<th>算子</th><th>Shape</th>'
        '<th>加速比</th>'
        '<th>PyTorch (ms)</th>'
        '<th>AscendC 优化前 (ms)</th>'
        '<th>AscendC 优化后 (ms)</th>'
        '</tr></thead>'
        f'<tbody>{op_table_rows}</tbody>'
        '</table></div>'
        '</div>'
        if op_table_rows
        else ""
    )

    # 渲染算子 sections
    op_htmls: list[str] = []
    chart_scripts: list[str] = []
    for idx, op in enumerate(operators, start=1):
        html, js = _render_operator_section(op, idx)
        if html:
            op_htmls.append(html)
            chart_scripts.append(js)
    operator_sections = "\n".join(op_htmls)
    chart_scripts_block = "\n".join(chart_scripts)

    # 全局知识卡片(实验级别)
    global_cards = _safe_get_list(normalized.get("knowledge_cards"))
    if global_cards:
        global_kc_block = (
            '<div class="op-section">'
            '<div class="op-header"><h2>实验级知识卡片</h2>'
            '<div class="op-subtitle">本次实验共享</div></div>'
            '<div class="section"><h3>优化技巧 (全局)</h3>'
            f'{_render_knowledge_cards_html(global_cards)}</div>'
            '</div>'
        )
    else:
        global_kc_block = ""

    # 全局 meta + 标题
    title = (normalized.get("experiment_name")
             or (operators[0].get("op_name") if operators else "unknown")
             or "unknown")
    env_grid = _render_meta_grid(_safe_get_dict(normalized.get("global_meta")))
    conclusion = _render_overall_conclusion(overall, n_ops, len(global_cards))

    html = HTML_TEMPLATE.substitute(
        title=_escape(title),
        generated_at=_escape(normalized.get("generated_at", "")),
        env_grid=env_grid,
        n_ops=n_ops,
        n_shapes_total=n_shapes_total,
        geo_mean_opt=f"{overall['geo_mean_opt']:.2f}",
        max_opt=f"{overall['max_opt']:.2f}",
        min_opt=f"{overall['min_opt']:.2f}",
        operator_sections=operator_sections,
        op_table=op_table_html,
        global_kc_block=global_kc_block,
        conclusion=conclusion,
        chart_scripts=chart_scripts_block,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="性能报告生成 (kope agent 配套 skill, 浅色主题 + 多算子)"
    )
    parser.add_argument("--input", "-i", required=True, help="perf_data.json 路径")
    parser.add_argument(
        "--output", "-o", required=True, help="performance_report.html 输出路径"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = render_report(data, args.output)

    normalized = _normalize_data(data)
    operators = normalized["operators"]
    overall = compute_overall_metrics(operators)
    print(f"报告已生成: {out}")
    print(f"  - 算子数: {len(operators)}")
    print(f"  - 总 Shape 数: {overall['n_shapes']}")
    print(f"  - 整体加速比: {overall['geo_mean_opt']:.2f}x")
    print(f"  - 最大加速比: {overall['max_opt']:.2f}x")
    print(f"  - 最小加速比: {overall['min_opt']:.2f}x")


if __name__ == "__main__":
    main()
