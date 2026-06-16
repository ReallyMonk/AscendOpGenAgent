#!/usr/bin/env python3
"""
performance-report 容错性实测: 用多种异常数据喂给 render_html.render_report,
观察是否崩溃 / 优雅降级 / 异常输出。

不做任何代码修改, 仅做观察。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "references"))
from render_html import render_report  # noqa: E402

OUT_DIR = Path("/tmp/perf_report_fuzz")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_good_shape(sid: str = "s1", desc: str = "M=128, N=128") -> dict:
    return {
        "shape_id": sid,
        "description": desc,
        "pytorch_ms": 0.15,
        "ascendc_before_ms": 0.45,
        "ascendc_after_ms": 0.08,
    }


def make_good_op(name: str = "add_rms_norm") -> dict:
    return {
        "op_name": name,
        "shapes": [make_good_shape()],
    }


CASES: list[tuple[str, object, str]] = [
    ("T01_baseline_good", {
        "experiment_name": "exp_001",
        "generated_at": "2026-06-06T10:00:00",
        "global_meta": {"device": "Ascend 910B", "cann_version": "CANN 8.0.RC2"},
        "operators": [make_good_op()],
    }, "正常基线"),

    ("T02_field_missing", {
        "experiment_name": "exp_002",
        "operators": [{
            "op_name": "op_x",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0.3,
            }],
        }],
    }, "缺 ascendc_after_ms"),

    ("T03_field_none", {
        "experiment_name": "exp_003",
        "operators": [{
            "op_name": "op_x",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": None,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "ascendc_before_ms = None"),

    ("T04_field_string", {
        "experiment_name": "exp_004",
        "operators": [{
            "op_name": "op_x",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": "0.15 ms",
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "pytorch_ms = 字符串带单位"),

    ("T05_field_negative", {
        "experiment_name": "exp_005",
        "operators": [{
            "op_name": "op_x",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": -0.1,
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "pytorch_ms = 负数"),

    ("T06_old_schema_top_shapes", {
        "op_name": "op_legacy",
        "device": "Ascend 910B",
        "shapes": [make_good_shape()],
        "knowledge_cards": [],
    }, "旧 schema: 顶层 shapes + op_name"),

    ("T07_old_schema_no_opname", {
        "shapes": [make_good_shape()],
    }, "旧 schema: 缺 op_name"),

    ("T08_shapes_empty", {
        "operators": [{"op_name": "op_empty", "shapes": []}],
    }, "shapes = []"),

    ("T09_op_no_shapes", {
        "operators": [{"op_name": "op_no_shapes"}],
    }, "算子无 shapes 字段"),

    ("T10_kc_no_category", {
        "operators": [{
            "op_name": "op_kc",
            "shapes": [make_good_shape()],
            "knowledge_cards": [{"card_id": "c1", "title": "T1", "application": "A1"}],
        }],
    }, "knowledge_card 缺 category"),

    ("T11_desc_html_chars", {
        "operators": [{
            "op_name": "op_html",
            "shapes": [{
                "shape_id": "s1",
                "description": '<script>alert("xss")</script> M=128 & N=128 "test"',
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "description 含 HTML / 引号"),

    ("T12_desc_unicode", {
        "operators": [{
            "op_name": "op_unicode_算子_🚀",
            "shapes": [{
                "shape_id": "s1",
                "description": "🎉 M=128, N=128 (小型)",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "算子名 / 描述含 Unicode"),

    ("T13_extra_fields", {
        "operators": [{
            "op_name": "op_extra",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0.05,
                "extra_field_1": "ignored",
                "extra_field_2": 999,
            }],
        }],
    }, "shape 含多余字段"),

    ("T14_after_zero", {
        "operators": [{
            "op_name": "op_zero",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0,
            }],
        }],
    }, "ascendc_after_ms = 0"),

    ("T15_before_zero", {
        "operators": [{
            "op_name": "op_bz",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "ascendc_before_ms = 0"),

    ("T16_no_meta", {
        "operators": [make_good_op()],
    }, "无 global_meta"),

    ("T17_meta_none_values", {
        "operators": [make_good_op()],
        "global_meta": {
            "device": "Ascend 910B",
            "cann_version": None,
            "npu_model": "",
            "extra": [],
        },
    }, "global_meta 含 None / 空"),

    ("T18_no_generated_at", {
        "operators": [make_good_op()],
    }, "无 generated_at"),

    ("T19_no_exp_name", {
        "operators": [make_good_op()],
        "generated_at": "2026-06-06T10:00:00",
    }, "无 experiment_name"),

    ("T20_empty_operators", {
        "operators": [],
    }, "operators = []"),

    ("T21_data_is_list", ["not a dict"], "data 是 list 而非 dict"),

    ("T22_data_is_none", None, "data 是 None"),

    ("T23_operators_none", {
        "operators": None,
    }, "operators = None"),

    ("T24_kc_not_list", {
        "operators": [{
            "op_name": "op_x",
            "shapes": [make_good_shape()],
            "knowledge_cards": "not a list",
        }],
    }, "knowledge_cards 是字符串"),

    ("T25_partial_shapes", {
        "operators": [
            make_good_op("op_with"),
            {"op_name": "op_without"},
            make_good_op("op_with2"),
        ],
    }, "3 算子, 中间一个无 shapes"),

    ("T26_shapeid_special", {
        "operators": [{
            "op_name": "op_x",
            "shapes": [{
                "shape_id": "s/1.2&3",
                "description": "d1",
                "pytorch_ms": 0.1,
                "ascendc_before_ms": 0.3,
                "ascendc_after_ms": 0.05,
            }],
        }],
    }, "shape_id 含 / & ."),

    ("T27_huge_value", {
        "operators": [{
            "op_name": "op_huge",
            "shapes": [{
                "shape_id": "s1",
                "description": "d1",
                "pytorch_ms": 1e9,
                "ascendc_before_ms": 1e9,
                "ascendc_after_ms": 1e-9,
            }],
        }],
    }, "超大 / 超小数值 (1e9 / 1e-9)"),

    ("T28_duplicate_opname", {
        "operators": [make_good_op("dup"), make_good_op("dup")],
    }, "算子同名"),
]


def main():
    results: list[dict] = []
    for case_id, payload, desc in CASES:
        out = OUT_DIR / f"{case_id}.html"
        try:
            render_report(payload, str(out))
            sz = out.stat().st_size
            html = out.read_text(encoding="utf-8")
            # 基本结构校验. 注意: count("<th") 会误算 <thead> 中的 <th 子串,
            # 必须用正则 <th[ >] 精确匹配标签边界.
            import re
            table_balance = html.count("<table") == html.count("</table>")
            tr_balance = html.count("<tr") == html.count("</tr>")
            td_balance = html.count("<td") == html.count("</td>")
            th_open = len(re.findall(r"<th[ >/]", html))
            th_close = html.count("</th>")
            th_balance = th_open == th_close
            ok = table_balance and tr_balance and td_balance and th_balance
            results.append({
                "id": case_id,
                "desc": desc,
                "status": "PASS" if ok else "STRUCT_BAD",
                "size": sz,
                "err": "" if ok else f"t={table_balance} tr={tr_balance} td={td_balance} th={th_balance} (th_open={th_open}, th_close={th_close})",
            })
        except Exception as e:
            results.append({
                "id": case_id,
                "desc": desc,
                "status": "CRASH",
                "size": 0,
                "err": f"{type(e).__name__}: {str(e)[:80]}",
            })

    print()
    print(f"{'ID':<32} {'STATUS':<10} {'SIZE':>7}  {'DESC':<32} {'ERR'}")
    print("-" * 130)
    for r in results:
        size = f"{r['size']}B" if r['size'] else "-"
        err = r["err"][:50] + ("..." if len(r["err"]) > 50 else "")
        print(f"{r['id']:<32} {r['status']:<10} {size:>7}  {r['desc']:<32} {err}")
    print()
    pass_cnt = sum(1 for r in results if r["status"] == "PASS")
    bad = sum(1 for r in results if r["status"] == "STRUCT_BAD")
    crash = sum(1 for r in results if r["status"] == "CRASH")
    print(f"PASS: {pass_cnt}  STRUCT_BAD: {bad}  CRASH: {crash}  TOTAL: {len(results)}")


if __name__ == "__main__":
    main()
