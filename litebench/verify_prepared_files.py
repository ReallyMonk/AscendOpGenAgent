#!/usr/bin/env python3
"""Verify prepared files for all operators."""

from pathlib import Path

OUTPUT_DIR = Path("/home/hrl-kope/AscendOpGenAgent/output")

OPERATORS = [
    "add_layer_norm",
    "add_rms_norm",
    "add_rms_norm_dynamic_quant",
    "add_rms_norm_quant",
    "avg_pool2d",
    "batch_mat_mul_v3",
    "clipped_swiglu",
    "conv2d_v2",
    "cross_entropy_loss",
    "embedding_bag",
    "gather_v2",
    "ge_glu_v2",
    "gelu",
    "gelu_mul",
    "layer_norm_v3",
    "masked_softmax_with_rel_pos_bias",
    "mat_mul_v3",
    "relu",
    "rms_norm",
    "scaled_masked_softmax_v2",
    "softmax_v2",
    "swi_glu",
]

def capitalize_op_name(op_name):
    parts = op_name.split('_')
    return ''.join(word.capitalize() for word in parts)

def main():
    print("Operator File Preparation Status:")
    print("=" * 70)

    results = []
    for op_name in OPERATORS:
        op_dir = OUTPUT_DIR / op_name
        if not op_dir.exists():
            results.append((op_name, "skip", "Directory not found"))
            continue

        dsl_file = op_dir / f"{op_name}_dsl.py"
        custom_dir = op_dir / f"{capitalize_op_name(op_name)}Custom"

        dsl_ok = dsl_file.exists()
        custom_ok = custom_dir.exists() and custom_dir.is_dir()

        if dsl_ok and custom_ok:
            results.append((op_name, "ok", ""))
        elif dsl_ok and not custom_ok:
            results.append((op_name, "partial", "DSL OK, Custom missing"))
        elif not dsl_ok and custom_ok:
            results.append((op_name, "partial", "Custom OK, DSL missing"))
        else:
            results.append((op_name, "fail", "Both missing"))

    ok_count = partial_count = fail_count = skip_count = 0

    for op_name, status, note in results:
        if status == "ok":
            ok_count += 1
            print(f"  [OK] {op_name}")
        elif status == "partial":
            partial_count += 1
            print(f"  [PARTIAL] {op_name} - {note}")
        elif status == "fail":
            fail_count += 1
            print(f"  [FAIL] {op_name} - {note}")
        else:
            skip_count += 1
            print(f"  [SKIP] {op_name} - {note}")

    print("=" * 70)
    print(f"Total: {len(results)} operators")
    print(f"  OK: {ok_count}")
    print(f"  Partial: {partial_count}")
    print(f"  Fail: {fail_count}")
    print(f"  Skip: {skip_count}")

if __name__ == "__main__":
    main()