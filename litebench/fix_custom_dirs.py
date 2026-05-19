#!/usr/bin/env python3
"""
Fix Custom directory naming - capitalize properly.
"""

import os
from pathlib import Path

OUTPUT_DIR = Path("/home/hrl-kope/AscendOpGenAgent/output")

def capitalize_op_name(op_name):
    """Capitalize operator name (e.g., gelu -> Gelu, swi_glu -> SwiGlu)."""
    parts = op_name.split('_')
    return ''.join(word.capitalize() for word in parts)


def main():
    operators = [
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

    for op_name in operators:
        op_dir = OUTPUT_DIR / op_name
        if not op_dir.exists():
            continue

        # Find lowercase-custom directory (e.g., geluCustom)
        wrong_name = op_dir / f"{op_name}Custom"
        correct_name = op_dir / f"{capitalize_op_name(op_name)}Custom"

        if wrong_name.exists() and not correct_name.exists():
            os.rename(wrong_name, correct_name)
            print(f"Renamed: {op_name}Custom -> {capitalize_op_name(op_name)}Custom")
        elif wrong_name.exists() and correct_name.exists():
            # Both exist - remove the lowercase one
            import shutil
            shutil.rmtree(wrong_name)
            print(f"Removed duplicate: {op_name}Custom")
        elif correct_name.exists():
            print(f"Already correct: {correct_name.name}")

if __name__ == "__main__":
    main()