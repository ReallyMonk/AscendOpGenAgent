#!/usr/bin/env python3
"""
Prepare litebench run-custom test files for all operators.
Creates {op_name}_dsl.py and {op_name}Custom/ directory structure.
"""

import os
import shutil
from pathlib import Path

OUTPUT_DIR = Path("/home/hrl-kope/AscendOpGenAgent/output")

# Operators to process
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

def create_dsl_file(op_dir, op_name):
    """Create {op_name}_dsl.py from model.py."""
    model_py = op_dir / "model.py"
    dsl_py = op_dir / f"{op_name}_dsl.py"

    if not model_py.exists():
        print(f"  [SKIP] model.py not found")
        return False

    # Read model.py
    content = model_py.read_text(encoding="utf-8")

    # Create DSL file - simplified version of model.py
    # Remove sys.path manipulations, _HAS_ASCENDC_KERNEL checks, and _ext imports
    lines = content.split('\n')
    new_lines = []
    skip_next = False

    for line in lines:
        # Skip imports we don't need in DSL
        if 'import sys' in line and 'sys.path' in line:
            continue
        if 'from pathlib' in line and 'Path' in line:
            continue
        if '_KERNEL_BUILD' in line:
            continue
        if '_HAS_ASCENDC_KERNEL' in line:
            continue
        if '_ext' in line and ('import' in line):
            continue
        if 'sys.path.insert' in line:
            continue
        # Skip the try-except block lines for _ext import
        if 'try:' in line or 'ImportError' in line or '_HAS_ASCENDC_KERNEL' in line:
            continue
        if 'except' in line and 'ImportError' in line:
            continue
        new_lines.append(line)

    dsl_content = '\n'.join(new_lines)

    # Ensure proper formatting - remove empty lines at start
    while dsl_content.startswith('\n'):
        dsl_content = dsl_content[1:]

    dsl_py.write_text(dsl_content, encoding="utf-8")
    print(f"  [OK] Created {dsl_py.name}")
    return True


def _capitalize_op_name(op_name):
    """Capitalize operator name for directory naming (e.g., gelu -> Gelu, swi_glu -> SwiGlu)."""
    parts = op_name.split('_')
    return ''.join(word.capitalize() for word in parts)


def create_custom_dir(op_dir, op_name):
    """Create {op_name}Custom/ directory with AscendC kernel files."""
    kernel_dir = op_dir / "kernel"
    custom_dir = op_dir / f"{_capitalize_op_name(op_name)}Custom"

    # Check if kernel directory has required files
    if not kernel_dir.exists():
        print(f"  [SKIP] kernel/ directory not found")
        return False

    cpp_files = list(kernel_dir.glob("*.cpp"))
    if not cpp_files:
        print(f"  [SKIP] No .cpp files in kernel/")
        return False

    # Find main cpp file (not pybind11)
    main_cpp = None
    for cpp in cpp_files:
        if cpp.name != "pybind11.cpp":
            main_cpp = cpp
            break

    if main_cpp is None:
        main_cpp = cpp_files[0]

    # Find kernel header
    kernel_h_files = list(kernel_dir.glob("*_kernel.h"))
    kernel_h = kernel_h_files[0] if kernel_h_files else None

    # Find common header
    common_h = kernel_dir / "kernel_common.h"

    # Find pybind11
    pybind11_cpp = kernel_dir / "pybind11.cpp"

    # Create Custom directory
    custom_dir.mkdir(exist_ok=True)

    # Copy files
    if main_cpp:
        shutil.copy2(main_cpp, custom_dir / main_cpp.name)
        print(f"  [OK] Copied {main_cpp.name}")

    if kernel_h:
        shutil.copy2(kernel_h, custom_dir / kernel_h.name)
        print(f"  [OK] Copied {kernel_h.name}")

    if common_h.exists():
        shutil.copy2(common_h, custom_dir / common_h.name)
        print(f"  [OK] Copied {common_h.name}")

    if pybind11_cpp.exists():
        shutil.copy2(pybind11_cpp, custom_dir / pybind11_cpp.name)
        print(f"  [OK] Copied {pybind11_cpp.name}")

    # Copy any additional header files
    for h_file in kernel_dir.glob("*.h"):
        if kernel_h and h_file.name == kernel_h.name:
            continue
        if h_file.name == common_h.name:
            continue
        shutil.copy2(h_file, custom_dir / h_file.name)
        print(f"  [OK] Copied {h_file.name}")

    # Create build.sh script
    build_sh = custom_dir / "build.sh"
    build_script = f'''#!/bin/bash
# Build script for {op_name}Custom

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_NAME="{op_name}"

echo "Building $KERNEL_NAME AscendC kernel..."

# Source CANN environment if available
if [ -n "$ASCEND_HOME_PATH" ]; then
    source $ASCEND_HOME_PATH/bin/set_env.sh 2>/dev/null || true
fi

# Create build directory
mkdir -p build
cd build

# Run cmake
cmake .. -DKERNEL_NAME=$KERNEL_NAME
make -j$(nproc)

echo "Build complete. Output in build/ directory."
'''
    build_sh.write_text(build_script, encoding="utf-8")
    build_sh.chmod(0o755)
    print(f"  [OK] Created build.sh")

    # Create CMakeLists.txt
    cmake_txt = custom_dir / "CMakeLists.txt"
    cmake_content = f'''cmake_minimum_required(VERSION 3.14)
project({op_name}_custom)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${{TORCH_INCLUDE_DIRS}})
include_directories(${{Python_INCLUDE_DIRS}})

# Kernel source files
set(KERNEL_SOURCES
    {main_cpp.name if main_cpp else ""}
    {pybind11_cpp.name if pybind11_cpp.exists() else ""}
)

# Add pybind11 module
pybind11_add_module(_ext_cpp ${{KERNEL_SOURCES}})

target_link_libraries(_ext_cpp PRIVATE ${{TORCH_LIBRARIES}})
'''
    cmake_txt.write_text(cmake_content, encoding="utf-8")
    print(f"  [OK] Created CMakeLists.txt")

    return True


def main():
    print(f"Processing operators in {OUTPUT_DIR}")
    print("=" * 60)

    results = []

    for op_name in OPERATORS:
        op_dir = OUTPUT_DIR / op_name
        if not op_dir.exists():
            print(f"\n{op_name}: [SKIP] Directory not found")
            results.append((op_name, "skip", "Directory not found"))
            continue

        print(f"\n{op_name}:")

        # Create DSL file
        dsl_ok = create_dsl_file(op_dir, op_name)

        # Create Custom directory
        custom_ok = create_custom_dir(op_dir, op_name)

        if dsl_ok and custom_ok:
            results.append((op_name, "ok", ""))
        elif dsl_ok and not custom_ok:
            results.append((op_name, "partial", "DSL created, Custom dir incomplete"))
        elif not dsl_ok and custom_ok:
            results.append((op_name, "partial", "Custom dir created, DSL incomplete"))
        else:
            results.append((op_name, "fail", "Both failed"))

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    ok_count = 0
    partial_count = 0
    fail_count = 0
    skip_count = 0

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

    print(f"\nTotal: {len(results)} operators")
    print(f"  OK: {ok_count}")
    print(f"  Partial: {partial_count}")
    print(f"  Fail: {fail_count}")
    print(f"  Skip: {skip_count}")


if __name__ == "__main__":
    main()