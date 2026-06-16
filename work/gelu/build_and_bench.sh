#!/bin/bash
# GELU kernel: build + correctness + benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/kernel"
cd "$KERNEL_DIR"

# --- Discover env ---
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_PATH="$HOME/Ascend/ascend-toolkit/latest"
    elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi

SOC_VERSION="${SOC_VERSION:-Ascend910B2C}"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-$ASCEND_HOME_PATH}"
export ASCEND_HOME_PATH SOC_VERSION ASCEND_CANN_PACKAGE_PATH

echo "=== Build ==="
rm -rf build
mkdir -p build
cmake -B build -S . \
    -DSOC_VERSION="$SOC_VERSION" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "=== Correctness Test ==="
python3 -c "
import sys
sys.path.insert(0, '$KERNEL_DIR/build')
import torch
import torch_npu
from _gelu_ext import run_gelu

# Test fp16
x = torch.randn(1024, 1024, dtype=torch.float16, device='npu')
ref = torch.nn.functional.gelu(x.float()).half()
y = run_gelu(x)[0]
max_err = (y.float() - ref).abs().max().item()
mean_err = (y.float() - ref).abs().mean().item()
print(f'fp16 1024x1024: max_err={max_err:.6f}, mean_err={mean_err:.6f}')
assert max_err < 0.01, f'fp16 max_err too large: {max_err}'

# Test fp32
xf = torch.randn(1024, 1024, dtype=torch.float32, device='npu')
ref32 = torch.nn.functional.gelu(xf)
yf = run_gelu(xf)[0]
max_err32 = (yf - ref32).abs().max().item()
print(f'fp32 1024x1024: max_err={max_err32:.8f}')
assert max_err32 < 0.001, f'fp32 max_err too large: {max_err32}'

print('Correctness PASSED')
"

echo "=== Benchmark ==="
python3 -c "
import sys, time
sys.path.insert(0, '$KERNEL_DIR/build')
import torch
import torch_npu
from _gelu_ext import run_gelu

def bench(label, fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - t0) / repeat * 1000
    print(f'  {label}: {elapsed:.3f} ms')

# Benchmark shape from model.py
M, N = 4096, 393216
x = torch.randn(M, N, dtype=torch.float16, device='npu')

bench('PyTorch gelu(fp16)', lambda: torch.nn.functional.gelu(x))
bench('AscendC gelu(fp16)', lambda: run_gelu(x))

xf = torch.randn(M, N, dtype=torch.float32, device='npu')
bench('PyTorch gelu(fp32)', lambda: torch.nn.functional.gelu(xf))
bench('AscendC gelu(fp32)', lambda: run_gelu(xf))
"
