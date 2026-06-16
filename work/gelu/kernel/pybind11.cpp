#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "gelu_tiling.h"

extern "C" void gelu_do_fp32(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*);
extern "C" void gelu_do_fp16(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*);

static int32_t selectBlockM(int32_t M) {
    if (M <= 1024) return 64;
    if (M <= 4096) return 64;
    return 128;
}

namespace gelu_ext {
using LF = void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*);

pybind11::tuple run_gelu(const at::Tensor &x) {
    TORCH_CHECK(x.dim() == 2, "x must be [M,N]");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
                "unsupported dtype");
    TORCH_CHECK(x.is_contiguous(), "must be contiguous");

    const int32_t M = static_cast<int32_t>(x.sizes()[0]);
    const int32_t N = static_cast<int32_t>(x.sizes()[1]);

    int32_t blkM = selectBlockM(M);
    int32_t tileN = (N <= TILE_N_MAX) ? N : TILE_N_MAX;
    int32_t nTiles = (N + tileN - 1) / tileN;

    int32_t blkCount = (M + blkM - 1) / blkM;
    int32_t uc = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, blkCount);
    int32_t tpc = (blkCount + uc - 1) / uc;

    at::Tensor y = at::empty_like(x);
    at::Tensor tc = at::empty(
        {static_cast<long>(sizeof(GeluKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *t = reinterpret_cast<GeluKernelTiling*>(tc.data_ptr());
    t->M = M;
    t->N = N;
    t->blockM = blkM;
    t->usedCoreNum = uc;
    t->tasksPerCore = tpc;
    t->tileN = tileN;
    t->nTiles = nTiles;

    auto tn = tc.to(at::kPrivateUse1);
    auto stm = c10_npu::getCurrentNPUStream().stream(false);

    LF l = nullptr;
    if (x.scalar_type() == at::kFloat) l = gelu_do_fp32;
    else if (x.scalar_type() == at::kHalf) l = gelu_do_fp16;
    else TORCH_CHECK(false, "unsupported");

    l(uc, stm,
      static_cast<uint8_t*>(const_cast<void*>(x.storage().data())),
      static_cast<uint8_t*>(const_cast<void*>(y.storage().data())),
      static_cast<uint8_t*>(const_cast<void*>(tn.storage().data())));

    return pybind11::make_tuple(y);
}
}

PYBIND11_MODULE(_gelu_ext, m) {
    m.def("run_gelu", &gelu_ext::run_gelu, "");
}
