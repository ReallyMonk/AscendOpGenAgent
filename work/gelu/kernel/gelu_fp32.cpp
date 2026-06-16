#include "gelu_kernel.h"
extern "C" __global__ __aicore__ void gelu_custom_fp32(GM_ADDR x, GM_ADDR y, GM_ADDR t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe p;
    GeluKernel<float> k; k.Init(x, y, t, &p); k.Process();
}
extern "C" void gelu_do_fp32(uint32_t d, void* s, uint8_t* x, uint8_t* y, uint8_t* t) {
    gelu_custom_fp32<<<d, nullptr, s>>>(x, y, t);
}
