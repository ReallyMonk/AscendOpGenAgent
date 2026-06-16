#pragma once
#ifndef GELU_KERNEL_H
#define GELU_KERNEL_H
#include <cmath>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "gelu_tiling.h"

template<typename T>
class GeluKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe) {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), tiling_.M * tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), tiling_.M * tiling_.N);
        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            uint32_t tileElems = tiling_.tileN;
            pipe_->InitBuffer(inBuf_, tileElems * sizeof(T));
            pipe_->InitBuffer(outBuf_, tileElems * sizeof(T));
            if constexpr (!std::is_same_v<T, float>) {
                pipe_->InitBuffer(castBuf_, tileElems * sizeof(float));
            }
            pipe_->InitBuffer(cBuf_, tileElems * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIV {
            const int ci = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            for (int li = 0; li < tiling_.tasksPerCore; ++li) {
                const int bx = ci * tiling_.tasksPerCore + li;
                if (bx >= BlockCount()) continue;
                for (int r = 0; r < subBlockRows_; ++r) {
                    const int ri = bx * tiling_.blockM +
                                   AscendC::GetSubBlockIdx() * subBlockRows_ + r;
                    if (ri < tiling_.M) ProcessRow(ri);
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline void ProcessRow(int ri) {
        for (int ti = 0; ti < tiling_.nTiles; ++ti) {
            uint32_t offset = ri * tiling_.N + ti * tiling_.tileN;
            uint32_t count = (ti == tiling_.nTiles - 1 && tiling_.N % tiling_.tileN != 0)
                             ? tiling_.N % tiling_.tileN : tiling_.tileN;

            auto xIn = inBuf_.Get<T>();
            auto yOut = outBuf_.Get<T>();

            AscendC::DataCopy(xIn, xGM_[offset], count);

            if constexpr (std::is_same_v<T, float>) {
                auto tmp = cBuf_.Get<float>();
                AscendC::Mul(tmp, xIn, xIn, count);
                AscendC::Mul(tmp, tmp, xIn, count);
                AscendC::Muls(tmp, tmp, 0.044715f, count);
                AscendC::Add(tmp, xIn, tmp, count);
                AscendC::Muls(tmp, tmp, 0.7978845608f, count);
                AscendC::Tanh(tmp, tmp, count);
                AscendC::Adds(tmp, tmp, 1.0f, count);
                AscendC::Mul(yOut, xIn, tmp, count);
                AscendC::Muls(yOut, yOut, 0.5f, count);
            } else {
                auto xFp32 = castBuf_.Get<float>();
                auto yFp32 = cBuf_.Get<float>();
                AscendC::Cast(xFp32, xIn, AscendC::RoundMode::CAST_NONE, count);
                AscendC::Mul(yFp32, xFp32, xFp32, count);
                AscendC::Mul(yFp32, yFp32, xFp32, count);
                AscendC::Muls(yFp32, yFp32, 0.044715f, count);
                AscendC::Add(yFp32, xFp32, yFp32, count);
                AscendC::Muls(yFp32, yFp32, 0.7978845608f, count);
                AscendC::Tanh(yFp32, yFp32, count);
                AscendC::Adds(yFp32, yFp32, 1.0f, count);
                AscendC::Mul(yFp32, xFp32, yFp32, count);
                AscendC::Muls(yFp32, yFp32, 0.5f, count);
                AscendC::Cast(yOut, yFp32, AscendC::RoundMode::CAST_NONE, count);
            }

            AscendC::DataCopy(yGM_[offset], yOut, count);
        }
    }

private:
    GeluKernelTiling tiling_{};
    AscendC::TPipe *pipe_{};
    int subBlockRows_{};
    AscendC::GlobalTensor<T> xGM_, yGM_;
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf_, cBuf_;
};
#endif
