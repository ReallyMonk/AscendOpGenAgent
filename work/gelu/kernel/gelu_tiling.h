#pragma once
#ifndef GELU_TILING_H
#define GELU_TILING_H
#include <cstdint>

constexpr int32_t DEFAULT_BLOCK_M = 128;
constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 32;
constexpr int32_t TILE_N_MAX = 10240;

struct GeluKernelTiling {
    int32_t M;
    int32_t N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t tileN;
    int32_t nTiles;
};
#endif
