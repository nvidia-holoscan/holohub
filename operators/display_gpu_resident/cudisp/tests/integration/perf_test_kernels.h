/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PERF_TEST_KERNELS_H
#define PERF_TEST_KERNELS_H

#include <cstdint>
#include <cuda_runtime.h>

void launch_fill_flat(uint32_t* buf, uint32_t count, uint32_t color, cudaStream_t stream);

#endif
