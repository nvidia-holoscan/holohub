/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOST_PRESENT_KERNELS_H
#define HOST_PRESENT_KERNELS_H

#include <cstdint>
#include <cuda_runtime.h>

void launch_fill_bars(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                      uint32_t frame, cudaStream_t stream);

void launch_fill_solid(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                       uint32_t color, cudaStream_t stream);

void launch_fill_gradient(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                          uint32_t frame, cudaStream_t stream);

#endif
