/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef STEREO_DEPTH_KERNELS_HPP
#define STEREO_DEPTH_KERNELS_HPP

#include <cuda.h>


void makeRectificationMap(float* M, float* d, float* R, float* P, float* mapx, float* mapy,
                          uint32_t width, uint32_t height, cudaStream_t stream);

void heatmapF32(float* grayscale, uint8_t* rgb, float min_val, float max_val, uint32_t width,
                uint32_t height, cudaStream_t stream);

void confidenceMask(float* disp, float* confidence, float thres, uint32_t width, uint32_t height,
                    cudaStream_t stream);

void preprocessESS(uint8_t* input, float* output, uint32_t input_width, uint32_t input_height,
                   uint32_t input_channels, uint32_t output_width, uint32_t output_height,
                   cudaStream_t stream);

#endif
