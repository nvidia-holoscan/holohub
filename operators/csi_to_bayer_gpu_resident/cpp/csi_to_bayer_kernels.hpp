/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

namespace hololink::operators {

inline constexpr const char* csi_to_bayer_kernel_source = R"(
extern "C" {

__global__ void frameReconstruction8(unsigned short * out,
                                     const unsigned char * in,
                                     int per_line_size,
                                     int width,
                                     int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x;
    const int out_index = idx_y * width + idx_x;

    out[out_index] = in[in_index] << 8;
}

__global__ void frameReconstruction10(unsigned short * out,
                                      const unsigned char * in,
                                      int per_line_size,
                                      int quater_width,
                                      int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= quater_width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x * 5;
    const int out_index = (idx_y * quater_width + idx_x) * 4;

    const unsigned short lsbs = in[in_index + 4];
    out[out_index + 0] = ((in[in_index + 0] << 2) | (lsbs & 0x03)) << 6;
    out[out_index + 1] = ((in[in_index + 1] << 4) | (lsbs & 0x0C)) << 4;
    out[out_index + 2] = ((in[in_index + 2] << 6) | (lsbs & 0x30)) << 2;
    out[out_index + 3] = ((in[in_index + 3] << 8) | (lsbs & 0xC0));
}

__global__ void frameReconstruction12(unsigned short * out,
                                      const unsigned char * in,
                                      int per_line_size,
                                      int half_width,
                                      int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= half_width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x * 3;
    const int out_index = (idx_y * half_width + idx_x) * 2;

    const unsigned short lsbs = in[in_index + 2];
    out[out_index + 0] = ((in[in_index + 0] << 4) | (lsbs & 0x0F)) << 4;
    out[out_index + 1] = ((in[in_index + 1] << 8) | (lsbs & 0xF0));
}

})";

}  // namespace hololink::operators
