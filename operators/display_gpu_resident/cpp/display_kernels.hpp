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

#ifndef HOLOHUB_DISPLAY_GPU_RESIDENT_DISPLAY_KERNELS
#define HOLOHUB_DISPLAY_GPU_RESIDENT_DISPLAY_KERNELS

#include <cuda_runtime.h>

void launch_display_image_gamma_corrected(
    cudaStream_t stream,
    const unsigned short* input, int width, int height, int channels,
    unsigned int display_width, unsigned int display_height,
    unsigned int display_channels, void** display_ptr_location,
    int surface_format, unsigned short* resize_buffer,
    unsigned int display_pitch_bytes);

#endif /* HOLOHUB_DISPLAY_GPU_RESIDENT_DISPLAY_KERNELS */
