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

#include "display_kernels.hpp"

#include <stdexcept>

// Input: RGBA 16-bit (unsigned short per channel)
// Output: ARGB 8-bit display buffer (DRM_FORMAT_ARGB8888 byte order: B, G, R, A)
__global__ void displayImageRGBA16ToARGB8Kernel(const unsigned short* input, int width, int height,
                                                int channels, unsigned int displayWidth,
                                                unsigned int displayHeight,
                                                unsigned int displayNumChannels,
                                                void** displayPointerLocation,
                                                unsigned int displayPitchBytes) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  void* displayPtr = *displayPointerLocation;

  if (x < displayWidth && y < displayHeight) {
    unsigned int displayIdx = y * displayPitchBytes + x * displayNumChannels;

    if (x < width && y < height) {
      int idx = (y * width + x) * channels;

      const float range = (1 << (sizeof(unsigned short) * 8)) - 1;
      const float GAMMA = 2.2f;
      float value;

      if (channels >= 3) {
        value = (float)(input[idx + 0]);
        value = powf(value / range, 1.f / GAMMA) * range;
        unsigned char r8 = (unsigned char)((value / 257.0f) + 0.5f);

        value = (float)(input[idx + 1]);
        value = powf(value / range, 1.f / GAMMA) * range;
        unsigned char g8 = (unsigned char)((value / 257.0f) + 0.5f);

        value = (float)(input[idx + 2]);
        value = powf(value / range, 1.f / GAMMA) * range;
        unsigned char b8 = (unsigned char)((value / 257.0f) + 0.5f);

        unsigned char a8 = 0xFF;
        if (channels == 4) {
          a8 = (unsigned char)((((unsigned int)input[idx + 3]) * 255) / 65535);
        }

        ((unsigned char*)displayPtr)[displayIdx + 0] = b8;
        ((unsigned char*)displayPtr)[displayIdx + 1] = g8;
        ((unsigned char*)displayPtr)[displayIdx + 2] = r8;
        ((unsigned char*)displayPtr)[displayIdx + 3] = a8;
      } else if (channels == 1) {
        value = (float)(input[idx]);
        value = powf(value / range, 1.f / GAMMA) * range;
        unsigned char gray8 = (unsigned char)((value / 257.0f) + 0.5f);

        ((unsigned char*)displayPtr)[displayIdx + 0] = gray8;
        ((unsigned char*)displayPtr)[displayIdx + 1] = gray8;
        ((unsigned char*)displayPtr)[displayIdx + 2] = gray8;
        ((unsigned char*)displayPtr)[displayIdx + 3] = 0xFF;
      }
    } else {
      ((unsigned char*)displayPtr)[displayIdx + 0] = 0;
      ((unsigned char*)displayPtr)[displayIdx + 1] = 0;
      ((unsigned char*)displayPtr)[displayIdx + 2] = 0;
      ((unsigned char*)displayPtr)[displayIdx + 3] = 0xFF;
    }
  }
}

// Input: RGBA 16-bit (unsigned short per channel)
// Output: RGBA 16-bit display buffer (DRM_FORMAT_ABGR16161616)
__global__ void displayImageRGBA16ToRGBA16Kernel(const unsigned short* input, int width, int height,
                                                 int channels, unsigned int displayWidth,
                                                 unsigned int displayHeight,
                                                 unsigned int displayNumChannels,
                                                 void** displayPointerLocation,
                                                 unsigned int displayPitchBytes) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  void* displayPtr = *displayPointerLocation;

  if (x < displayWidth && y < displayHeight) {
    // Pitch is in bytes; convert to unsigned short units for 16-bit output
    unsigned int displayIdx =
        y * (displayPitchBytes / sizeof(unsigned short)) + x * displayNumChannels;

    if (x < width && y < height) {
      int idx = (y * width + x) * channels;

      ((unsigned short*)displayPtr)[displayIdx + 3] = 0;

      const float range = (1 << (sizeof(unsigned short) * 8)) - 1;
      const float GAMMA = 2.2f;
      float value;

      if (channels >= 3) {
        value = (float)(input[idx + 0]);
        value = powf(value / range, 1.f / GAMMA) * range;
        ((unsigned short*)displayPtr)[displayIdx + 0] = (unsigned short)(value + 0.5f);

        value = (float)(input[idx + 1]);
        value = powf(value / range, 1.f / GAMMA) * range;
        ((unsigned short*)displayPtr)[displayIdx + 1] = (unsigned short)(value + 0.5f);

        value = (float)(input[idx + 2]);
        value = powf(value / range, 1.f / GAMMA) * range;
        ((unsigned short*)displayPtr)[displayIdx + 2] = (unsigned short)(value + 0.5f);

        if (channels == 4) {
          ((unsigned short*)displayPtr)[displayIdx + 3] = input[idx + 3];
        }
      } else if (channels == 1) {
        value = (float)(input[idx]);
        value = powf(value / range, 1.f / GAMMA) * range;
        ((unsigned short*)displayPtr)[displayIdx + 0] = (unsigned short)(value + 0.5f);
        ((unsigned short*)displayPtr)[displayIdx + 1] = (unsigned short)(value + 0.5f);
        ((unsigned short*)displayPtr)[displayIdx + 2] = (unsigned short)(value + 0.5f);
      }
    }
  }
}

// Bilinear resize kernel for RGBA16 data.
// Uses center-pixel mapping and __ldg() cached reads for low-latency scaling.
__global__ void bilinearResizeRGBA16Kernel(
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    int src_width, int src_height, int channels,
    unsigned int dst_width, unsigned int dst_height,
    float x_scale, float y_scale) {
    const unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_width || dst_y >= dst_height) return;

    const float src_xf = (dst_x + 0.5f) * x_scale - 0.5f;
    const float src_yf = (dst_y + 0.5f) * y_scale - 0.5f;

    const float fx_floor = floorf(src_xf);
    const float fy_floor = floorf(src_yf);
    const float fx = src_xf - fx_floor;
    const float fy = src_yf - fy_floor;

    const int x0 = max(0, static_cast<int>(fx_floor));
    const int y0 = max(0, static_cast<int>(fy_floor));
    const int x1 = min(x0 + 1, src_width - 1);
    const int y1 = min(y0 + 1, src_height - 1);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const int idx00 = (y0 * src_width + x0) * channels;
    const int idx10 = (y0 * src_width + x1) * channels;
    const int idx01 = (y1 * src_width + x0) * channels;
    const int idx11 = (y1 * src_width + x1) * channels;
    const unsigned int dst_idx = (dst_y * dst_width + dst_x) * channels;

    for (int c = 0; c < channels; c++) {
        // __ldg() is used to load the data from a different global memory read-only path
        // this will hit a different read-only cache, with more hits as more threads are accessing 
        // the same data
        const float val = w00 * static_cast<float>(__ldg(&input[idx00 + c]))
                        + w10 * static_cast<float>(__ldg(&input[idx10 + c]))
                        + w01 * static_cast<float>(__ldg(&input[idx01 + c]))
                        + w11 * static_cast<float>(__ldg(&input[idx11 + c]));
        output[dst_idx + c] = static_cast<unsigned short>(val + 0.5f);
    }
}

// Host-callable launcher functions

void launch_display_image_gamma_corrected(cudaStream_t stream, const unsigned short* input,
                                          int width, int height, int channels,
                                          unsigned int display_width, unsigned int display_height,
                                          unsigned int display_channels,
                                          void** display_ptr_location, int surface_format,
                                          unsigned short* resize_buffer,
                                          unsigned int display_pitch_bytes) {
  constexpr unsigned int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((display_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (display_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

  const unsigned short* gamma_input = input;
  int gamma_width = width;
  int gamma_height = height;

  const bool needs_resize =
      (width != static_cast<int>(display_width) || height != static_cast<int>(display_height));
  if (needs_resize && resize_buffer) {
    const float x_scale = static_cast<float>(width) / static_cast<float>(display_width);
    const float y_scale = static_cast<float>(height) / static_cast<float>(display_height);

    bilinearResizeRGBA16Kernel<<<grid, block, 0, stream>>>(input,
                                                           resize_buffer,
                                                           width,
                                                           height,
                                                           channels,
                                                           display_width,
                                                           display_height,
                                                           x_scale,
                                                           y_scale);

    gamma_input = resize_buffer;
    gamma_width = static_cast<int>(display_width);
    gamma_height = static_cast<int>(display_height);
  }

  if (surface_format == 2) {
    displayImageRGBA16ToRGBA16Kernel<<<grid, block, 0, stream>>>(gamma_input,
                                                                 gamma_width,
                                                                 gamma_height,
                                                                 channels,
                                                                 display_width,
                                                                 display_height,
                                                                 display_channels,
                                                                 display_ptr_location,
                                                                 display_pitch_bytes);
  } else {
    displayImageRGBA16ToARGB8Kernel<<<grid, block, 0, stream>>>(gamma_input,
                                                                gamma_width,
                                                                gamma_height,
                                                                channels,
                                                                display_width,
                                                                display_height,
                                                                display_channels,
                                                                display_ptr_location,
                                                                display_pitch_bytes);
  }
}
