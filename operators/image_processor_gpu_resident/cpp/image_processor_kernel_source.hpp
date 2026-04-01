/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOHUB_HOLOLINK_IMAGE_PROCESSOR_GPU_RESIDENT_IMAGE_PROCESSOR_KERNEL_SOURCE_HPP
#define HOLOHUB_HOLOLINK_IMAGE_PROCESSOR_GPU_RESIDENT_IMAGE_PROCESSOR_KERNEL_SOURCE_HPP

namespace hololink {
namespace operators {
namespace image_processor {

/** CUDA kernel source for black level, histogram, white balance gains, and apply operations. */
static const char kernel_source[] = R"(
#include <device_atomic_functions.h>
#include <cooperative_groups.h>

extern "C" {

// bayer component offsets
__inline__ __device__ unsigned int getBayerOffset(unsigned int x, unsigned int y)
{
    const unsigned int offsets[2][2]{{X0Y0_OFFSET, X1Y0_OFFSET}, {X0Y1_OFFSET, X1Y1_OFFSET}};
    return offsets[y & 1][x & 1];
}

/**
 * Apply black level correction.
 *
 * @param image [in] pointer to input image
 * @param components_per_line [in] components per input image line (width * 3 for RGB)
 * @param height [in] height of the input image
 */
__global__ void applyBlackLevel(unsigned short *image,
                                int components_per_line,
                                int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= components_per_line) || (idx_y >= height))
        return;

    const int index = idx_y * components_per_line + idx_x;

    // subtract optical black and clamp
    float value = max(float(image[index]) - OPTICAL_BLACK, 0.f);
    // fix white level
    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;
    value *= range / (range - float(OPTICAL_BLACK));
    image[index] = (unsigned short)(value + 0.5f);
}

/**
 * Calculate the histogram of an image.
 *
 * Based on the Cuda SDK histogram256 sample.
 *
 * First each warp of a thread builds a sub-histogram in shared memory. Then the per-warp
 * sub-histograms are merged per block and written to global memory using atomics.
 *
 * Note, this kernel needs HISTOGRAM_THREADBLOCK_MEMORY bytes of shared
 * memory.
 *
 * @param in [in] pointer to image data
 * @param histogram [in] pointer to the histogram data (must be able to hold HISTOGRAM_BIN_COUNT values)
 * @param width [in] width of the image
 * @param height [in] height of the image
 */
__global__ void histogram(const unsigned short *in,
                          unsigned int *histogram,
                          unsigned int width,
                          unsigned int height)
{
    uint2 index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (index.y >= height)
    {
        return;
    }

    // per-warp subhistogram storage
    __shared__ unsigned int s_hist[HISTOGRAM_THREADBLOCK_MEMORY / sizeof(unsigned int)];

    // clear shared memory storage for current threadblock before processing
    if (threadIdx.y == 0)
    {
#pragma unroll
        for (int i = 0; i < ((HISTOGRAM_THREADBLOCK_MEMORY / sizeof(unsigned int)) / HISTOGRAM_THREADBLOCK_SIZE); ++i)
        {
            s_hist[threadIdx.x + i * HISTOGRAM_THREADBLOCK_SIZE] = 0;
        }
    }

    // handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    cooperative_groups::sync(cta);

    // cycle through the entire data set, update subhistograms for each warp
    unsigned int *const s_warp_hist = s_hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM_BIN_COUNT * CHANNELS;
    while (index.x < width)
    {
        // take the upper 8 bits
        const unsigned char bin = ((unsigned char*)&in[index.y * width + index.x])[1];
        atomicAdd(s_warp_hist + bin + getBayerOffset(index.x, index.y) * HISTOGRAM_BIN_COUNT, 1u);
        index.x += blockDim.x * gridDim.x;
    }

    // Merge per-warp histograms into per-block and write to global memory
    cooperative_groups::sync(cta);

    if (threadIdx.y == 0)
    {
        for (int bin = threadIdx.x; bin < HISTOGRAM_BIN_COUNT * CHANNELS; bin += HISTOGRAM_THREADBLOCK_SIZE)
        {
            unsigned int sum = 0;

#pragma unroll
            for (int i = 0; i < HISTOGRAM_WARP_COUNT; ++i)
            {
                sum += s_hist[bin + i * HISTOGRAM_BIN_COUNT * CHANNELS];
            }

            atomicAdd(&histogram[bin], sum);
        }
    }
}

/**
 * Calculate the white balance gains using the per channel histograms
 *
 * @param histogram [in] pointer to histogram data (HISTOGRAM_BIN_COUNT * CHANNELS values)
 * @param gains [in] pointer to the white balance gains (CHANNELS values)
 */
__global__ void calcWBGains(const unsigned int *histogram,
                            float *gains)
{
    unsigned long long int average[CHANNELS];
    unsigned long long int max_gain = 0;
    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        unsigned long long int value = 0.f;
        for (int bin = 1; bin < HISTOGRAM_BIN_COUNT; ++bin)
        {
            value += histogram[channel * HISTOGRAM_BIN_COUNT + bin] * bin;
        }
        if (channel == 1)
        {
            // there are two green channels in the image which both are counted
            // in one histogram therefore divide green channel by 2
            value /= 2;
        }
        max_gain = max(max_gain, value);
        average[channel] = max(value, 1ull);
    }

    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        gains[channel] = float(max_gain) / float(average[channel]);
    }
}

/**
 * Apply white balance gains.
 *
 * @param in [in] pointer to image
 * @param width [in] width of the image
 * @param height [in] height of the image
 * @param gains [in] pointer to the white balance gains (CHANNELS values)
 */
__global__ void applyOperations(unsigned short *image,
                             int width,
                             int height,
                             const float *gains)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int index = idx_y * width + idx_x;

    float value = (float)(image[index]);

    // apply gain
    const unsigned int channel = getBayerOffset(idx_x, idx_y);
    value *= gains[channel];

    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;

    // clamp
    value = max(min(value, range), 0.f);

    image[index] = (unsigned short)(value + 0.5f);
}

})";

}  // namespace image_processor
}  // namespace operators
}  // namespace hololink

#endif  // HOLOHUB_HOLOLINK_IMAGE_PROCESSOR_GPU_RESIDENT_IMAGE_PROCESSOR_KERNEL_SOURCE_HPP
