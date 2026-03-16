/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDISP_DEVICE_H
#define CUDISP_DEVICE_H

#include <cuda_runtime.h>

/**
 * \file cuDispDevice.h
 * \brief Device-side API for cuDisp GPU present (flip notifications).
 *
 * Provides host-callable and device-callable functions for GPU-driven
 * present. The swapchain GPU structure is opaque (void*); its layout
 * is internal to the library.
 *
 * For each layer, the application supplies:
 * - A displayPtrLocation (pointer-to-pointer): the kernel reads the
 *   current buffer and updates it to the next buffer in the ring.
 * - A contiguous region of buffer pointers (displayPtrs_device) laid
 *   out as: [layer0 buf0, layer0 buf1, ..., layer1 buf0, ...].
 * - A numBuffersPerLayer array giving the buffer count for each layer.
 */

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Host-callable launcher for cuDispPresentImageKernel.
 * Must be called instead of launching the kernel directly from outside
 * the cuDisp library, because CUDA device code cannot be launched
 * across shared library boundaries.
 *
 * @param stream              CUDA stream to launch on
 * @param displayPtrLocations Array of numLayers pointers; each points to
 *                            the current buffer for that layer
 * @param swapchainGPU        Opaque GPU present handle (from GPU_PRESENT attribute)
 * @param displayPtrs_device  Flat array of buffer pointers for all layers
 *                            (layer 0 ring, then layer 1 ring, etc.)
 * @param numBuffersPerLayer  Array of numLayers entries; buffer count per layer
 * @param numLayers           Total number of layers
 */
void cuDispLaunchPresentKernel(cudaStream_t stream,
                               void** displayPtrLocations,
                               void* swapchainGPU,
                               void** displayPtrs_device,
                               unsigned int* numBuffersPerLayer,
                               unsigned int numLayers);

#if defined(__cplusplus)
}
#endif

#if defined(__CUDACC__)

/**
 * Device function: notify the host to present the current buffers for
 * all layers and advance each layer to its next buffer in the ring.
 *
 * For each layer L, the buffer at *displayPtrLocations[L] is queued
 * for display, then *displayPtrLocations[L] is updated to the next
 * buffer from the displayPtrs_device ring for that layer.
 *
 * @param displayPtrLocations Array of numLayers pointers (in: current buffer;
 *                            out: next buffer to render into)
 * @param swapchainGPU        Opaque GPU present handle
 * @param displayPtrs_device  Flat array of buffer pointers for all layers
 * @param numBuffersPerLayer  Array of numLayers entries; buffer count per layer
 * @param numLayers           Total number of layers
 */
__device__ void cuDispGPUPresent(void** displayPtrLocations,
                                 void* swapchainGPU,
                                 void** displayPtrs_device,
                                 unsigned int* numBuffersPerLayer,
                                 unsigned int numLayers);

/**
 * Kernel entry point: single-thread launch that calls cuDispGPUPresent.
 * Launch with <<<1, 1>>> (or one thread).
 */
extern "C" __global__ void cuDispPresentImageKernel(void** displayPtrLocations,
                                                    void* swapchainGPU,
                                                    void** displayPtrs_device,
                                                    unsigned int* numBuffersPerLayer,
                                                    unsigned int numLayers);

#endif /* __CUDACC__ */

#endif /* CUDISP_DEVICE_H */
