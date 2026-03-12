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

#include <cuda_runtime.h>
#include <stdio.h>
#include "cuDispDevice.h"
#include "cuDispDeviceInternal.h"

extern "C" void cuDispLaunchPresentKernel(cudaStream_t stream, void** displayPtrLocations,
                                          void* swapchainGPU, void** displayPtrs_device,
                                          unsigned int* numBuffersPerLayer,
                                          unsigned int numLayers) {
  cuDispPresentImageKernel<<<1, 1, 0, stream>>>(
      displayPtrLocations, swapchainGPU, displayPtrs_device,
      numBuffersPerLayer, numLayers);
}

inline __device__ void cuDispGPUPresent(void** displayPtrLocations, void* swapchainGPU,
                                        void** displayPtrs_device,
                                        unsigned int* numBuffersPerLayer,
                                        unsigned int numLayers) {
  if (numBuffersPerLayer[0] == 0)
    return;

  static __device__ unsigned int bufNumToUse = 0;
  cuDispSwapchainGPU* sc = (cuDispSwapchainGPU*)swapchainGPU;
  unsigned int slotToNotify;

  {
    unsigned int next_slot = 0;
    if (sc->prevSlot != 0xffffffffU) {
      next_slot = (sc->prevSlot + 1) % CUDISP_PRESENT_NOTIFY_SLOTS;
    }
    sc->bufSlots[next_slot][0] = *((void**)displayPtrLocations[0]);
    slotToNotify = next_slot;
    sc->prevSlot = next_slot;

    sc->notifySlots[slotToNotify] = 1;
  }

  asm volatile("membar.sys;" ::: "memory");
  bufNumToUse++;
  bufNumToUse = bufNumToUse % numBuffersPerLayer[0];
  *((void**)displayPtrLocations[0]) = displayPtrs_device[bufNumToUse];
}

extern "C" __global__ void cuDispPresentImageKernel(void** displayPtrLocations,
                                                    void* swapchainGPU,
                                                    void** displayPtrs_device,
                                                    unsigned int* numBuffersPerLayer,
                                                    unsigned int numLayers) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    cuDispGPUPresent(displayPtrLocations, swapchainGPU, displayPtrs_device,
                     numBuffersPerLayer, numLayers);
  }
}
