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

#ifndef CUDISP_DEVICE_INTERNAL_H
#define CUDISP_DEVICE_INTERNAL_H

/**
 * \file cuDispDeviceInternal.h
 * \brief Internal GPU present structures. NOT part of the public API.
 *
 * This header is included only by the library's own .cpp and .cu files.
 * It must not be shipped to or included by application code. The layout
 * of cuDispSwapchainGPU and the values of CUDISP_PRESENT_NOTIFY_SLOTS
 * and CUDISP_MAX_LAYERS may change without notice; applications interact
 * with the GPU present handle as an opaque void*.
 */

#include <stdint.h>

#define CUDISP_PRESENT_NOTIFY_SLOTS 10
#define CUDISP_MAX_LAYERS 8

/**
 * GPU-visible structure for flip notifications.
 * Allocated with cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP) so it is
 * accessible from both host (present thread) and device (kernels).
 *
 * bufSlots[slot][layer]  — buffer pointer written by the kernel for
 *                          each layer before signalling.
 * notifySlots[slot]      — set to 1 by the kernel to signal the host
 *                          present thread that all layers in this slot
 *                          are ready to flip.
 * prevSlot               — ring index shared between the GPU (producer)
 *                          and the host present thread (consumer).
 *                          Initialized to 0xFFFFFFFF (no slot used yet).
 */
typedef struct {
    void*    bufSlots[CUDISP_PRESENT_NOTIFY_SLOTS][CUDISP_MAX_LAYERS];
    uint32_t notifySlots[CUDISP_PRESENT_NOTIFY_SLOTS];
    uint32_t prevSlot;
} cuDispSwapchainGPU;

#endif /* CUDISP_DEVICE_INTERNAL_H */
