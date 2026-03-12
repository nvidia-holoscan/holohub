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

#include "data_ready_kernels.hpp"

#include <stdio.h>
#include <cstdint>

// ControlCommand enum values from holoscan/core/executors/gpu_resident/controlcommand.hpp
static constexpr unsigned int CTRL_DATA_NOT_READY = 1;
static constexpr unsigned int CTRL_DATA_READY = 2;
static constexpr unsigned int METADATA_SIZE = 128;

__global__ void receive_frame_gpu_resident_kernel(volatile void* frame_memory,
                                                  unsigned int frame_size,
                                                  unsigned char** chosen_frame_memory,
                                                  unsigned int* data_ready_ptr) {
  volatile unsigned char* frame_memory1 = (volatile unsigned char*)frame_memory;
  volatile unsigned char* metadata_memory = (volatile unsigned char*)(frame_memory1 + frame_size);

  volatile unsigned char* frame_memory2 =
      (volatile unsigned char*)(frame_memory1 + frame_size + METADATA_SIZE);
  volatile unsigned char* metadata_memory2 = (volatile unsigned char*)(frame_memory2 + frame_size);

  volatile uint32_t frame_number1 =
      ((uint32_t)metadata_memory[32] << 24) | ((uint32_t)metadata_memory[33] << 16) |
      ((uint32_t)metadata_memory[34] << 8) | ((uint32_t)metadata_memory[35]);

  volatile uint32_t frame_number2 =
      ((uint32_t)metadata_memory2[32] << 24) | ((uint32_t)metadata_memory2[33] << 16) |
      ((uint32_t)metadata_memory2[34] << 8) | ((uint32_t)metadata_memory2[35]);

  // Skip if same frame numbers as previous iteration
  static uint32_t prev_frame_number1 = 0xFFFFFFFF, prev_frame_number2 = 0xFFFFFFFF;
  if (prev_frame_number1 != 0xFFFFFFFF || prev_frame_number2 != 0xFFFFFFFF) {
    if (prev_frame_number1 == frame_number1 && prev_frame_number2 == frame_number2) {
      *data_ready_ptr = CTRL_DATA_NOT_READY;
      return;
    }
  }
  prev_frame_number1 = frame_number1;
  prev_frame_number2 = frame_number2;

  // Use signed difference to handle wrap-around correctly
  if ((int32_t)(frame_number1 - frame_number2) > 0) {
    *((volatile unsigned char**)chosen_frame_memory) = frame_memory1;
  } else {
    *((volatile unsigned char**)chosen_frame_memory) = frame_memory2;
  }

  *data_ready_ptr = CTRL_DATA_READY;
}

void launch_receive_frame_gpu_resident(cudaStream_t stream, volatile void* frame_memory,
                                       unsigned int frame_size, unsigned char** chosen_frame_memory,
                                       unsigned int* data_ready_ptr) {
  receive_frame_gpu_resident_kernel<<<1, 1, 0, stream>>>(
      frame_memory, frame_size, chosen_frame_memory, data_ready_ptr);
}
