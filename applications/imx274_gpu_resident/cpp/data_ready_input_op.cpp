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

#include "data_ready_input_op.hpp"
#include "data_ready_kernels.hpp"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>

#include <holoscan/logger/logger.hpp>
#include <holoscan/utils/cuda_macros.hpp>

namespace imx274_gpu_resident {

void DataReadyInputOp::setup(holoscan::OperatorSpec& spec) {
  (void)spec;
}

void DataReadyInputOp::initialize() {
  holoscan::GPUResidentOperator::initialize();

  if (!shared_state_) { throw std::runtime_error("DataReadyInputOp requires shared_state"); }

  constexpr int max_wait_seconds = 10;
  int waited_seconds = 0;
  while (!frame_memory_) {
    {
      std::lock_guard<std::mutex> lock(shared_state_->mutex);
      if (!shared_state_->get_frame_memory_base) {
        throw std::runtime_error("SharedFrameState::get_frame_memory_base not configured");
      }
      frame_memory_ = shared_state_->get_frame_memory_base();
      frame_size_rounded_ = shared_state_->frame_size_rounded;
    }
    if (!frame_memory_) {
      if (waited_seconds >= max_wait_seconds) {
        throw std::runtime_error(
            "Timeout waiting for frame memory base address from RoCE receiver");
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
      waited_seconds++;
    }
  }

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set CUDA device");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&chosen_frame_memory_, sizeof(unsigned char*)),
                                 "Failed to allocate chosen_frame_memory");

  std::lock_guard<std::mutex> lock(shared_state_->mutex);
  shared_state_->chosen_frame_memory = chosen_frame_memory_;
}

void DataReadyInputOp::stop() {
  if (chosen_frame_memory_) {
    cudaFree(chosen_frame_memory_);
    chosen_frame_memory_ = nullptr;
    std::lock_guard<std::mutex> lock(shared_state_->mutex);
    shared_state_->chosen_frame_memory = nullptr;
  }
}

void DataReadyInputOp::set_shared_state(std::shared_ptr<SharedFrameState> shared_state) {
  shared_state_ = std::move(shared_state);
}

void DataReadyInputOp::compute(holoscan::InputContext&, holoscan::OutputContext&,
                               holoscan::ExecutionContext&) {
  auto* data_ready_addr = static_cast<unsigned int*>(data_ready_device_address());
  if (!data_ready_addr || !shared_state_ || !frame_memory_) {
    throw std::runtime_error("DataReadyInputOp::compute() preconditions not met");
  }

  auto stream_ptr = data_ready_handler_cuda_stream();
  if (!stream_ptr) {
    throw std::runtime_error("Data ready handler CUDA stream is null");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_ptr);

  // Single-thread kernel: check frame metadata and pick the latest frame
  launch_receive_frame_gpu_resident(stream,
                                    (volatile void*)frame_memory_,
                                    static_cast<unsigned int>(frame_size_rounded_),
                                    chosen_frame_memory_,
                                    data_ready_addr);
}

}  // namespace imx274_gpu_resident
