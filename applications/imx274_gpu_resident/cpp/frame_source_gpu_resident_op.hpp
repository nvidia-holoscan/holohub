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

#ifndef HOLOHUB_IMX274_GPU_RESIDENT_FRAME_SOURCE_GPU_RESIDENT_OP
#define HOLOHUB_IMX274_GPU_RESIDENT_FRAME_SOURCE_GPU_RESIDENT_OP

#include <memory>
#include <stdexcept>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/logger/logger.hpp>

#include "copy_frame_kernels.hpp"
#include "shared_frame_state.hpp"

namespace imx274_gpu_resident {

class FrameSourceGPUResidentOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(FrameSourceGPUResidentOp,
                                       holoscan::GPUResidentOperator);

  void setup(holoscan::OperatorSpec& spec) override {
    spec.param(output_size_, "output_size", "OutputSize", "Output size in bytes", 0UL);
    spec.device_output("out", 0);
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("FrameSourceGPUResidentOp::initialize() called");
    holoscan::GPUResidentOperator::initialize();
    if (output_size_.get() == 0) {
      throw std::runtime_error("FrameSourceGPUResidentOp requires output_size.");
    }
    spec()->device_output("out", output_size_.get());
    HOLOSCAN_LOG_INFO("FrameSourceGPUResidentOp::initialize() completed, output_size={}",
                      output_size_.get());
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    unsigned char** chosen_frame_memory = nullptr;
    if (!shared_state_) {
      throw std::runtime_error("FrameSourceGPUResidentOp: shared_state is not set");
    }
    {
      std::lock_guard<std::mutex> lock(shared_state_->mutex);
      if (!shared_state_->chosen_frame_memory) {
        throw std::runtime_error("FrameSourceGPUResidentOp: chosen_frame_memory is not set");
      }

      chosen_frame_memory = shared_state_->chosen_frame_memory;
    }

    auto* output_ptr = device_memory("out");
    if (!output_ptr) {
      throw std::runtime_error("FrameSourceGPUResidentOp: output device memory is null");
    }

    auto stream_ptr = cuda_stream();
    if (!stream_ptr) {
      throw std::runtime_error("FrameSourceGPUResidentOp: CUDA stream is not available");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_ptr);

    launch_copy_frame(stream,
                      output_ptr,
                      chosen_frame_memory,
                      static_cast<unsigned int>(output_size_.get()));
  }

  void set_shared_state(std::shared_ptr<SharedFrameState> shared_state) {
    shared_state_ = std::move(shared_state);
  }

  void set_chosen_frame_memory(unsigned char** mem) {
    if (shared_state_) {
      std::lock_guard<std::mutex> lock(shared_state_->mutex);
      shared_state_->chosen_frame_memory = mem;
    }
  }

 private:
  holoscan::Parameter<size_t> output_size_;
  std::shared_ptr<SharedFrameState> shared_state_;
};

}  // namespace imx274_gpu_resident

#endif /* HOLOHUB_IMX274_GPU_RESIDENT_FRAME_SOURCE_GPU_RESIDENT_OP */
