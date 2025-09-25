/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "frame_saver.hpp"

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <gxf/core/gxf.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/executor.hpp>
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

void FrameSaverOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input_frames");

  spec.param(output_dir_, "output_dir", "Output Directory",
             "Directory where frames will be saved", std::string("output_frames"));
  spec.param(base_filename_, "base_filename", "Base Filename",
             "Base name for saved frame files", std::string("frame_"));
  spec.param(save_as_raw_, "save_as_raw", "Save as Raw",
             "Whether to save frames as raw binary files", false);
}

void FrameSaverOp::initialize() {
  Operator::initialize();

  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_dir_.get());

  HOLOSCAN_LOG_INFO("FrameSaverOp initialized: saving to {}", output_dir_.get());
}

void FrameSaverOp::compute(holoscan::InputContext& op_input,
                          holoscan::OutputContext& op_output,
                          holoscan::ExecutionContext& context) {
  // Get input message
  auto input_message = op_input.receive<holoscan::gxf::Entity>("input_frames");
  if (!input_message) {
    HOLOSCAN_LOG_ERROR("No input message received");
    return;
  }

  // Get tensor from message
  auto tensor = input_message.value().get<holoscan::Tensor>();
  if (!tensor) {
    HOLOSCAN_LOG_ERROR("No tensor in input message");
    return;
  }

  // Generate filename
  std::stringstream ss;
  ss << output_dir_.get() << "/" << base_filename_.get()
     << std::setw(6) << std::setfill('0') << frame_count_++;

  if (save_as_raw_.get()) {
    ss << ".raw";
  } else {
    ss << ".bgr";
  }

  current_file_ = ss.str();

  try {
    // Open output file
    output_file_.open(current_file_, std::ios::binary);
    if (!output_file_.is_open()) {
      HOLOSCAN_LOG_ERROR("Failed to open output file: {}", current_file_);
      return;
    }

    // Get tensor data
    const uint8_t* data = static_cast<const uint8_t*>(tensor->data());
    size_t data_size = tensor->nbytes();

    // Add debug logging for data analysis
    bool all_zeros = true;
    int non_zero_count = 0;
    for (size_t i = 0; i < std::min(data_size, size_t(100)); i++) {
      if (data[i] != 0) {
        all_zeros = false;
        non_zero_count++;
      }
    }
    HOLOSCAN_LOG_INFO("Frame {} data analysis: size={}, all_zeros={}, non_zero_count={}",
                     frame_count_ - 1, data_size, all_zeros, non_zero_count);

    // Check if tensor is on GPU
    if (tensor->device().device_type == kDLCUDA) {
      // Allocate host memory
      std::vector<uint8_t> host_data(data_size);

      // Copy from GPU to CPU
      cudaError_t cuda_status = cudaMemcpy(host_data.data(), data, data_size, cudaMemcpyDeviceToHost);
      if (cuda_status != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("CUDA memcpy failed: {}", cudaGetErrorString(cuda_status));
        output_file_.close();
        return;
      }

      // Write to file
      output_file_.write(reinterpret_cast<const char*>(host_data.data()), data_size);
    } else {
      // Write directly from CPU memory
      output_file_.write(reinterpret_cast<const char*>(data), data_size);
    }

    output_file_.close();

    HOLOSCAN_LOG_INFO("Saved frame {} to {}", frame_count_ - 1, current_file_);

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error saving frame: {}", e.what());
    if (output_file_.is_open()) {
      output_file_.close();
    }
  }
}

} // namespace holoscan::ops
