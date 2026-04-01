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

#include <csi_to_bayer_gpu_resident/csi_to_bayer_gpu_resident.hpp>
#include "csi_to_bayer_kernels.hpp"

#include <stdexcept>

#include <fmt/format.h>

#include <hololink/common/cuda_helper.hpp>

namespace hololink::operators {

void CsiToBayerGpuResidentOp::setup(holoscan::OperatorSpec& spec) {
  // initially declare input and output buffer size to zero
  // in initialize(), the buffer size will be set to the actual CSI length and frame size
  spec.device_input("in", 0);
  spec.device_output("out", 0);
}

void CsiToBayerGpuResidentOp::initialize() {
  HOLOSCAN_LOG_DEBUG("CsiToBayerGpuResidentOp::initialize() called");
  holoscan::GPUResidentOperator::initialize();

  if (!configured_) {
    throw std::runtime_error("CsiToBayerGpuResidentOp is not configured.");
  }

  if (pixel_width_ == 0 || pixel_height_ == 0) {
    throw std::runtime_error("CsiToBayerGpuResidentOp has invalid image dimensions.");
  }

  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

  hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
  cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
      csi_to_bayer_kernel_source,
      {"frameReconstruction8", "frameReconstruction10", "frameReconstruction12"}));

  const size_t input_size = csi_length_;
  const size_t output_size = static_cast<size_t>(pixel_width_) * pixel_height_ * sizeof(uint16_t);
  spec()->device_input("in", input_size);
  spec()->device_output("out", output_size);
  HOLOSCAN_LOG_DEBUG(
      "CsiToBayerGpuResidentOp::initialize() completed, input_size={}, output_size={}",
      input_size,
      output_size);
}

void CsiToBayerGpuResidentOp::stop() {
  if (!cuda_context_) {
    return;
  }
  hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
  cuda_function_launcher_.reset();
  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

void CsiToBayerGpuResidentOp::compute(holoscan::InputContext&, holoscan::OutputContext&,
                                      holoscan::ExecutionContext&) {
  HOLOSCAN_LOG_DEBUG("CsiToBayerGpuResidentOp::compute() called");
  hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

  void* input_ptr = device_memory("in");
  void* output_ptr = device_memory("out");
  if (input_ptr == nullptr || output_ptr == nullptr) {
    throw std::runtime_error("CsiToBayerGpuResidentOp device memory is not available.");
  }

  auto stream_ptr = cuda_stream();
  if (!stream_ptr) {
    throw std::runtime_error("CsiToBayerGpuResidentOp CUDA stream is not available.");
  }

  CUstream stream = reinterpret_cast<CUstream>(*stream_ptr);
  auto* input_bytes = reinterpret_cast<unsigned char*>(input_ptr) + start_byte_;

  switch (pixel_format_) {
    case hololink::csi::PixelFormat::RAW_8:
      cuda_function_launcher_->launch("frameReconstruction8",
                                      {pixel_width_, pixel_height_, 1},
                                      stream,
                                      output_ptr,
                                      input_bytes,
                                      bytes_per_line_,
                                      pixel_width_,
                                      pixel_height_);
      break;
    case hololink::csi::PixelFormat::RAW_10:
      cuda_function_launcher_->launch("frameReconstruction10",
                                      {pixel_width_ / 4, pixel_height_, 1},
                                      stream,
                                      output_ptr,
                                      input_bytes,
                                      bytes_per_line_,
                                      pixel_width_ / 4,
                                      pixel_height_);
      break;
    case hololink::csi::PixelFormat::RAW_12:
      cuda_function_launcher_->launch("frameReconstruction12",
                                      {pixel_width_ / 2, pixel_height_, 1},
                                      stream,
                                      output_ptr,
                                      input_bytes,
                                      bytes_per_line_,
                                      pixel_width_ / 2,
                                      pixel_height_);
      break;
    default:
      throw std::runtime_error("Unsupported pixel format.");
  }
}

}  // namespace hololink::operators
