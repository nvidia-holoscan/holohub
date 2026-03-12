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

#include <image_processor_gpu_resident/image_processor_gpu_resident.hpp>
#include "image_processor_kernel_source.hpp"

#include <stdexcept>

#include <fmt/format.h>

#include <hololink/core/csi_formats.hpp>
#include <holoscan/logger/logger.hpp>

namespace {

// 3 channels (RGB)
constexpr auto CHANNELS = 3;
// histogram bin's
constexpr auto HISTOGRAM_BIN_COUNT = 256;

}  // anonymous namespace

namespace hololink::operators {

void ImageProcessorGpuResidentOp::setup(holoscan::OperatorSpec& spec) {
  // allocate as device pointer
  spec.device_input("in", internal_compute_buffer_);
  spec.device_output("out", internal_compute_buffer_);

  spec.param(
      bayer_format_, "bayer_format", "BayerFormat", "Bayer format (hololink::csi::BayerFormat)");
  spec.param(
      pixel_format_, "pixel_format", "PixelFormat", "Pixel format (hololink::csi::PixelFormat)");
  spec.param(optical_black_, "optical_black", "OpticalBlack", "Optical black value", 0);
  spec.param(width_, "width", "Width", "Image width in pixels", 0);
  spec.param(height_, "height", "Height", "Image height in pixels", 0);
}

void ImageProcessorGpuResidentOp::initialize() {
  HOLOSCAN_LOG_DEBUG("ImageProcessorGpuResidentOp::initialize() called");
  holoscan::GPUResidentOperator::initialize();

  if (width_.get() <= 0 || height_.get() <= 0) {
    throw std::runtime_error("ImageProcessorGpuResidentOp has invalid width/height.");
  }

  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

  hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

  // histogram setup
  const auto log2_warp_size = 5;
  const auto warp_size = 1 << log2_warp_size;

  // size of histogram memory
  constexpr auto histogram_warp_memory = HISTOGRAM_BIN_COUNT * sizeof(uint32_t) * CHANNELS;
  histogram_memory_.reset([] {
    CUdeviceptr mem = 0;
    CudaCheck(cuMemAlloc(&mem, histogram_warp_memory));
    return mem;
  }());

  // calculate the maximum warp count supported by the available shared memory
  int shm_size = 0;
  CudaCheck(cuDeviceGetAttribute(
      &shm_size, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuda_device_));

  const auto histogram_warp_count = shm_size / histogram_warp_memory;
  const auto histogram_threadblock_memory = histogram_warp_memory * histogram_warp_count;
  histogram_threadblock_size_ = histogram_warp_count * warp_size;

  uint32_t least_significant_bit;
  switch (hololink::csi::PixelFormat(pixel_format_.get())) {
    case hololink::csi::PixelFormat::RAW_8:
      least_significant_bit = 0;
      break;
    case hololink::csi::PixelFormat::RAW_10:
      least_significant_bit = 16 - 10;
      break;
    case hololink::csi::PixelFormat::RAW_12:
      least_significant_bit = 16 - 12;
      break;
    default:
      throw std::runtime_error(
          fmt::format("Camera pixel format {} not supported.", int(pixel_format_.get())));
  }

  uint32_t x0y0_offset, x1y0_offset, x0y1_offset, x1y1_offset;
  switch (hololink::csi::BayerFormat(bayer_format_.get())) {
    case hololink::csi::BayerFormat::RGGB:
      x0y0_offset = 0;  // R
      x1y0_offset = 1;  // G
      x0y1_offset = 1;  // G
      x1y1_offset = 2;  // B
      break;
    case hololink::csi::BayerFormat::GBRG:
      x0y0_offset = 1;  // G
      x1y0_offset = 2;  // B
      x0y1_offset = 0;  // R
      x1y1_offset = 1;  // G
      break;
    case hololink::csi::BayerFormat::BGGR:
      x0y0_offset = 2;  // B
      x1y0_offset = 1;  // G
      x0y1_offset = 1;  // G
      x1y1_offset = 0;  // R
      break;
    case hololink::csi::BayerFormat::GRBG:
      x0y0_offset = 1;  // G
      x1y0_offset = 0;  // R
      x0y1_offset = 2;  // B
      x1y1_offset = 1;  // G
      break;
    default:
      throw std::runtime_error(
          fmt::format("Camera bayer format {} not supported.", int(bayer_format_.get())));
  }

  cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
      hololink::operators::image_processor::kernel_source,
      {"applyBlackLevel", "histogram", "calcWBGains", "applyOperations"},
      {fmt::format("-D CHANNELS={}", CHANNELS),
       fmt::format("-D X0Y0_OFFSET={}", x0y0_offset),
       fmt::format("-D X1Y0_OFFSET={}", x1y0_offset),
       fmt::format("-D X0Y1_OFFSET={}", x0y1_offset),
       fmt::format("-D X1Y1_OFFSET={}", x1y1_offset),
       fmt::format("-D HISTOGRAM_BIN_COUNT={}", HISTOGRAM_BIN_COUNT),
       fmt::format("-D LOG2_WARP_SIZE={}", log2_warp_size),
       fmt::format("-D HISTOGRAM_WARP_COUNT={}", histogram_warp_count),
       fmt::format("-D HISTOGRAM_THREADBLOCK_SIZE={}", histogram_threadblock_size_),
       fmt::format("-D HISTOGRAM_THREADBLOCK_MEMORY={}", histogram_threadblock_memory),
       fmt::format("-D OPTICAL_BLACK={}", optical_black_.get() * (1 << least_significant_bit))}));

  white_balance_gains_memory_.reset([] {
    CUdeviceptr mem = 0;
    CudaCheck(cuMemAlloc(&mem, CHANNELS * sizeof(float)));
    return mem;
  }());

  const size_t input_size = static_cast<size_t>(width_.get()) * height_.get() * sizeof(uint16_t);
  // allocate the internal buffer of the input size
  // the internal buffer is used for in-place computation and hence,
  // used for both input and output
  CudaCheck(cuMemAlloc(&internal_compute_buffer_, input_size));
  spec()->device_input("in", internal_compute_buffer_);
  spec()->device_output("out", internal_compute_buffer_);
  HOLOSCAN_LOG_DEBUG("ImageProcessorGpuResidentOp::initialize() completed, size={}", input_size);
}

void ImageProcessorGpuResidentOp::stop() {
  if (!cuda_context_) {
    return;
  }
  hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
  cuda_function_launcher_.reset();
  histogram_memory_.reset();
  white_balance_gains_memory_.reset();
  // free the internal compute buffer
  CudaCheck(cuMemFree(internal_compute_buffer_));
  internal_compute_buffer_ = 0;
  if (cuda_context_) {
    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
  }
}

void ImageProcessorGpuResidentOp::compute(holoscan::InputContext&, holoscan::OutputContext&,
                                          holoscan::ExecutionContext&) {
  HOLOSCAN_LOG_DEBUG("ImageProcessorGpuResidentOp::compute() called");
  hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

  void* input_ptr = device_memory("in");
  void* output_ptr = device_memory("out");
  if (input_ptr == nullptr || output_ptr == nullptr) {
    throw std::runtime_error("ImageProcessorGpuResidentOp device memory is not available.");
  }

  auto stream_ptr = cuda_stream();
  if (!stream_ptr) {
    throw std::runtime_error("ImageProcessorGpuResidentOp CUDA stream is not available.");
  }

  const int32_t width = width_.get();
  const int32_t height = height_.get();
  const size_t frame_bytes = static_cast<size_t>(width) * height * sizeof(uint16_t);
  CUstream stream = reinterpret_cast<CUstream>(*stream_ptr);

  // Copy input to output so we can process in-place on the output buffer.
  if (input_ptr != output_ptr) {
    throw std::runtime_error(
      "ImageProcessorGpuResidentOp device memory for input and output are separate.");
  }

  if (optical_black_.get() != 0) {
    cuda_function_launcher_->launch(
        "applyBlackLevel",
        {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        stream,
        output_ptr,
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height));
  }

  CudaCheck(cuMemsetD32Async(histogram_memory_.get(), 0, CHANNELS * HISTOGRAM_BIN_COUNT, stream));

  cuda_function_launcher_->launch("histogram",
                                  {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
                                  {histogram_threadblock_size_, 2, 1},
                                  stream,
                                  output_ptr,
                                  histogram_memory_.get(),
                                  static_cast<uint32_t>(width),
                                  static_cast<uint32_t>(height));

  cuda_function_launcher_->launch("calcWBGains",
                                  {1, 1, 1},
                                  {1, 1, 1},
                                  stream,
                                  histogram_memory_.get(),
                                  white_balance_gains_memory_.get());

  cuda_function_launcher_->launch("applyOperations",
                                  {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
                                  stream,
                                  output_ptr,
                                  width,
                                  height,
                                  white_balance_gains_memory_.get());
}

}  // namespace hololink::operators
