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

#include <cuDispDevice.h>
#include <display_gpu_resident/display_gpu_resident.hpp>
#include "display_kernels.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include <holoscan/utils/cuda_macros.hpp>

namespace holoscan::ops {

void DisplayGpuResidentOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(width_, "width", "Width", "Image width", 0);
  spec.param(height_, "height", "Height", "Image height", 0);
  spec.param(out_channels_, "out_channels", "OutputChannels", "Output channels", 4);
  spec.param(element_size_, "element_size", "ElementSize", "Bytes per channel element", 2);
  spec.param(display_width_, "display_width", "DisplayWidth", "Display width in pixels", 2560);
  spec.param(display_height_, "display_height", "DisplayHeight", "Display height in pixels", 1440);
  spec.param(refresh_rate_,
             "refresh_rate",
             "RefreshRate",
             "Display refresh rate in milliHz (e.g., 60000 for 60Hz, 239970 for 239.97Hz), "
             "default: 0 (use the maximum available)",
             0);
  spec.param(surface_format_,
             "surface_format",
             "SurfaceFormat",
             "Display surface format: 0=A8R8G8B8_RGB (only supported value)",
             static_cast<int32_t>(DisplayOpSurfaceFormat::kDisplayOpSurfaceFormatA8R8G8B8));
  spec.param(gsync_, "gsync", "GSync", "Enable G-Sync / VRR", false);
  spec.param(front_buffer_rendering_,
             "front_buffer_rendering",
             "FrontBufferRendering",
             "Use front buffer rendering instead of continuous flip mode",
             true);
  spec.device_input("in", 0);
}

void DisplayGpuResidentOp::initialize() {
  holoscan::GPUResidentOperator::initialize();
  const int32_t input_width = width_.get();
  const int32_t input_height = height_.get();
  const bool is_supported_input_resolution =
      (input_width == 1920 && input_height == 1080) ||
      (input_width == 3840 && input_height == 2160);
  if (!is_supported_input_resolution) {
    throw std::runtime_error("DisplayGpuResidentOp: unsupported input resolution " +
                             std::to_string(input_width) + "x" +
                             std::to_string(input_height) +
                             ". Supported input resolutions are 1920x1080 (1080p) and "
                             "3840x2160 (4K).");
  }

  const size_t size =
      static_cast<size_t>(width_.get()) * height_.get() * out_channels_.get() * element_size_.get();
  spec()->device_input("in", size);

  const int32_t display_width = display_width_.get();
  const int32_t display_height = display_height_.get();
  const bool is_supported_resolution =
      (display_width == 2560 && display_height == 1440) ||
      (display_width == 1920 && display_height == 1080);
  if (!is_supported_resolution) {
    throw std::runtime_error("DisplayGpuResidentOp: unsupported display resolution " +
                             std::to_string(display_width) + "x" +
                             std::to_string(display_height) +
                             ". Supported resolutions are 2560x1440 and 1920x1080.");
  }

  try {
    if (input_width != display_width || input_height != display_height) {
      const size_t resize_buf_size = static_cast<size_t>(display_width) * display_height *
                                     out_channels_.get() * sizeof(unsigned short);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&resize_buffer_, resize_buf_size),
                                     "Failed to allocate resize buffer");
      HOLOSCAN_LOG_INFO(
          "DisplayGpuResidentOp: resize buffer allocated for {}x{} -> {}x{} ({} bytes)",
          input_width,
          input_height,
          display_width,
          display_height,
          resize_buf_size);
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set CUDA device");

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaStreamCreateWithFlags(&display_init_stream_, cudaStreamNonBlocking),
        "Failed to create display init stream");

    initialize_backend();

    if (!display_ptr0_) {
      throw std::runtime_error(
          "DisplayGpuResidentOp: display backend did not provide display buffer 0");
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemset(display_ptr0_, 0, static_cast<size_t>(buffer_size_)),
                                   "Failed to clear display buffer 0");
    if (display_ptr1_) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemset(display_ptr1_, 0, static_cast<size_t>(buffer_size_)),
          "Failed to clear display buffer 1");
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&display_ptr_location_ptr_, sizeof(void*)),
                                   "Failed to allocate display pointer location");
    // - initial CPU present uses buffer0
    // - first render target pointer starts at buffer1 (if available)
    void* initial_ptr =
        (!front_buffer_rendering_.get() && display_ptr1_) ? display_ptr1_ : display_ptr0_;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(display_ptr_location_ptr_, &initial_ptr, sizeof(void*), cudaMemcpyHostToDevice),
        "Failed to initialize display pointer location");
    HOLOSCAN_LOG_DEBUG(
        "DisplayGpuResidentOp: pointer init: ptr0={}, ptr1={}, initial render ptr={}",
        display_ptr0_,
        display_ptr1_,
        initial_ptr);

    if (!front_buffer_rendering_.get()) {
      // Allocate device-side array of per-layer buffer pointer locations
      cudaError_t cuda_err = cudaMalloc(&display_ptr_locations_device_, 1 * sizeof(void*));
      if (cuda_err != cudaSuccess) {
        throw std::runtime_error(
            "DisplayGpuResidentOp: cudaMalloc for display_ptr_locations_device_ failed");
      }
      void* host_locations[1] = {display_ptr_location_ptr_};
      cuda_err = cudaMemcpy(
          display_ptr_locations_device_, host_locations, 1 * sizeof(void*), cudaMemcpyHostToDevice);
      if (cuda_err != cudaSuccess) {
        throw std::runtime_error(
            "DisplayGpuResidentOp: cudaMemcpy for display_ptr_locations_device_ failed");
      }

      // Allocate device-side numBuffersPerLayer array
      cuda_err = cudaMalloc(&num_buffers_per_layer_device_, 1 * sizeof(unsigned int));
      if (cuda_err != cudaSuccess) {
        throw std::runtime_error(
            "DisplayGpuResidentOp: cudaMalloc for num_buffers_per_layer_device_ failed");
      }
      unsigned int host_num_bufs[1] = {NUM_BUFFERS};
      cuda_err = cudaMemcpy(num_buffers_per_layer_device_,
                            host_num_bufs,
                            1 * sizeof(unsigned int),
                            cudaMemcpyHostToDevice);
      if (cuda_err != cudaSuccess) {
        throw std::runtime_error(
            "DisplayGpuResidentOp: cudaMemcpy for num_buffers_per_layer_device_ failed");
      }
    }

    // High-level API only exposes one supported format.
    if (surface_format_.get() != kDisplayOpSurfaceFormatA8R8G8B8) {
      throw std::runtime_error("DisplayGpuResidentOp: expected surface_format=0 (A8R8G8B8_RGB)");
    }

    CUdeviceptr initial_present_device_ptr = display_device_ptr0_;
    HOLOSCAN_LOG_DEBUG("DisplayGpuResidentOp: initial CPU present buffer dptr=0x{:x}",
                       static_cast<uint64_t>(initial_present_device_ptr));
    cuDispBufferMemory present_buf = {.devicePtr = &initial_present_device_ptr,
                                      .size = nullptr,
                                      .stride = nullptr,
                                      .pHDRMetadata = nullptr};
    cuDispStatus present_err = cuDispPresent(swapchain_, display_init_stream_, &present_buf, 1, 0);
    if (present_err != cuDispSuccess) {
      throw std::runtime_error("DisplayGpuResidentOp: initial cuDispPresent failed with error " +
                               std::to_string(static_cast<int>(present_err)));
    }
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(display_init_stream_),
                                   "Failed to synchronize display init stream");

    // wait for display hardware to initialize
    std::this_thread::sleep_for(std::chrono::seconds(3));

    initialized_ = true;
  } catch (...) {
    try {
      stop();
    } catch (...) {
      // ignore any exceptions thrown from stop()
    }
    throw;  // rethrows the original exception
  }
}

void DisplayGpuResidentOp::initialize_backend() {
  unsigned int attribute_size = front_buffer_rendering_.get() ? 3 : 4;
  cuDispCreateAttribute attributes[4];
  std::memset(attributes, 0, sizeof(attributes));

  attributes[0].id = CUDISP_CREATE_ATTRIBUTE_MODE_INFO;
  attributes[0].value.modeInfo.modeWidth = static_cast<uint32_t>(display_width_.get());
  attributes[0].value.modeInfo.modeHeight = static_cast<uint32_t>(display_height_.get());
  attributes[0].value.modeInfo.refreshRateMilliHz = static_cast<uint32_t>(refresh_rate_.get());
  attributes[0].value.modeInfo.enableVrr = gsync_.get() ? 1 : 0;
  attributes[0].value.modeInfo.maxBpc = CUDISP_MAX_BPC_DEFAULT;

  attributes[1].id = CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO;
  attributes[1].value.bufferInfo.layerIndex = 0;
  attributes[1].value.bufferInfo.numBuffers = (front_buffer_rendering_.get() ? 1 : NUM_BUFFERS);
  attributes[1].value.bufferInfo.format = CUDISP_SURFACE_FORMAT_ARGB8888;
  attributes[1].value.bufferInfo.width = static_cast<uint32_t>(display_width_.get());
  attributes[1].value.bufferInfo.height = static_cast<uint32_t>(display_height_.get());
  attributes[1].value.bufferInfo.posX = 0;
  attributes[1].value.bufferInfo.posY = 0;
  attributes[1].value.bufferInfo.scaleWidth = 0;
  attributes[1].value.bufferInfo.scaleHeight = 0;
  attributes[1].value.bufferInfo.alpha = 0xFFFF;
  attributes[1].value.bufferInfo.blendMode = CUDISP_BLEND_MODE_DEFAULT;
  attributes[1].value.bufferInfo.rotation = CUDISP_ROTATE_0;
  attributes[1].value.bufferInfo.colorEncoding = CUDISP_COLOR_ENCODING_DEFAULT;
  attributes[1].value.bufferInfo.colorRange = CUDISP_COLOR_RANGE_DEFAULT;

  attributes[2].id = CUDISP_CREATE_ATTRIBUTE_IGNORE;

  if (!front_buffer_rendering_.get()) {
    attributes[2].id = CUDISP_CREATE_ATTRIBUTE_GPU_PRESENT;
    attributes[2].value.gpuPresent.handleGPUPresent = &present_handle_;

    attributes[3].id = CUDISP_CREATE_ATTRIBUTE_IGNORE;
  }

  cuDispStatus err = cuDispCreateSwapchain(&swapchain_, attributes, attribute_size, 0);
  if (err != cuDispSuccess) {
    stop();
    throw std::runtime_error("DisplayGpuResidentOp: cuDispCreateSwapchain failed with error " +
                             std::to_string(static_cast<int>(err)));
  }

  cuDispBufferMemory buf0 = {
      .devicePtr = &display_device_ptr0_,
      .size = &buffer_size_,
      .stride = &display_stride_,
      .pHDRMetadata = nullptr};
  err = cuDispGetBuffer(swapchain_, 0, 0, &buf0, 0);
  if (err != cuDispSuccess) {
    stop();
    throw std::runtime_error("DisplayGpuResidentOp: cuDispGetBuffer(0) failed with error " +
                             std::to_string(static_cast<int>(err)));
  }
  display_ptr0_ = reinterpret_cast<void*>(display_device_ptr0_);
  HOLOSCAN_LOG_INFO("DisplayGpuResidentOp: buffer0 stride={} bytes (width*bpp={})",
                    display_stride_,
                    display_width_.get() * 4);

  if (!front_buffer_rendering_.get()) {
    cuDispBufferMemory buf1 = {
        .devicePtr = &display_device_ptr1_,
        .size = nullptr,
        .stride = nullptr,
        .pHDRMetadata = nullptr};
    err = cuDispGetBuffer(swapchain_, 0, 1, &buf1, 0);
    if (err != cuDispSuccess) {
      stop();
      throw std::runtime_error("DisplayGpuResidentOp: cuDispGetBuffer(1) failed with error " +
                               std::to_string(static_cast<int>(err)));
    }
    display_ptr1_ = reinterpret_cast<void*>(display_device_ptr1_);

    cudaError_t cuda_err = cudaMalloc(&display_ptrs_device_, NUM_BUFFERS * sizeof(void*));
    if (cuda_err != cudaSuccess) {
      stop();
      throw std::runtime_error("DisplayGpuResidentOp: cudaMalloc for display_ptrs_device_ failed");
    }
    void* host_ptrs[NUM_BUFFERS] = {display_ptr0_, display_ptr1_};
    cuda_err = cudaMemcpy(
        display_ptrs_device_, host_ptrs, NUM_BUFFERS * sizeof(void*), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
      stop();
      throw std::runtime_error("DisplayGpuResidentOp: cudaMemcpy for display_ptrs_device_ failed");
    }
  }

  HOLOSCAN_LOG_INFO(
      "DisplayGpuResidentOp: cuDisp swapchain created ({}x{}, format={}, refresh={} milliHz, "
      "gsync={}, fbr={})",
      display_width_.get(),
      display_height_.get(),
      surface_format_.get(),
      refresh_rate_.get(),
      gsync_.get(),
      front_buffer_rendering_.get());
}

void DisplayGpuResidentOp::cleanup_cudisp() {
  if (display_ptr_location_ptr_ != nullptr) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(display_ptr_location_ptr_),
                                   "Failed to free display_ptr_location_ptr_");
    display_ptr_location_ptr_ = nullptr;
  }
  if (display_ptrs_device_ != nullptr) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(display_ptrs_device_),
                                   "Failed to free display_ptrs_device_");
    display_ptrs_device_ = nullptr;
  }
  if (display_ptr_locations_device_ != nullptr) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(display_ptr_locations_device_),
                                   "Failed to free display_ptr_locations_device_");
    display_ptr_locations_device_ = nullptr;
  }
  if (num_buffers_per_layer_device_ != nullptr) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(num_buffers_per_layer_device_),
                                   "Failed to free num_buffers_per_layer_device_");
    num_buffers_per_layer_device_ = nullptr;
  }
  if (swapchain_ != nullptr) {
    cuDispDestroySwapchain(swapchain_);
    swapchain_ = nullptr;
  }
  display_ptr0_ = nullptr;
  display_ptr1_ = nullptr;
  present_handle_ = nullptr;
  display_device_ptr0_ = 0;
  display_device_ptr1_ = 0;
}

void DisplayGpuResidentOp::stop() {
  HOLOSCAN_LOG_DEBUG("DisplayGpuResidentOp::stop()");
  if (display_init_stream_) {
    cudaStreamDestroy(display_init_stream_);
    display_init_stream_ = nullptr;
  }
  if (resize_buffer_) {
    HOLOSCAN_CUDA_CALL_WARN(cudaFree(resize_buffer_));
    resize_buffer_ = nullptr;
  }

  cleanup_cudisp();

  initialized_ = false;
}

void DisplayGpuResidentOp::compute(holoscan::InputContext&, holoscan::OutputContext&,
                                   holoscan::ExecutionContext&) {
  if (!initialized_) {
    throw std::runtime_error("DisplayGpuResidentOp::compute() called before initialize()");
  }

  auto* input_ptr = device_memory("in");
  if (!input_ptr) {
    throw std::runtime_error("DisplayGpuResidentOp::compute() device memory 'in' is null");
  }

  auto stream_ptr = cuda_stream();
  if (!stream_ptr) {
    throw std::runtime_error("DisplayGpuResidentOp::compute() CUDA stream is not available");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_ptr);

  launch_display_image_gamma_corrected(stream,
                                       static_cast<const unsigned short*>(input_ptr),
                                       width_.get(),
                                       height_.get(),
                                       out_channels_.get(),
                                       static_cast<unsigned int>(display_width_.get()),
                                       static_cast<unsigned int>(display_height_.get()),
                                       4,
                                       display_ptr_location_ptr_,
                                       surface_format_.get(),
                                       resize_buffer_,
                                       display_stride_);

  if (!front_buffer_rendering_.get()) {
    cuDispLaunchPresentKernel(stream,
                              static_cast<void**>(display_ptr_locations_device_),
                              present_handle_,
                              static_cast<void**>(display_ptrs_device_),
                              num_buffers_per_layer_device_,
                              1u);
  }
  HOLOSCAN_LOG_INFO("DisplayGpuResidentOp::compute() -- Done");
}

}  // namespace holoscan::ops
