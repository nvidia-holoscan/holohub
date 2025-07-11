/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef XR_SWAPCHAIN_CUDA_HPP
#define XR_SWAPCHAIN_CUDA_HPP

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#include "openxr/openxr.h"
#include "openxr/openxr_platform.h"

#include "openxr/openxr.hpp"

#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"

namespace holoscan {

// Manages an OpenXR swapchain that can be written to using CUDA.
class XrSwapchainCuda {
 public:
  enum class Format {
    R8G8B8A8_SRGB,
    R8G8B8A8_UNORM,
    D16_UNORM,
    D32_SFLOAT,
  };

  XrSwapchainCuda(XrSession& session, Format format, uint32_t width, uint32_t height) noexcept;
  ~XrSwapchainCuda();

  // Acquires a CUDA swapchain image to write into.
  holoscan::Tensor acquire();

  // Releases the oldest swapchain image which has been acquired.
  void release(cudaStream_t cuda_stream);

  xr::Swapchain& get() { return *swapchain_; }
  uint32_t width() { return width_; }
  uint32_t height() { return height_; }

 private:
  // Resources for each image in the swapchain.
  struct Image {
    // Vulkan image from the OpenXR swapchain.
    vk::raii::Image vk_image{nullptr};

    // Vulkan/CUDA interop buffer for the application to write to.
    vk::raii::Buffer transfer_buffer{nullptr};
    vk::raii::DeviceMemory transfer_memory{nullptr};
    cudaExternalMemory_t cuda_transfer_memory;
    void* cuda_transfer_buffer;
    int cuda_device;

    // Vulkan command buffer to copy the interop buffer to the swapchain image.
    vk::raii::CommandBuffer command_buffer{nullptr};
    vk::raii::Semaphore render_done_vk_semaphore{nullptr};
    cudaExternalSemaphore_t render_done_cuda_semaphore;
  };

  XrSession& session_;
  xr::UniqueSwapchain swapchain_;
  Format format_;
  uint32_t width_;
  uint32_t height_;

  vk::raii::CommandPool command_pool_{nullptr};

  // Vector of swapchain images.
  std::vector<Image> images_;
  uint32_t current_image_index_;
};

}  // namespace holoscan

#endif /* XR_SWAPCHAIN_CUDA_HPP */
