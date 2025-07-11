/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_XR_CUDA_INTEROP_SWAPCHAIN_HPP
#define HOLOSCAN_OPERATORS_OPENXR_XR_CUDA_INTEROP_SWAPCHAIN_HPP

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr.hpp>

#include <cuda_runtime.h>

#include "gxf/cuda/cuda_event.hpp"
#include "gxf/multimedia/video.hpp"
#include "holoscan/holoscan.hpp"

namespace holoscan::openxr {

class XrSession;

// Creates
class XrCudaInteropSwapchain {
 public:
  static std::unique_ptr<XrCudaInteropSwapchain> create(
      XrSession& session, xr::SwapchainCreateInfo swapchain_create_info);

  XrCudaInteropSwapchain(XrSession& session, xr::SwapchainCreateInfo swapchain_create_info);
  ~XrCudaInteropSwapchain();

  void acquire(holoscan::gxf::Entity entity);
  void release(holoscan::gxf::Entity entity);

  xr::Swapchain& handle() { return *handle_; }
  xr::Extent2Di image_extent() {
    return {static_cast<int32_t>(swapchain_create_info_.width),
            static_cast<int32_t>(swapchain_create_info_.height)};
  }

 private:
  struct Image {
    vk::Image vk_image;
    vk::raii::Buffer transfer_buffer{nullptr};
    vk::raii::DeviceMemory transfer_memory{nullptr};
    cudaExternalMemory_t cuda_transfer_memory;
    void* cuda_transfer_buffer;
    vk::raii::CommandBuffer command_buffer{nullptr};
    vk::raii::Semaphore render_done_vk_semaphore{nullptr};
    cudaExternalSemaphore_t render_done_cuda_semaphore;
  };

  XrSession& session_;
  xr::Extent2Di image_extent_;
  xr::SwapchainCreateInfo swapchain_create_info_;
  xr::UniqueSwapchain handle_;
  vk::raii::CommandPool command_pool_{nullptr};
  std::vector<Image> images_;
  uint32_t current_image_index_;
};

}  // namespace holoscan::openxr

#endif /* HOLOSCAN_OPERATORS_OPENXR_XR_CUDA_INTEROP_SWAPCHAIN_HPP */
