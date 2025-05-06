/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "xr_composition_layer_manager.hpp"
#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "xr_composition_layers.hpp"

namespace holoscan {

void XrCompositionLayerManager::setup(ComponentSpec& spec) {
  spec.param(xr_session_, "xr_session", "OpenXR Session", "OpenXR Session");
}

void XrCompositionLayerManager::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Resource '{}' is already initialized. Skipping...", name());
    return;
  }

  Resource::initialize();
  auto xr_session = xr_session_.get();
  uint32_t width = xr_session->view_configurations()[0].recommendedImageRectWidth *
                   xr_session->view_configurations().size();
  uint32_t height = xr_session->view_configurations()[0].recommendedImageRectHeight;

  // Create swapchains
  color_swapchain_ = std::make_unique<XrSwapchainCuda>(
      *xr_session, XrSwapchainCuda::Format::R8G8B8A8_SRGB, width, height);
  depth_swapchain_ = std::make_unique<XrSwapchainCuda>(
      *xr_session, XrSwapchainCuda::Format::D32_SFLOAT, width, height);

  is_initialized_ = true;
  HOLOSCAN_LOG_INFO("XrCompositionLayerManager initialized width: {} height: {}", width, height);
}

XrCompositionLayerManager::~XrCompositionLayerManager() {}

holoscan::Tensor XrCompositionLayerManager::acquire_color_swapchain_image() {
  if (!color_swapchain_) {
    throw std::runtime_error("Color swapchain not initialized");
  }
  return color_swapchain_->acquire();
}

holoscan::Tensor XrCompositionLayerManager::acquire_depth_swapchain_image() {
  if (!depth_swapchain_) {
    throw std::runtime_error("Depth swapchain not initialized");
  }
  return depth_swapchain_->acquire();
}

void XrCompositionLayerManager::release_swapchain_images(cudaStream_t cuda_stream) {
  if (color_swapchain_) {
    color_swapchain_->release(cuda_stream);
  }
  if (depth_swapchain_) {
    depth_swapchain_->release(cuda_stream);
  }
}

std::shared_ptr<XrCompositionLayerProjectionStorage>
XrCompositionLayerManager::create_composition_layer(const xr::FrameState& xr_frame_state) {
  auto xr_session = xr_session_.get();
  return XrCompositionLayerProjectionStorage::create_for_frame(
      xr_frame_state, *xr_session, *color_swapchain_, *depth_swapchain_);
}

}  // namespace holoscan
