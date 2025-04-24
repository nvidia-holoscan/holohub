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

#include "xr_manager.hpp"
#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "xr_composition_layers.hpp"

namespace holoscan {

void XrManager::setup(ComponentSpec& spec) {
  spec.param(xr_session_, "xr_session", "OpenXR Session", "OpenXR Session");
}

void XrManager::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Resource '{}' is already initialized. Skipping...", name());
    return;
  }

  Resource::initialize();
  create_swapchains();
  is_initialized_ = true;
  HOLOSCAN_LOG_INFO("XrManager initialized width: {} height: {}", width_, height_);
}

void XrManager::create_swapchains() {
  std::shared_ptr<holoscan::XrSession> xr_session = get_xr_session();
  width_ = xr_session->view_configurations()[0].recommendedImageRectWidth *
           xr_session->view_configurations().size();
  height_ = xr_session->view_configurations()[0].recommendedImageRectHeight;
  // Create swapchains
  color_swapchain_ = std::make_unique<XrSwapchainCuda>(
      *xr_session, XrSwapchainCuda::Format::R8G8B8A8_SRGB, width_, height_);
  depth_swapchain_ = std::make_unique<XrSwapchainCuda>(
      *xr_session, XrSwapchainCuda::Format::D32_SFLOAT, width_, height_);
}

XrManager::~XrManager() {}

std::shared_ptr<holoscan::XrSession> XrManager::get_xr_session() {
  return xr_session_.get();
}

holoscan::Tensor XrManager::acquire_color_swapchain() {
  if (!color_swapchain_) {
    throw std::runtime_error("Color swapchain not initialized");
  }
  return color_swapchain_->acquire();
}

holoscan::Tensor XrManager::acquire_depth_swapchain() {
  if (!depth_swapchain_) {
    throw std::runtime_error("Depth swapchain not initialized");
  }
  return depth_swapchain_->acquire();
}

void XrManager::release_swapchains(cudaStream_t cuda_stream) {
  if (color_swapchain_) {
    color_swapchain_->release(cuda_stream);
  }
  if (depth_swapchain_) {
    depth_swapchain_->release(cuda_stream);
  }
}

std::shared_ptr<XrCompositionLayerProjectionStorage> XrManager::create_composition_layer() {
  auto xr_session = get_xr_session();
  return XrCompositionLayerProjectionStorage::create_for_frame(
      *xr_session, *color_swapchain_, *depth_swapchain_, located_views_);
}

const std::vector<xr::View>& XrManager::update_located_views(const xr::FrameState& frame_state) {
  auto xr_session = get_xr_session();
  xr::ViewLocateInfo view_locate_info(xr::ViewConfigurationType::PrimaryStereo,
                                      frame_state.predictedDisplayTime,
                                      xr_session->reference_space());
  xr::ViewState view_state;
  located_views_ = xr_session->get().locateViewsToVector(view_locate_info, view_state.put());
  return located_views_;
}

}  // namespace holoscan
