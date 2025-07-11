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

#include "xr_composition_layers.hpp"

#include <memory>
#include <vector>

namespace holoscan {

std::shared_ptr<XrCompositionLayerProjectionStorage>
XrCompositionLayerProjectionStorage::create_for_frame(xr::FrameState xr_frame_state,
                                                      XrSession& xr_session,
                                                      XrSwapchainCuda& color_swapchain,
                                                      XrSwapchainCuda& depth_swapchain) {
  // Locate the views for xr_frame_state.
  xr::ViewLocateInfo view_locate_info(xr::ViewConfigurationType::PrimaryStereo,
                                      xr_frame_state.predictedDisplayTime,
                                      xr_session.reference_space());
  xr::ViewState view_state;
  std::vector<xr::View> views =
      xr_session.get().locateViewsToVector(view_locate_info, view_state.put());

  // Create the layer storage.
  auto layer = std::make_shared<XrCompositionLayerProjectionStorage>();

  assert(layer->views.size() >= views.size());
  assert(layer->depth_info.size() >= views.size());

  // Populate each projection layer view.
  for (int i = 0; i < views.size(); i++) {
    // For stereo views, use side-by-side image layout.
    xr::Extent2Di image_extent(color_swapchain.width() / 2, color_swapchain.height());
    xr::Rect2Di image_rect(xr::Offset2Di(i * color_swapchain.width() / 2, 0), image_extent);

    // Make full use of the normalized depth range for depth image values.
    constexpr float kMinDepth = 0.f;
    constexpr float kMaxDepth = 1.f;

    layer->depth_info[i] = xr::CompositionLayerDepthInfoKHR({
        xr::SwapchainSubImage(depth_swapchain.get(), image_rect, 0),
        kMinDepth,
        kMaxDepth,
        xr_session.view_configuration_depth_range().recommendedNearZ,
        xr_session.view_configuration_depth_range().recommendedFarZ,
    });

    layer->views[i] = xr::CompositionLayerProjectionView({
        views[i].pose,
        views[i].fov,
        xr::SwapchainSubImage(color_swapchain.get(), image_rect, 0),
        &layer->depth_info[i],
    });
  }

  layer->layerFlags = xr::CompositionLayerFlagBits::BlendTextureSourceAlpha;
  layer->space = xr_session.reference_space();
  layer->viewCount = static_cast<uint32_t>(views.size());
  layer->xr::CompositionLayerProjection::views = layer->views.data();

  return layer;
}

}  // namespace holoscan

