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

#ifndef XR_COMPOSITION_LAYERS_HPP
#define XR_COMPOSITION_LAYERS_HPP

#include <memory>

#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "openxr/openxr.hpp"

namespace holoscan {

// An xr::CompositionLayerProjection that stores its own views and depth info.
//
// XrCompositionLayerProjectionStorage is intended to be emitted by XR rendering
// operators and received by XrEndFrameOp.
struct XrCompositionLayerProjectionStorage : public xr::CompositionLayerProjection {
  using xr::CompositionLayerProjection::CompositionLayerProjection;

  // Creates an XrCompositionLayerProjectionStorage which is populated with
  // side-by-side projection views.
  static std::shared_ptr<XrCompositionLayerProjectionStorage> create_for_frame(
      xr::FrameState xr_frame_state, XrSession& xr_session, XrSwapchainCuda& color_swapchain,
      XrSwapchainCuda& depth_swapchain);

  static constexpr int32_t kMaxViews = 2;
  std::array<xr::CompositionLayerProjectionView, kMaxViews> views;
  std::array<xr::CompositionLayerDepthInfoKHR, kMaxViews> depth_info;
};

}  // namespace holoscan

#endif  // XR_COMPOSITION_LAYERS_HPP
