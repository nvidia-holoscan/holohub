/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "xr_end_frame_op.hpp"

#include "Eigen/Dense"
#include "gxf/core/entity.hpp"
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"

namespace holoscan::openxr {

void XrEndFrameOp::setup(OperatorSpec& spec) {
  spec.input<XrFrame>("xr_frame");
  spec.input<holoscan::gxf::Entity>("color_buffer");
  spec.input<holoscan::gxf::Entity>("depth_buffer");
  spec.param(session_, "session", "OpenXR Session", "handles to OpenXR and Vulkan context");
}

void XrEndFrameOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
  std::shared_ptr<holoscan::openxr::XrSession> session = session_.get();

  auto color_message = input.receive<holoscan::gxf::Entity>("color_buffer").value();
  auto depth_message = input.receive<holoscan::gxf::Entity>("depth_buffer").value();
  auto frame = input.receive<XrFrame>("xr_frame");

  frame->color_swapchain.release(color_message);
  frame->depth_swapchain.release(depth_message);

  uint32_t display_width = session->display_width();
  uint32_t display_height = session->display_height();
  xr::Extent2Di extent(display_width, display_height);

  std::vector<xr::CompositionLayerProjectionView> projection_layer_views(2);
  std::vector<xr::CompositionLayerDepthInfoKHR> depth_info(2);
  for (int i = 0; i < projection_layer_views.size(); i++) {
    // stereo views are stacked vertically
    xr::Rect2Di rect(xr::Offset2Di(0, i * display_height), extent);

    depth_info[i] = xr::CompositionLayerDepthInfoKHR({
        xr::SwapchainSubImage(frame->depth_swapchain.handle(), rect, 0),
        /*minDepth=*/0,
        /*maxDepth=*/1,
        session->view_configuration_depth_range().recommendedNearZ,
        session->view_configuration_depth_range().recommendedFarZ,
    });
    projection_layer_views[i] = xr::CompositionLayerProjectionView({
        frame->views[i].pose,
        frame->views[i].fov,
        xr::SwapchainSubImage(frame->color_swapchain.handle(), rect, 0),
        &depth_info[i],
    });
  }
  xr::CompositionLayerProjection projection_layer({
      xr::CompositionLayerFlagBits::BlendTextureSourceAlpha,
      session->reference_space(),
      static_cast<uint32_t>(projection_layer_views.size()),
      projection_layer_views.data(),
  });
  std::vector<xr::CompositionLayerBaseHeader*> layers = {
      reinterpret_cast<xr::CompositionLayerBaseHeader*>(&projection_layer),
  };

  session->handle().endFrame({
      frame->state.predictedDisplayTime,
      session->blend_mode(),
      static_cast<uint32_t>(layers.size()),
      layers.data(),
  });
}

}  // namespace holoscan::openxr
