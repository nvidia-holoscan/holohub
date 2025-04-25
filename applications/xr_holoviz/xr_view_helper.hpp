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
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"
#include "xr_begin_frame_op.hpp"
#include "xr_composition_layers.hpp"
#include "xr_end_frame_op.hpp"
#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "holoviz/holoviz.hpp"
#include "openxr/openxr.hpp"

#include "holoscan/operators/holoviz/holoviz.hpp"
#include "xr_manager.hpp"
namespace holoscan::ops {

namespace XrViewsHelper {

HolovizOp::InputSpec::View create_view(int idx, const xr::View& view,
                                       std::shared_ptr<holoscan::XrSession> xr_session,
                                       const glm::mat4& model_matrix) {
  // Calculate view matrix from pose
  glm::mat4 view_orientation = glm::mat4_cast(glm::make_quat(&view.pose.orientation.x));
  glm::mat4 view_translation = glm::translate(glm::mat4{1}, glm::make_vec3(&view.pose.position.x));
  glm::mat4 view_matrix = glm::inverse(view_translation * view_orientation);

  float nearZ = xr_session->view_configuration_depth_range().recommendedNearZ;
  float farZ = xr_session->view_configuration_depth_range().recommendedFarZ;

  // Calculate projection matrix from FOV
  glm::mat4 projection_matrix = glm::frustumRH_ZO(nearZ * glm::tan(view.fov.angleLeft),
                                                  nearZ * glm::tan(view.fov.angleRight),
                                                  nearZ * glm::tan(view.fov.angleUp),
                                                  nearZ * glm::tan(view.fov.angleDown),
                                                  nearZ,
                                                  farZ);

  glm::mat4 view_projection_matrix_row_major =
      glm::transpose(projection_matrix * view_matrix * model_matrix);

  // For stereo views, use side-by-side image layout.
  HolovizOp::InputSpec::View xr_view;
  xr_view.offset_x_ = idx * 0.5f;
  xr_view.offset_y_ = 0;
  xr_view.width_ = 0.5f;
  xr_view.height_ = 1.0f;

  std::array<float, 16> matrix_array;
  const float* matrix_ptr = glm::value_ptr(view_projection_matrix_row_major);
  std::copy(matrix_ptr, matrix_ptr + 16, matrix_array.begin());
  xr_view.matrix_ = matrix_array;

  return xr_view;
}

// Create HolovizOp::InputSpec with XR stereo views
HolovizOp::InputSpec create_spec_with_views(const std::string& tensor_name,
                                            const HolovizOp::InputType type,
                                            std::vector<xr::View>& located_views,
                                            std::shared_ptr<holoscan::XrSession> xr_session,
                                            const glm::mat4& model_matrix = glm::mat4(1.0f)) {
  // Setup HolovizOp::InputSpec
  HolovizOp::InputSpec spec;
  spec.tensor_name_ = tensor_name;
  spec.type_ = type;

  for (int i = 0; i < located_views.size(); i++) {
    // Calculate XR stereo views
    spec.views_.push_back(create_view(i, located_views[i], xr_session, model_matrix));
  }
  return spec;
}

}  // namespace XrViewsHelper

}  // namespace holoscan::ops
