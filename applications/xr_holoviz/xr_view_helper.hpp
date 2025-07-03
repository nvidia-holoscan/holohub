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
#include "xr_composition_layers.hpp"
#include "holoviz/holoviz.hpp"
#include "openxr/openxr.hpp"

#include "holoscan/operators/holoviz/holoviz.hpp"
namespace holoscan::ops {

namespace XrViewsHelper {

// Create HolovizOp::InputSpec with XR stereo views
HolovizOp::InputSpec create_spec_with_views(
    const std::string& tensor_name, const HolovizOp::InputType type,
    std::shared_ptr<XrCompositionLayerProjectionStorage> composition_layer,
    const glm::mat4& model_matrix = glm::mat4(1.0f)) {
  // Setup HolovizOp::InputSpec
  HolovizOp::InputSpec spec;
  spec.tensor_name_ = tensor_name;
  spec.type_ = type;

  // Add a Holoviz view for each XR composition layer view.
  for (int i = 0; i < composition_layer->viewCount; i++) {
    xr::CompositionLayerProjectionView& view = composition_layer->views[i];

    glm::mat4 view_orientation = glm::mat4_cast(glm::make_quat(&view.pose.orientation.x));
    glm::mat4 view_translation =
        glm::translate(glm::mat4{1}, glm::make_vec3(&view.pose.position.x));
    glm::mat4 view_matrix = glm::inverse(view_translation * view_orientation);

    // Holoviz uses zero-to-one projection matrix. OpenXR uses right-handed
    // coordinate system.
    glm::mat4 projection_matrix =
        glm::frustumRH_ZO(composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleLeft),
                          composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleRight),
                          composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleUp),
                          composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleDown),
                          composition_layer->depth_info[i].nearZ,
                          composition_layer->depth_info[i].farZ);

    glm::mat4 view_projection_matrix_row_major =
        glm::transpose(projection_matrix * view_matrix * model_matrix);

    // For stereo views, use side-by-side image layout.
    HolovizOp::InputSpec::View xr_view;
    xr_view.offset_x_ = i * 0.5f;
    xr_view.offset_y_ = 0;
    xr_view.width_ = 0.5f;
    xr_view.height_ = 1.0f;

    std::array<float, 16> matrix_array;
    const float* matrix_ptr = glm::value_ptr(view_projection_matrix_row_major);
    std::copy(matrix_ptr, matrix_ptr + 16, matrix_array.begin());
    xr_view.matrix_ = matrix_array;

    spec.views_.push_back(xr_view);
  }

  return spec;
}

}  // namespace XrViewsHelper

}  // namespace holoscan::ops
