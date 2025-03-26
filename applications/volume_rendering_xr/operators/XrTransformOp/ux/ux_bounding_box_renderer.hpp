/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_UX_UX_BOUNDING_BOX_RENDERER_HPP
#define HOLOSCAN_OPERATORS_OPENXR_UX_UX_BOUNDING_BOX_RENDERER_HPP

#include <array>
#include <vector>

#include "ux_widgets.hpp"

namespace holoscan::openxr {
class UxBoundingBoxRenderer {
 public:
  void render(UxBoundingBox& box, Eigen::Vector3f eye_pos);

 private:
  void drawCorner(UxCorner& state, Eigen::Vector3f point, Eigen::Vector3f dx, Eigen::Vector3f dy,
                  Eigen::Vector3f dz, Eigen::Vector3f& eye_pos);
  void drawEdge(UxEdge& state, Eigen::Vector3f p0, Eigen::Vector3f p1, Eigen::Vector3f& eye_pos);
  void drawFace(UxFace& state, Eigen::Vector3f p0, Eigen::Vector3f pu, Eigen::Vector3f pv,
                Eigen::Affine3f& transform);

  void drawOutline(Eigen::Vector3f vertices[8]);

  void drawAxes(float length);
  void showAll(UxBoundingBox& box);
};
}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_UX_UX_BOUNDING_BOX_CONTROLLER_HPP
