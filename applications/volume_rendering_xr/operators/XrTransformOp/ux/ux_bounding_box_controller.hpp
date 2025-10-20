/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_UX_UX_BOUNDING_BOX_CONTROLLER_HPP
#define HOLOSCAN_OPERATORS_OPENXR_UX_UX_BOUNDING_BOX_CONTROLLER_HPP

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "ux_widgets.hpp"

namespace holoscan::openxr {
class UxBoundingBoxController {
 public:
  explicit UxBoundingBoxController(UxBoundingBox& boundingBox);

  void cursorMove(Eigen::Affine3f pose);
  void cursorClick(Eigen::Affine3f cursor);
  void cursorRelease();

  void trackPadDown(Eigen::Vector2f dxy);
  void trackPadMove(Eigen::Vector2f dxy);
  void trackPadUp();

  void reset();

 private:
  UxBoundingBox& box_;

  Eigen::Affine3f local_transform_;
  Eigen::Affine3f local_transform_inv_;

  Eigen::Affine3f global_transform_;
  Eigen::Affine3f global_transform_inv_;

  // coordinate system
  Eigen::Vector3f axes_[3];

  // control surfaces
  bool test_box(Eigen::Vector3f& cursor);
  void drag_box(Eigen::Vector3f& cursor);

  // Edges
  struct Edge {
    Edge(int p_0x, int p0_y, int p0_z, int p_1x, int p_1y, int p_1z, UxEdge& edge, bool enabled);

    int sign;
    int p0_x;
    int p0_y;
    int p0_z;
    int p1_x;
    int p1_y;
    int p1_z;
    bool enabled;
    UxEdge& state;
  };
  std::vector<Edge> edges_;
  float test_edge(Eigen::Vector3f& cursor, Edge& edge);
  void drag_edge(Eigen::Vector3f& cursor, Edge& edge);

  // Faces
  struct Face {
    Face(int n_sign, int n, int u_sign, int u, int v_sign, int v, UxFace& state);

    int n;  // normal axis
    int n_sign;
    int u;  // up axes
    int u_sign;
    int v;  // left axes
    int v_sign;
    UxFace& state;
  };
  std::vector<Face> faces_;
  float test_face(Eigen::Vector3f& cursor, Face& face);
  void drag_face(Eigen::Vector3f& cursor, Face& face);

  // Corners
  struct Corner {
   public:
    Corner(int sign_x, int sign_y, int sign_z, UxCorner& state);

    int sign_x;
    int sign_y;
    int sign_z;
    UxCorner& state;
  };
  std::vector<Corner> corners_;
  float test_corner(Eigen::Vector3f& cursor, Corner& corner);
  void drag_corner(Eigen::Vector3f& cursor, Corner& corner);

  // control actions
  enum Action { UNDEFINED, DRAG_FACE, DRAG_CORNER, DRAG_BOX, DRAG_EDGE };

  Action active_action_;
  Action pending_action_;
  int widget_;

  // action starting conditions
  Eigen::Vector3f start_cursor_;
  Eigen::Vector3f start_extent_;
  Eigen::Vector2f start_trackpad_;

  // unscaled extent of bounding box
  float start_scale_;

  // Debugging
  void printPending();
};
}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_UX_UX_BOUNDING_BOX_CONTROLLER_HPP
