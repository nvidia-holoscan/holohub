/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_UX_UX_WINDOW_CONTROLLER_HPP
#define HOLOSCAN_OPERATORS_OPENXR_UX_UX_WINDOW_CONTROLLER_HPP

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "ux_widgets.hpp"

namespace holoscan::openxr {
class UxWindowController {
 public:
  explicit UxWindowController(UxWindow& window);

  void cursorMove(Eigen::Affine3f pose);
  void cursorClick(Eigen::Affine3f cursor);
  void cursorRelease();

  void trackPadDown(Eigen::Vector2f dxy);
  void trackPadMove(Eigen::Vector2f dxy);
  void trackPadUp();

  void alignWith(Eigen::Vector3f point);

  void reset();

 private:
  bool inside_window(Eigen::Vector3f& cursor);
  bool inside_header(Eigen::Vector3f& cursor);

  UxWindow& window_;

  Eigen::Affine3f transform_;
  Eigen::Affine3f transform_inv_;

  // control actions
  enum Action { UNDEFINED, DRAGGING, EDITING };

  Action active_action_;
  Action pending_action_;

  Eigen::Vector3f start_cursor_;
};
}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_UX_UX_WINDOW_CONTROLLER_HPP
