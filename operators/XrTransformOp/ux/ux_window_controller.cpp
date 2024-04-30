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

#include "ux_window_controller.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>
#include "holoscan/holoscan.hpp"

namespace holoscan::openxr {

// TODO is this in Eigen ?
static float clamp(float v, float min, float max) {
  return (v < min) ? min : (v > max) ? max : v;
}

UxWindowController::UxWindowController(UxWindow& window)
    : window_(window), transform_(window.transform) {
  transform_inv_ = transform_.inverse();
  active_action_ = UNDEFINED;
}

void UxWindowController::reset() {
  transform_ = window_.transform;
  transform_inv_ = transform_.inverse();

  window_.handle.action = holoscan::openxr::IDLE;
  window_.handle.range = 0;
  window_.face.action = holoscan::openxr::IDLE;
  window_.face.range = 0;

  pending_action_ = UNDEFINED;
  active_action_ = UNDEFINED;
  window_.state = holoscan::openxr::INACTIVE;
}

void UxWindowController::cursorMove(Eigen::Affine3f pose) {
  // convert to local coordinate system
  Eigen::Affine3f local_pose = transform_inv_ * pose;
  Eigen::Vector3f cursor = local_pose.translation();

  window_.cursor(0) = 0.5 + cursor(0) / (2 * window_.content(0));
  window_.cursor(1) = 0.5 + cursor(1) / (2 * window_.content(1));
  switch (active_action_) {
    case UNDEFINED:
      window_.handle.action = holoscan::openxr::IDLE;
      window_.face.action = holoscan::openxr::IDLE;
      window_.handle.range = 0;
      window_.face.range = 0;
      if (inside_window(cursor)) {
        pending_action_ = EDITING;
      } else if (inside_header(cursor)) {
        window_.handle.action = holoscan::openxr::DRAGGABLE;
        pending_action_ = DRAGGING;
      } else {
        pending_action_ = UNDEFINED;
      }
      break;
    case DRAGGING:
      window_.face.range = 0;
      window_.transform = transform_;
      window_.transform.translate(cursor - start_cursor_);
      break;
  }

  window_.state = pending_action_ == UNDEFINED && active_action_ == UNDEFINED
                      ? holoscan::openxr::INACTIVE
                      : holoscan::openxr::ACTIVE;
}

void UxWindowController::cursorClick(Eigen::Affine3f pose) {
  // convert to local coordinate system
  Eigen::Affine3f local_pose = transform_inv_ * pose;
  Eigen::Vector3f cursor = local_pose.translation();

  switch (pending_action_) {
    case DRAGGING:
      start_cursor_ = cursor;
      active_action_ = DRAGGING;
      window_.handle.action = holoscan::openxr::DRAGGING;
      break;
    case EDITING:
      window_.button = 1;
      active_action_ = EDITING;
      break;
  }
}

void UxWindowController::cursorRelease() {
  switch (active_action_) {
    case DRAGGING:
      transform_ = window_.transform;
      transform_inv_ = window_.transform.inverse();
      break;
    case EDITING:
      window_.button = 0;
      break;
  }

  window_.handle.action = holoscan::openxr::IDLE;
  window_.handle.range = 0;
  window_.face.action = holoscan::openxr::IDLE;
  window_.face.range = 0;

  pending_action_ = UNDEFINED;
  active_action_ = UNDEFINED;
  window_.state = holoscan::openxr::INACTIVE;
}

void UxWindowController::alignWith(Eigen::Vector3f point) {
  Eigen::Vector3f view_vector = point - window_.transform.translation();
  view_vector[1] = 0;
  view_vector = view_vector.normalized();

  Eigen::Vector3f normal = window_.transform.rotation().col(2);
  if (acos(normal.dot(view_vector)) > 0.1) {
    Eigen::Vector3f zAxis = (view_vector + normal).normalized();
    Eigen::Vector3f yAxis = Eigen::Vector3f::UnitY();
    Eigen::Vector3f xAxis = yAxis.cross(zAxis);

    Eigen::Matrix3f rotation;
    rotation.col(0) = xAxis;
    rotation.col(1) = yAxis;
    rotation.col(2) = zAxis;
    window_.transform.matrix().block<3, 3>(0, 0) = rotation;

    transform_ = window_.transform;
    transform_inv_ = window_.transform.inverse();
  }
}

bool UxWindowController::inside_window(Eigen::Vector3f& cursor) {
  bool inside = cursor(0) > -window_.content[0] && cursor(0) < window_.content[0] &&
                cursor(1) > -window_.content[1] && cursor(1) < window_.content[1] &&
                cursor(2) > 0 && cursor(2) < window_.content[2];

  float distance = Eigen::Vector3f::UnitZ().dot(cursor);
  window_.face.range = 1.0f - std::max(std::min(distance / window_.content(2), 1.0f), 0.0f);

  return inside && window_.face.range > 0;
}

bool UxWindowController::inside_header(Eigen::Vector3f& cursor) {
  bool inside = cursor(0) > -window_.content[0] && cursor(0) < window_.content[0] &&
                cursor(1) > window_.content[1] && cursor(1) < window_.content[1] + HEADER_HEIGHT &&
                cursor(2) > 0 && cursor(2) < window_.content[2];

  float distance = Eigen::Vector3f::UnitZ().dot(cursor);
  float range = 1.0f - std::max(std::min(distance / window_.content(2), 1.0f), 0.0f);

  return inside && range > 0;
}

void UxWindowController::trackPadDown(Eigen::Vector2f trackpad) {}

void UxWindowController::trackPadMove(Eigen::Vector2f trackpad) {}

void UxWindowController::trackPadUp() {}

}  // namespace holoscan::openxr
