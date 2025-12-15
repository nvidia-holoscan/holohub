/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "xr_transform_control_op.hpp"

#include <nlohmann/json.hpp>
#include "holoviz/holoviz.hpp"

namespace holoscan::openxr {

void XrTransformControlOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Pose3D>("aim_pose");
  spec.input<bool>("trigger_click");
  spec.input<bool>("shoulder_click");
  spec.input<std::array<float, 2>>("trackpad");
  spec.input<bool>("trackpad_touch");
  spec.input<nvidia::gxf::Pose3D>("head_pose");
  spec.input<std::array<float, 3>>("extent").condition(ConditionType::kNone);

  spec.output<nvidia::gxf::Pose3D>("volume_pose");
  spec.output<std::array<nvidia::gxf::Vector2f, 3>>("crop_box");
  spec.output<UxBoundingBox>("ux_box");
  spec.output<UxCursor>("ux_cursor");
  spec.output<UxWindow>("ux_window");
}

void XrTransformControlOp::start() {
  cursor_down_ = false;
  trackpad_touched_ = false;

  // create cursor control
  ui_cursor_controller_ = std::make_shared<UxCursorController>(ui_cursor_);

  // setup box control
  ui_box_.half_extent = Eigen::Vector3f(1.f, 1.f, 1.f);
  ui_box_.scale = 1.0;
  ui_box_.local_transform.setIdentity();
  ui_box_.local_transform.translate(Eigen::Vector3f(0, 0, 0));
  ui_box_.global_transform.setIdentity();
  ui_box_.global_transform.translate(Eigen::Vector3f{0.0, 0.0, -1.0});
  ui_box_controller_ = std::make_shared<UxBoundingBoxController>(ui_box_);

  // setup window controller
  ui_window_.transform = Eigen::Matrix4f::Identity();
  ui_window_.transform.translate(Eigen::Vector3f{0.65, 0.1, -0.75});
  ui_window_.content = {0.25, 0.25, 0.2};  // window half extent in meters
  ui_window_.cursor = {0.74, 0.14};
  ui_window_controller_ = std::make_shared<UxWindowController>(ui_window_);
}

void XrTransformControlOp::compute(InputContext& input, OutputContext& output,
                                   ExecutionContext& context) {
  auto aim_pose = input.receive<nvidia::gxf::Pose3D>("aim_pose");
  auto trigger_click = input.receive<bool>("trigger_click");
  auto shoulder_click = input.receive<bool>("shoulder_click");
  Eigen::Map<Eigen::Vector2f> trackpad(input.receive<std::array<float, 2>>("trackpad")->data());
  auto trackpad_touch = input.receive<bool>("trackpad_touch");
  auto head_pose = input.receive<nvidia::gxf::Pose3D>("head_pose");
  auto extent = input.receive<std::array<float, 3>>("extent");

  if (extent) {
    // extent is in millimeters, convert to meters
    model_half_extent_ =
        Eigen::Vector3f(extent->at(0), extent->at(1), extent->at(2)) * (0.5f / 1000.f);
    ui_box_.half_extent = model_half_extent_;
  }

  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  if (*shoulder_click) {
    if (std::chrono::duration_cast<std::chrono::seconds>(now - timestamp_) >
        std::chrono::seconds(3)) {
      Eigen::Affine3f head_transform;
      head_transform.translation() = Eigen::Map<Eigen::Vector3f>(head_pose->translation.data());
      head_transform.linear() =
          Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(head_pose->rotation.data());

      ui_box_.half_extent = model_half_extent_;
      ui_box_.scale = 1.0;
      ui_box_.local_transform.setIdentity();
      ui_box_.local_transform.translate(Eigen::Vector3f(0, 0, 0));
      ui_box_.global_transform.setIdentity();
      ui_box_.global_transform.translate(head_transform * Eigen::Vector3f(0.0, 0.0, -1.0));
      ui_box_controller_->reset();

      ui_window_.transform = Eigen::Matrix4f::Identity();
      ui_window_.transform.translate(head_transform * Eigen::Vector3f{0.65, 0.1, -0.75});
      ui_window_controller_->reset();
    }
  } else {
    timestamp_ = now;
  }

  // update trackpad state
  if (*trackpad_touch && !trackpad_touched_) {
    ui_box_controller_->trackPadDown(trackpad);
    trackpad_touched_ = true;
  } else if (*trackpad_touch) {
    ui_box_controller_->trackPadMove(trackpad);
  } else if (trackpad_touched_) {
    ui_box_controller_->trackPadUp();
    trackpad_touched_ = false;
  }

  // initialize cursor matrix
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  matrix.matrix().topLeftCorner<3, 3>() =
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(aim_pose->rotation.data());
  matrix.matrix().topRightCorner<3, 1>() =
      Eigen::Map<Eigen::Vector3f>(aim_pose->translation.data());

  Eigen::Affine3f cursor(matrix);
  if (*trigger_click) {
    if (!cursor_down_) {
      if (ui_box_.state == holoscan::openxr::INACTIVE) {
        ui_window_controller_->cursorClick(cursor);
      }
      if (ui_window_.state == holoscan::openxr::INACTIVE) {
        ui_box_controller_->cursorClick(cursor);
      }
      cursor_down_ = true;
    } else {
      if (ui_box_.state == holoscan::openxr::INACTIVE) {
        ui_window_controller_->cursorMove(cursor);
      }
      if (ui_window_.state == holoscan::openxr::INACTIVE) {
        ui_box_controller_->cursorMove(cursor);
      }
    }
  } else if (cursor_down_) {
    ui_window_controller_->cursorRelease();
    ui_box_controller_->cursorRelease();
    cursor_down_ = false;
  } else {
    if (ui_box_.state == holoscan::openxr::INACTIVE) {
      ui_window_controller_->cursorMove(cursor);
    }
    if (ui_window_.state == holoscan::openxr::INACTIVE) {
      ui_box_controller_->cursorMove(cursor);
    }
  }

  if (ui_window_.state == holoscan::openxr::INACTIVE) {
    Eigen::Vector3f focus = Eigen::Map<Eigen::Vector3f>(head_pose->translation.data());
    ui_window_controller_->alignWith(focus);
  }

  // Emit the object transform as a Pose3D.
  Eigen::Affine3f transform = ui_box_.global_transform;
  transform.scale(ui_box_.scale);

  nvidia::gxf::Pose3D volume_pose;
  Eigen::Map<Eigen::Vector3f>(volume_pose.translation.data()) = transform.translation();
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(volume_pose.rotation.data()) =
      transform.matrix().block(0, 0, 3, 3);
  output.emit(volume_pose, "volume_pose");

  // Emit the normalized slice volume
  Eigen::Vector3f box_position = ui_box_.local_transform.translation();
  Eigen::Vector3f model_half_extent = model_half_extent_ * ui_box_.scale;
  std::array<nvidia::gxf::Vector2f, 3> crop_box;
  for (int i = 0; i < 3; i++) {
    crop_box[i].x = (box_position(i) - ui_box_.half_extent[i] + model_half_extent(i)) /
                    (2 * model_half_extent(i));
    crop_box[i].y = (box_position(i) + ui_box_.half_extent[i] + model_half_extent(i)) /
                    (2 * model_half_extent(i));
  }

  output.emit(crop_box, "crop_box");

  // Emit widget state.
  output.emit(ui_box_, "ux_box");
  output.emit(ui_cursor_, "ux_cursor");
  output.emit(ui_window_, "ux_window");
}

}  // namespace holoscan::openxr
