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

#ifndef HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_CONTROL_HPP
#define HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_CONTROL_HPP

#include "Eigen/Dense"
#include "holoscan/holoscan.hpp"

#include "ux/ux_bounding_box_controller.hpp"
#include "ux/ux_cursor.hpp"
#include "ux/ux_window_controller.hpp"

#include "gxf/multimedia/camera.hpp"

namespace holoscan::openxr {

class XrTransformControlOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrTransformControlOp)

  XrTransformControlOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Eigen::Vector3f model_half_extent_;

  UxBoundingBox ui_box_;
  std::shared_ptr<UxBoundingBoxController> ui_box_controller_;

  UxCursor ui_cursor_;
  std::shared_ptr<UxCursorController> ui_cursor_controller_;

  UxWindow ui_window_;
  std::shared_ptr<UxWindowController> ui_window_controller_;

  bool trackpad_touched_;
  bool cursor_down_;
  std::chrono::time_point<std::chrono::system_clock> timestamp_;
};

}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_CONTROL_HPP
