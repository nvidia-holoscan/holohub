/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_XR_BEGIN_FRAME_OP_HPP
#define HOLOSCAN_OPERATORS_OPENXR_XR_BEGIN_FRAME_OP_HPP

#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"

#include "gxf/multimedia/camera.hpp"

namespace holoscan::openxr {

// Initializes an OpenXR session and begins a new OpenXR frame on each tick.
class XrBeginFrameOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrBeginFrameOp)

  XrBeginFrameOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<holoscan::openxr::XrSession>> session_;
  Parameter<bool> enable_eye_tracking_;

  std::unique_ptr<XrCudaInteropSwapchain> color_swapchain_;
  std::unique_ptr<XrCudaInteropSwapchain> depth_swapchain_;

  xr::UniqueActionSet action_set_;
  std::array<xr::UniqueSpace, 2> aim_space_;
  std::array<xr::UniqueSpace, 2> grip_space_;
  xr::UniqueSpace eye_gaze_space_;

  std::map<std::string, xr::UniqueAction> action_map_;

  bool boolAction(const xr::Session& xr_session, const std::string& name);
  nvidia::gxf::Pose3D poseAction(const xr::Posef& pose);

  nvidia::gxf::Pose3D toPose3D(const xr::View& view);
  nvidia::gxf::CameraModel toCameraModel(const xr::View& view, uint32_t display_width,
                                         uint32_t display_height);
};

}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_XR_BEGIN_FRAME_OP_HPP
