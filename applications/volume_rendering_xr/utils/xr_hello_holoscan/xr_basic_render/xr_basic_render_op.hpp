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

#ifndef HOLOSCAN_XR_BASIC_RENDER_OP
#define HOLOSCAN_XR_BASIC_RENDER_OP

#include <Eigen/Dense>
#include <gxf/multimedia/camera.hpp>
#include <holoscan/holoscan.hpp>
#include <holoviz/holoviz.hpp>

namespace holoscan::ops {

/// @brief BasicRenderOp is an operator that renders a simple scene with a cube and axes.
/// @see HelloXRApp
class BasicRenderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicRenderOp)

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  void drawVizLayers(InputContext& input);
  void drawImGuiLayer(InputContext& input);
  void drawGeometryLayer(InputContext& input);
  void drawWorldCubes();
  void drawController();
  void drawEyeGaze();
  void drawCube(const Eigen::Affine3f& pose);
  void drawAxes(const Eigen::Affine3f& pose, const float lineWidth, const float scale = 1.0f);

  cudaStream_t cuda_stream_;
  Parameter<uint32_t> display_width_;
  Parameter<uint32_t> display_height_;

  holoscan::viz::InstanceHandle instance_;

  Eigen::Affine3f head_pose_{Eigen::Affine3f::Identity()};
  Eigen::Affine3f aim_pose_{Eigen::Affine3f::Identity()};
  Eigen::Affine3f grip_pose_{Eigen::Affine3f::Identity()};
  Eigen::Affine3f eye_gaze_pose_{Eigen::Affine3f::Identity()};
  bool trigger_clicked_{false};

  std::array<float, 16> toModelView(Eigen::Affine3f world, nvidia::gxf::CameraModel camera,
                                    float near_z, float far_z);

  Eigen::Affine3f toEigen(nvidia::gxf::Pose3D pose);
  void ImGuiVec2Text(const Eigen::Vector2f& p);
  void ImGuiVec3Text(const Eigen::Vector3f& p);
  void ImGuiQuatText(const Eigen::Quaternionf& q);
  void ImGuiPoseText(const Eigen::Affine3f& q);
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_XR_BASIC_RENDER_OP */
