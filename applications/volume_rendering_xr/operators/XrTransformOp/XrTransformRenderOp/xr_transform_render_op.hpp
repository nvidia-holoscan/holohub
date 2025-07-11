/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_RENDER_HPP
#define HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_RENDER_HPP

#include "Eigen/Dense"
#include "holoscan/holoscan.hpp"
#include "holoviz/holoviz.hpp"

#include "gxf/multimedia/camera.hpp"
#include "ux/ux_bounding_box_renderer.hpp"
#include "ux/ux_window_renderer.hpp"

namespace holoscan::openxr {

class XrTransformRenderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrTransformRenderOp)

  XrTransformRenderOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  cudaStream_t cuda_stream_;

  Parameter<std::string> config_file_;
  Parameter<uint32_t> display_width_;
  Parameter<uint32_t> display_height_;

  UxBoundingBoxRenderer ui_box_renderer_;
  UxWindowRenderer ui_window_renderer_;

  holoscan::viz::InstanceHandle instance_;

  struct Params;
  std::shared_ptr<Params> render_params_;

  std::array<float, 16> toModelView(Eigen::Affine3f world, nvidia::gxf::CameraModel camera,
                                    float near_z, float far_z);

  Eigen::Affine3f toEigen(nvidia::gxf::Pose3D pose);
};

}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_RENDER_HPP
