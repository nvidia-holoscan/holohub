/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef B6576727_38D3_46C4_B968_7A74CBF8E455
#define B6576727_38D3_46C4_B968_7A74CBF8E455

#ifndef HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_RENDER_HPP
#define HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_RENDER_HPP

#include "Eigen/Dense"
#include "holoscan/holoscan.hpp"

#include "ux/ux_bounding_box_renderer.hpp"

#include "gxf/multimedia/camera.hpp"

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

  Parameter<uint32_t> display_width_;
  Parameter<uint32_t> display_height_;

  UxBoundingBoxRenderer ui_box_renderer_;

  bool initialized_ = false;

  std::array<float, 16> toModelView(nvidia::gxf::Pose3D pose, nvidia::gxf::CameraModel camera,
                                    float near_z, float far_z);
};

}  // namespace holoscan::openxr
#endif  // HOLOSCAN_OPERATORS_OPENXR_XR_TRANSFORM_RENDER_HPP


#endif /* B6576727_38D3_46C4_B968_7A74CBF8E455 */
