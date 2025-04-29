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

#ifndef XR_RENDER_RGBD_OP_HPP
#define XR_RENDER_RGBD_OP_HPP

#include <memory>
#include <optional>

#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"
#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "holoviz/holoviz.hpp"

namespace holoscan::ops {

// Renders an RGB-D image in XR.
class XrRenderRgbdOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrRenderRgbdOp)

  XrRenderRgbdOp() = default;

  void setup(OperatorSpec& spec) override;

  void start() override;
  void stop() override;

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<holoscan::XrSession>> xr_session_;

  viz::InstanceHandle holoviz_instance_;

  CudaStreamHandler cuda_stream_handler_;

  std::unique_ptr<XrSwapchainCuda> color_swapchain_;
  std::unique_ptr<XrSwapchainCuda> depth_swapchain_;

  std::optional<holoscan::TensorMap> camera_frame_;
};

}  // namespace holoscan::ops

#endif  // XR_RENDER_RGBD_OP_HPP
