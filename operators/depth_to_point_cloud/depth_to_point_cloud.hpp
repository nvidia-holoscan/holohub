/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_DEPTH_TO_POINT_CLOUD_DEPTH_TO_POINT_CLOUD_HPP
#define HOLOSCAN_OPERATORS_DEPTH_TO_POINT_CLOUD_DEPTH_TO_POINT_CLOUD_HPP

#include <memory>
#include <string>

#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace holoscan::ops {

/**
 * @brief Deproject an organized depth image into an organized point cloud on the GPU.
 *
 * ==Named Inputs==
 *
 * - **depth** : `nvidia::gxf::Entity` containing a 2D depth `nvidia::gxf::Tensor`
 *   - Element type `uint16` (raw units scaled by `depth_scale`) or `float32` (meters when
 *     `depth_scale == 1.0`). Shape `[H, W]` or `[H, W, 1]`, device memory.
 * - **intrinsics** *(optional)* : `nvidia::gxf::Entity` containing a `float32` tensor of 4
 *   values `[fx, fy, cx, cy]`. When present, overrides the `fx/fy/cx/cy` parameters for the
 *   current frame (e.g. for sensors that publish per-stream intrinsics).
 * - **color** *(optional)* : `nvidia::gxf::Entity` containing an `H x W x 3` or `H x W x 4`
 *   `uint8` color image aligned to the depth image. When connected, a colored output is emitted.
 *
 * ==Named Outputs==
 *
 * - **point_cloud** : `nvidia::gxf::Entity` containing:
 *   - a `float32` tensor (`output_tensor_name`, default `"point_cloud"`) of shape `[H, W, 3]`
 *     (organized, AoS XYZ) in the camera optical frame (x-right, y-down, z-forward). Invalid
 *     pixels (depth == 0 or outside `[depth_min, depth_max]`) are set to `invalid_value`; and
 *   - *(only when `color` is connected)* a `uint8` tensor (`output_color_tensor_name`, default
 *     `"colors"`) of shape `[H, W, 3]`.
 *
 * ==Parameters==
 *
 * - **fx**, **fy**, **cx**, **cy**: pinhole intrinsics in pixels (used when the `intrinsics`
 *   port is not connected).
 * - **depth_scale**: multiply raw depth by this to get meters (default `0.001`, i.e. uint16 mm).
 * - **depth_min**, **depth_max**: valid metric depth range in meters (defaults `0.0` / `100.0`).
 * - **invalid_value**: value written to X/Y/Z for invalid pixels (default `NaN`).
 * - **depth_tensor_name**, **color_tensor_name**: input tensor names (default: first tensor).
 * - **output_tensor_name**: name of the emitted point-cloud tensor (default `"point_cloud"`).
 * - **output_color_tensor_name**: name of the emitted colors tensor (default `"colors"`).
 * - **allocator**: device `holoscan::Allocator` for the output tensors (e.g. BlockMemoryPool).
 * - **cuda_stream_pool**: optional `holoscan::CudaStreamPool` for stream allocation.
 */
class DepthToPointCloudOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DepthToPointCloudOp)

  DepthToPointCloudOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<float> fx_;
  Parameter<float> fy_;
  Parameter<float> cx_;
  Parameter<float> cy_;
  Parameter<float> depth_scale_;
  Parameter<float> depth_min_;
  Parameter<float> depth_max_;
  Parameter<float> invalid_value_;
  Parameter<std::string> depth_tensor_name_;
  Parameter<std::string> color_tensor_name_;
  Parameter<std::string> output_tensor_name_;
  Parameter<std::string> output_color_tensor_name_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_DEPTH_TO_POINT_CLOUD_DEPTH_TO_POINT_CLOUD_HPP */
