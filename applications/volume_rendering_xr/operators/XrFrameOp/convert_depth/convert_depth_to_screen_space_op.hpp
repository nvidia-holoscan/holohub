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

#ifndef HOLOSCAN_OPERATORS_OPENXR_CONVERT_DEPTH_TO_SCREEN_SPACE_OP_HPP
#define HOLOSCAN_OPERATORS_OPENXR_CONVERT_DEPTH_TO_SCREEN_SPACE_OP_HPP

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>

namespace holoscan::openxr {

// Converts a depth buffer from linear world units to screen space ([0,1]).
//
// For use when submitting depth buffers to OpenXR, which requires traditional
// screen space depth. The depth buffer is converted in-place.
class ConvertDepthToScreenSpaceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ConvertDepthToScreenSpaceOp)

  ConvertDepthToScreenSpaceOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  cudaStream_t stream_;
};

}  // namespace holoscan::openxr

#endif /* HOLOSCAN_OPERATORS_OPENXR_CONVERT_DEPTH_TO_SCREEN_SPACE_OP_HPP */
