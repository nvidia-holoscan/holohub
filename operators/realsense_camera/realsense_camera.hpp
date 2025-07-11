/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_REALSENSE_CAMERA
#define HOLOSCAN_OPERATORS_REALSENSE_CAMERA

#include "holoscan/core/operator.hpp"
#include "holoscan/holoscan.hpp"
#include "librealsense2/rs.hpp"

namespace holoscan::ops {

/**
 * @brief Captures frames from an Intel RealSense camera.
 */
class RealsenseCameraOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RealsenseCameraOp)

  RealsenseCameraOp() = default;

  void setup(OperatorSpec& spec) override;

  void start() override;
  void stop() override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;

  rs2::align align_{RS2_STREAM_COLOR};
  rs2::pipeline pipeline_;
  rs2::pipeline_profile profile_;
  rs2::units_transform units_transform_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_REALSENSE_CAMERA
