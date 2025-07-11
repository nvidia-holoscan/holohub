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

#ifndef SRC_HOLOLINK_OPERATORS_APRILTAG_DETECTOR_APRILTAG_DETECTOR
#define SRC_HOLOLINK_OPERATORS_APRILTAG_DETECTOR_APRILTAG_DETECTOR

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include "cuAprilTags.h"
#include "cuda.h"

namespace holoscan::ops {

// The ApriltagDetectorOp operator detects number_of_tags_ of
// NVAT_TAG36H11 family only in an RGB image of resolution
// width_xheight_.

class ApriltagDetectorOp : public holoscan::Operator {
 public:
  struct output_corners {
    float2 corners[4];
    uint16_t id;
  };

  HOLOSCAN_OPERATOR_FORWARD_ARGS(ApriltagDetectorOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override;

 private:
  holoscan::Parameter<int> width_;
  holoscan::Parameter<int> height_;
  // number_of_tags_ will let the operator know, how many
  // apriltags the user is expecting.
  holoscan::Parameter<int> number_of_tags_;

  cuAprilTagsHandle apriltag_handle_;

  CUcontext cuda_context_ = nullptr;
  CUdevice cuda_device_ = 0;
  holoscan::CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif  // SRC_HOLOLINK_OPERATORS_APRILTAG_DETECTOR_APRILTAG_DETECTOR
