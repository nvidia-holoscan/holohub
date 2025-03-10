/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VPI_STEREO_OP
#define VPI_STEREO_OP

#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Stream.h>
#include <vpi/Types.h>
#include <vpi/algo/StereoDisparity.h>
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class VPIStereoOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VPIStereoOp);
  VPIStereoOp() = default;
  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<int> maxDisparity_;
  Parameter<int> downscaleFactor_;
  int widthDownscaled_;
  int heightDownscaled_;

  // VPI objects that can be reused each compute() call
  VPIStream stream_;
  VPIStereoDisparityEstimatorCreationParams createParams_;
  VPIStereoDisparityEstimatorParams submitParams_;
  VPIPayload payload_;
  VPIImageFormat inFmt_;
  uint64_t backends_;
  VPIImage inLeftRGB_;
  VPIImage inRightRGB_;
  VPIImage inLeftMono_;
  VPIImage inRightMono_;
  VPIImage outConf16_;
  VPIImage outDisp16_;
  VPIImage outDisp_;
};

}  // namespace holoscan::ops

#endif  // VPI_STEREO_OP
