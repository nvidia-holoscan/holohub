/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef HOLOSCAN_OPERATORS_TENSOR_TO_VIDEO_BUFFER
#define HOLOSCAN_OPERATORS_TENSOR_TO_VIDEO_BUFFER

#include <string>

#include "gxf/multimedia/video.hpp"

#include "holoscan/core/operator.hpp"
namespace holoscan::ops {

/**
 * @brief Operator class to convert Tensor to VideoBuffer.
 *
 * This operator takes Tensor as input and outputs GXF VideoBuffer created from it.
 */
class TensorToVideoBufferOp: public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorToVideoBufferOp)

  TensorToVideoBufferOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<holoscan::IOSpec*> data_in_;
  Parameter<holoscan::IOSpec*> data_out_;
  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> video_format_;

  nvidia::gxf::VideoFormat video_format_type_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_TENSOR_TO_VIDEO_BUFFER
