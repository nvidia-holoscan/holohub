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

#ifndef HOLOSCAN_OPERATORS_USB_V4L2_CAPTURE_HPP
#define HOLOSCAN_OPERATORS_USB_V4L2_CAPTURE_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from V4L2.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::V4L2VideoCapture`).
 */
class V4L2VideoCaptureOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(V4L2VideoCaptureOp, holoscan::ops::GXFOperator)

  V4L2VideoCaptureOp() = default;

  const char* gxf_typename() const override { return "nvidia::holoscan::V4L2VideoCapture"; }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

 private:
  Parameter<holoscan::IOSpec*> signal_;

  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> device_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> num_buffers_;
  Parameter<std::string> pixel_format_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_HPP */
