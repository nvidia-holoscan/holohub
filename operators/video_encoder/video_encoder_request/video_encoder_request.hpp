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

#ifndef HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_REQUEST
#define HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_REQUEST

#include <memory>
#include <string>

#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"
#include "video_encoder_custom_params.hpp"

namespace holoscan::ops {

class VideoEncoderRequestOp : public ::holoscan::ops::GXFCodeletOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit VideoEncoderRequestOp(ArgT&& arg, ArgsT&&... args)
      : ::holoscan::ops::GXFCodeletOp("nvidia::gxf::VideoEncoderRequest", std::forward<ArgT>(arg),
                                      std::forward<ArgsT>(args)...) {}
  VideoEncoderRequestOp() : ::holoscan::ops::GXFCodeletOp("nvidia::gxf::VideoEncoderRequest") {}

  void setup(holoscan::OperatorSpec& spec) override {
    using namespace holoscan;
    // Ensure the parent class setup() is called before any additional setup code.
    ops::GXFCodeletOp::setup(spec);

    spec.param(input_format_,
               "input_format",
               "Input color format, nv12,nv24,yuv420planar",
               "Default: nv12",
               nvidia::gxf::EncoderInputFormat::kNV12);
    spec.param(config_,
               "config",
               "Preset of parameters, select from pframe_cqp, iframe_cqp, custom",
               "Preset of config",
               nvidia::gxf::EncoderConfig::kCustom);
  }

  void initialize() override {
    register_converter<nvidia::gxf::EncoderInputFormat>();
    register_converter<nvidia::gxf::EncoderConfig>();

    holoscan::ops::GXFCodeletOp::initialize();
  }

 private:
  Parameter<nvidia::gxf::EncoderInputFormat> input_format_;
  Parameter<nvidia::gxf::EncoderConfig> config_;
};

}  // namespace holoscan::ops
#endif  // HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_REQUEST
