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

#include "../video_encoder_request.hpp"
#include "../../../operator_util.hpp"
#include "../video_encoder_utils.hpp"
#include "./video_encoder_request_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <variant>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;
using nvidia::gxf::EncoderConfig;
using nvidia::gxf::EncoderInputFormat;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyVideoEncoderRequestOp : public VideoEncoderRequestOp {
 public:
  /* Inherit the constructors */
  using VideoEncoderRequestOp::VideoEncoderRequestOp;

  // Define a constructor that fully initializes the object.
  PyVideoEncoderRequestOp(
      Fragment* fragment, const py::args& args,
      const std::shared_ptr<holoscan::ops::VideoEncoderContext>& videoencoder_context,
      const uint32_t input_height, const uint32_t input_width,
      const uint32_t inbuf_storage_type = 1u, const int32_t codec = 0,
      const std::variant<std::string, EncoderInputFormat> input_format = EncoderInputFormat::kNV12,
      const int32_t profile = 2, const int32_t bitrate = 20000000, const int32_t framerate = 30,
      const uint32_t qp = 20u, const int32_t hw_preset_type = 0, const int32_t level = 14,
      const int32_t iframe_interval = 30, const int32_t rate_control_mode = 1,
      const std::variant<std::string, EncoderConfig> config = EncoderConfig::kCustom,
      const std::string& name = "video_decoder_request")
      : VideoEncoderRequestOp(ArgList{Arg{"videoencoder_context", videoencoder_context},
                                      Arg{"input_height", input_height},
                                      Arg{"input_width", input_width},
                                      Arg{"inbuf_storage_type", inbuf_storage_type},
                                      Arg{"codec", codec},
                                      Arg{"profile", profile},
                                      Arg{"bitrate", bitrate},
                                      Arg{"framerate", framerate},
                                      Arg{"qp", qp},
                                      Arg{"hw_preset_type", hw_preset_type},
                                      Arg{"level", level},
                                      Arg{"iframe_interval", iframe_interval},
                                      Arg{"rate_control_mode", rate_control_mode}}) {
    if (std::holds_alternative<std::string>(input_format)) {
      this->add_arg(Arg("input_format") =
                        nvidia::gxf::ToEncoderInputFormat(std::get<std::string>(input_format)));
    } else {
      this->add_arg(Arg("input_format") = input_format);
    }

    if (std::holds_alternative<std::string>(config)) {
      this->add_arg(Arg("config") = nvidia::gxf::ToEncoderConfig(std::get<std::string>(config)));
    } else {
      this->add_arg(Arg("config") = config);
    }

    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_encoder_request, m) {
  m.doc() = R"pbdoc(
        VideoEncoderRequestOp Python Bindings
        -------------------------------------
        .. currentmodule:: _video_encoder_request
    )pbdoc";

  py::enum_<EncoderInputFormat>(m, "EncoderInputFormat")
      .value("nv12", EncoderInputFormat::kNV12)
      .value("nv24", EncoderInputFormat::kNV24)
      .value("yuv420planar", EncoderInputFormat::kYUV420PLANAR)
      .value("unsupported", EncoderInputFormat::kUnsupported);

  py::enum_<EncoderConfig>(m, "EncoderConfig")
      .value("iframecqp", EncoderConfig::kIFrameCQP)
      .value("pframecqp", EncoderConfig::kPFrameCQP)
      .value("custom", EncoderConfig::kCustom)
      .value("unsupported", EncoderConfig::kUnsupported);

  py::class_<VideoEncoderRequestOp,
             PyVideoEncoderRequestOp,
             GXFOperator,
             std::shared_ptr<VideoEncoderRequestOp>>(
      m, "VideoEncoderRequestOp", doc::VideoEncoderRequestOp::doc_VideoEncoderRequestOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<holoscan::ops::VideoEncoderContext>&,
                    const uint32_t,
                    const uint32_t,
                    const uint32_t,
                    const int32_t,
                    const std::variant<std::string, EncoderInputFormat>,
                    const int32_t,
                    const int32_t,
                    const int32_t,
                    const uint32_t,
                    const int32_t,
                    const int32_t,
                    const int32_t,
                    const int32_t,
                    const std::variant<std::string, EncoderConfig>&,
                    const std::string&>(),
           "fragment"_a,
           "videoencoder_context"_a,
           "input_height"_a,
           "input_width"_a,
           "inbuf_storage_type"_a = 1u,
           "codec"_a = 0,
           "input_format"_a = EncoderInputFormat::kNV12,
           "profile"_a = 2,
           "bitrate"_a = 20000000,
           "framerate"_a = 30,
           "qp"_a = 20u,
           "hw_preset_type"_a = 0,
           "level"_a = 14,
           "iframe_interval"_a = 30,
           "rate_control_mode"_a = 1,
           "config"_a = EncoderConfig::kCustom,
           "name"_a = "video_decoder_request"s,
           doc::VideoEncoderRequestOp::doc_VideoEncoderRequestOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
