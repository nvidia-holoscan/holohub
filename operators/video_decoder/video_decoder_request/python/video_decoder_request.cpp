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

#include "../video_decoder_request.hpp"
#include "./video_decoder_request_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "../../../operator_util.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

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

class PyVideoDecoderRequestOp : public VideoDecoderRequestOp {
 public:
  /* Inherit the constructors */
  using VideoDecoderRequestOp::VideoDecoderRequestOp;

  // Define a constructor that fully initializes the object.
  PyVideoDecoderRequestOp(
      Fragment* fragment, const py::args& args, const uint32_t inbuf_storage_type,
      const std::shared_ptr<holoscan::AsynchronousCondition>& async_scheduling_term,
      const std::shared_ptr<holoscan::ops::VideoDecoderContext>& videodecoder_context,
      const uint32_t codec = 0u, const uint32_t disableDPB = 0u,
      const std::string& output_format = "nv12pl",
      const std::string& name = "video_decoder_request")
      : VideoDecoderRequestOp(ArgList{Arg{"inbuf_storage_type", inbuf_storage_type},
                                      Arg{"async_scheduling_term", async_scheduling_term},
                                      Arg{"videodecoder_context", videodecoder_context},
                                      Arg{"codec", codec},
                                      Arg{"disableDPB", disableDPB},
                                      Arg{"output_format", output_format}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_decoder_request, m) {
  m.doc() = R"pbdoc(
        VideoDecoderRequestOp Python Bindings
        -------------------------------------
        .. currentmodule:: _video_decoder_request
    )pbdoc";

  py::class_<VideoDecoderRequestOp,
             PyVideoDecoderRequestOp,
             GXFOperator,
             std::shared_ptr<VideoDecoderRequestOp>>(
      m, "VideoDecoderRequestOp", doc::VideoDecoderRequestOp::doc_VideoDecoderRequestOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    const std::shared_ptr<holoscan::AsynchronousCondition>&,
                    const std::shared_ptr<holoscan::ops::VideoDecoderContext>&,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "inbuf_storage_type"_a,
           "async_scheduling_term"_a,
           "videodecoder_context"_a,
           "codec"_a = 0u,
           "disableDPB"_a = 0u,
           "output_format"_a = "nv12pl",
           "name"_a = "video_decoder_request"s,
           doc::VideoDecoderRequestOp::doc_VideoDecoderRequestOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
