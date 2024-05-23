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

#include "../video_decoder_response.hpp"
#include "./video_decoder_response_pydoc.hpp"

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

class PyVideoDecoderResponseOp : public VideoDecoderResponseOp {
 public:
  /* Inherit the constructors */
  using VideoDecoderResponseOp::VideoDecoderResponseOp;

  // Define a constructor that fully initializes the object.
  PyVideoDecoderResponseOp(
      Fragment* fragment, const py::args& args, const std::shared_ptr<Allocator>& pool,
      const uint32_t outbuf_storage_type,
      const std::shared_ptr<holoscan::ops::VideoDecoderContext>& videodecoder_context,
      const std::string& name = "video_decoder_response")
      : VideoDecoderResponseOp(ArgList{Arg{"pool", pool},
                                       Arg{"outbuf_storage_type", outbuf_storage_type},
                                       Arg{"videodecoder_context", videodecoder_context}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_decoder_response, m) {
  m.doc() = R"pbdoc(
        VideoDecoderResponseOp Python Bindings
        --------------------------------------
        .. currentmodule:: _video_decoder_response
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<VideoDecoderResponseOp,
             PyVideoDecoderResponseOp,
             GXFOperator,
             std::shared_ptr<VideoDecoderResponseOp>>(
      m, "VideoDecoderResponseOp", doc::VideoDecoderResponseOp::doc_VideoDecoderResponseOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<Allocator>&,
                    uint32_t,
                    const std::shared_ptr<holoscan::ops::VideoDecoderContext>&,
                    const std::string&>(),
           "fragment"_a,
           "pool"_a,
           "outbuf_storage_type"_a,
           "videodecoder_context"_a,
           "name"_a = "video_decoder_response"s,
           doc::VideoDecoderResponseOp::doc_VideoDecoderResponseOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
