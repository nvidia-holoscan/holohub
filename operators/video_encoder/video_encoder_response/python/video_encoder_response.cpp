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

#include "../video_encoder_response.hpp"
#include "./video_encoder_response_pydoc.hpp"

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

class PyVideoEncoderResponseOp : public VideoEncoderResponseOp {
 public:
  /* Inherit the constructors */
  using VideoEncoderResponseOp::VideoEncoderResponseOp;

  // Define a constructor that fully initializes the object.
  PyVideoEncoderResponseOp(
      Fragment* fragment, const py::args& args, const std::shared_ptr<Allocator>& pool,
      const std::shared_ptr<holoscan::ops::VideoEncoderContext>& videoencoder_context,
      const uint32_t outbuf_storage_type, const std::string& name = "video_encoder_response")
      : VideoEncoderResponseOp(ArgList{Arg{"pool", pool},
                                       Arg{"videoencoder_context", videoencoder_context},
                                       Arg{"outbuf_storage_type", outbuf_storage_type}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_encoder_response, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _video_encoder_response
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<VideoEncoderResponseOp,
             PyVideoEncoderResponseOp,
             GXFOperator,
             std::shared_ptr<VideoEncoderResponseOp>>(
      m, "VideoEncoderResponseOp", doc::VideoEncoderResponseOp::doc_VideoEncoderResponseOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<Allocator>&,
                    const std::shared_ptr<holoscan::ops::VideoEncoderContext>&,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "pool"_a,
           "videoencoder_context"_a,
           "outbuf_storage_type"_a,
           "name"_a = "video_encoder_response"s,
           doc::VideoEncoderResponseOp::doc_VideoEncoderResponseOp_python)
      .def_property_readonly("gxf_typename",
                             &VideoEncoderResponseOp::gxf_typename,
                             doc::VideoEncoderResponseOp::doc_gxf_typename)
      .def("initialize",
           &VideoEncoderResponseOp::initialize,
           doc::VideoEncoderResponseOp::doc_initialize)
      .def("setup",
           &VideoEncoderResponseOp::setup,
           "spec"_a,
           doc::VideoEncoderResponseOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
