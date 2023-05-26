/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./v4l2_video_capture_pydoc.hpp"
#include "../v4l2_video_capture.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include "holoscan/core/gxf/gxf_operator.hpp"
#include <holoscan/core/operator_spec.hpp>
#include "holoscan/core/resources/gxf/allocator.hpp"

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

class PyV4L2VideoCaptureOp : public V4L2VideoCaptureOp {
 public:
  /* Inherit the constructors */
  using V4L2VideoCaptureOp::V4L2VideoCaptureOp;

  // Define a constructor that fully initializes the object.
  PyV4L2VideoCaptureOp(Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
                 const std::string& device = "/dev/video0"s, uint32_t width = 1920,
                 uint32_t height = 1080, uint32_t num_buffers = 4,
                 const std::string& pixel_format = "RGBA32",
                 const std::string& name = "v4l2_video_capture")
      : V4L2VideoCaptureOp(ArgList{Arg{"allocator", allocator},
                               Arg{"device", device},
                               Arg{"width", width},
                               Arg{"height", height},
                               Arg{"numBuffers", num_buffers},
                               Arg{"pixel_format", pixel_format}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

PYBIND11_MODULE(_v4l2_video_capture, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _v4l2_video_capture
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

  py::class_<V4L2VideoCaptureOp, PyV4L2VideoCaptureOp, Operator,
             std::shared_ptr<V4L2VideoCaptureOp>>(
      m, "V4L2VideoCaptureOp", doc::V4L2VideoCaptureOp::doc_V4L2VideoCaptureOp)
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "device"_a = "0"s,
           "width"_a = 1920,
           "height"_a = 1080,
           "num_buffers"_a = 4,
           "pixel_format"_a = "RGBA32"s,
           "name"_a = "v4l2_video_capture"s,
           doc::V4L2VideoCaptureOp::doc_V4L2VideoCaptureOp_python)
      .def_property_readonly(
          "gxf_typename", &V4L2VideoCaptureOp::gxf_typename,
           doc::V4L2VideoCaptureOp::doc_gxf_typename)
      .def("initialize", &V4L2VideoCaptureOp::initialize, doc::V4L2VideoCaptureOp::doc_initialize)
      .def("setup", &V4L2VideoCaptureOp::setup, "spec"_a, doc::V4L2VideoCaptureOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
