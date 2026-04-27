/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, TECNALIA. All rights reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gst_video_recorder_op_pydoc.hpp"

#include <map>
#include <string>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "../gst_video_recorder_op.hpp"
#include "../../operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace holoscan {

class PyGstVideoRecorderOp : public GstVideoRecorderOp {
 public:
  using GstVideoRecorderOp::GstVideoRecorderOp;

  PyGstVideoRecorderOp(Fragment* fragment,
                       const py::args& args,
                       const std::string& encoder = "nvh264",
                       const std::string& format = "RGBA",
                       const std::string& framerate = "30/1",
                       size_t max_buffers = 10,
                       bool block = true,
                       const std::string& filename = "output.mp4",
                       std::map<std::string, std::string> properties = {},
                       const std::string& name = "gst_video_recorder")
      : GstVideoRecorderOp(ArgList{Arg{"encoder", encoder},
                                   Arg{"format", format},
                                   Arg{"framerate", framerate},
                                   Arg{"max-buffers", max_buffers},
                                   Arg{"block", block},
                                   Arg{"filename", filename},
                                   Arg{"properties", properties},
                                   }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_holoscan_gstreamer_bridge, m) {
  m.doc() = R"pbdoc(
        Python bindings for the HoloHub GStreamer bridge
        ---------------------------------------
        Exposes GstVideoRecorderOp for direct use from Python.
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<GstVideoRecorderOp,
             PyGstVideoRecorderOp,
             Operator,
             std::shared_ptr<GstVideoRecorderOp>>(
      m, "GstVideoRecorderOp", doc::GstVideoRecorderOp::doc_GstVideoRecorderOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    size_t,
                    bool,
                    const std::string&,
                    std::map<std::string, std::string>,
                    const std::string&>(),
           "fragment"_a,
           "encoder"_a = "nvh264"s,
           "format"_a = "RGBA"s,
           "framerate"_a = "30/1"s,
           "max_buffers"_a = size_t(10),
           "block"_a = true,
           "filename"_a = "output.mp4"s,
           "properties"_a = std::map<std::string, std::string>{},
           "name"_a = "gst_video_recorder"s,
          doc::GstVideoRecorderOp::doc_GstVideoRecorderOp)
      .def("initialize",
          &GstVideoRecorderOp::initialize,
          doc::GstVideoRecorderOp::doc_initialize)
       .def("setup",
          &GstVideoRecorderOp::setup,
          "spec"_a,
          doc::GstVideoRecorderOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan
