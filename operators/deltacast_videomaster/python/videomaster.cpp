/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 DELTACAST.TV. All rights
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
#include "../videomaster_source.hpp"
#include "../videomaster_transmitter.hpp"
#include "videomaster_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"

#include <holoscan/core/fragment.hpp>
#include "holoscan/core/gxf/gxf_operator.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyVideoMasterSourceOp : public VideoMasterSourceOp {
 public:
  /* Inherit the constructors */
  using VideoMasterSourceOp::VideoMasterSourceOp;

  // Define a constructor that fully initializes the object.
  PyVideoMasterSourceOp(Fragment* fragment, const py::args& args, bool rdma = false,
                        uint32_t board = 0, uint32_t input = 0, uint32_t width = 1920,
                        uint32_t height = 1080, bool progressive = true,
                        uint32_t framerate = 60, std::shared_ptr<Allocator> pool = nullptr,
                        const std::string& name = "videomaster_source")
      : VideoMasterSourceOp(ArgList{
            Arg{"rdma", rdma},
            Arg{"board", board},
            Arg{"input", input},
            Arg{"width", width},
            Arg{"height", height},
            Arg{"progressive", progressive},
            Arg{"framerate", framerate},
            Arg{"pool", pool}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

class PyVideoMasterTransmitterOp : public VideoMasterTransmitterOp {
 public:
  /* Inherit the constructors */
  using VideoMasterTransmitterOp::VideoMasterTransmitterOp;

  // Define a constructor that fully initializes the object.
  PyVideoMasterTransmitterOp(Fragment* fragment, const py::args& args, bool rdma = false,
                             uint32_t board = 0, uint32_t output = 0, uint32_t width = 1920,
                             uint32_t height = 1080, bool progressive = true,
                             uint32_t framerate = 60, std::shared_ptr<Allocator> pool = nullptr,
                             bool enable_overlay = false,
                             const std::string& name = "videomaster_transmitter")
      : VideoMasterTransmitterOp(ArgList{Arg{"rdma", rdma},
                                         Arg{"board", board},
                                         Arg{"output", output},
                                         Arg{"width", width},
                                         Arg{"height", height},
                                         Arg{"progressive", progressive},
                                         Arg{"framerate", framerate},
                                         Arg{"pool", pool},
                                         Arg{"enable_overlay", enable_overlay}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_videomaster, m) {
  m.doc() = R"pbdoc(
         Holoscan SDK VideoMasterSourceOp Python Bindings
         ---------------------------------------
         .. currentmodule:: _videomaster
     )pbdoc";

  py::class_<VideoMasterSourceOp,
             PyVideoMasterSourceOp,
             GXFOperator,
             std::shared_ptr<VideoMasterSourceOp>>(
      m, "VideoMasterSourceOp", doc::VideoMasterSourceOp::doc_VideoMasterSourceOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    bool,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    uint32_t,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "rdma"_a = false,
           "board"_a = "0"s,
           "input"_a = "0"s,
           "width"_a = "0"s,
           "height"_a = "0"s,
           "progressive"_a = true,
           "framerate"_a = "60"s,
           "pool"_a,
           "name"_a = "videomaster_source"s,
           doc::VideoMasterSourceOp::doc_VideoMasterSourceOp_python)
      .def_property_readonly("gxf_typename",
                             &VideoMasterSourceOp::gxf_typename,
                             doc::VideoMasterSourceOp::doc_gxf_typename)
      .def("initialize", &VideoMasterSourceOp::initialize, doc::VideoMasterSourceOp::doc_initialize)
      .def("setup", &VideoMasterSourceOp::setup, "spec"_a, doc::VideoMasterSourceOp::doc_setup);

  py::class_<VideoMasterTransmitterOp,
             PyVideoMasterTransmitterOp,
             GXFOperator,
             std::shared_ptr<VideoMasterTransmitterOp>>(
      m, "VideoMasterTransmitterOp", doc::VideoMasterTransmitterOp::doc_VideoMasterTransmitterOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    bool,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    uint32_t,
                    std::shared_ptr<Allocator>,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "rdma"_a = false,
           "board"_a = "0"s,
           "output"_a = "0"s,
           "width"_a = "0"s,
           "height"_a = "0"s,
           "progressive"_a = true,
           "framerate"_a = "60"s,
           "pool"_a,
           "enable_overlay"_a = false,
           "name"_a = "videomaster_transmitter"s,
           doc::VideoMasterTransmitterOp::doc_VideoMasterTransmitterOp)
      .def_property_readonly("gxf_typename",
                             &VideoMasterTransmitterOp::gxf_typename,
                             doc::VideoMasterTransmitterOp::doc_gxf_typename)
      .def("initialize",
           &VideoMasterTransmitterOp::initialize,
           doc::VideoMasterTransmitterOp::doc_initialize)
      .def("setup",
           &VideoMasterTransmitterOp::setup,
           "spec"_a,
           doc::VideoMasterTransmitterOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
