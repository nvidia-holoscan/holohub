/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../adv_network_media_rx.h"
#include "./adv_network_media_rx_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "../../../operator_util.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

// Add advanced network headers
#include "advanced_network/common.h"
#include "advanced_network/types.h"

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

class PyAdvNetworkMediaRxOp : public AdvNetworkMediaRxOp {
 public:
  /* Inherit the constructors */
  using AdvNetworkMediaRxOp::AdvNetworkMediaRxOp;

  // Define a constructor that fully initializes the object.
  PyAdvNetworkMediaRxOp(Fragment* fragment, const py::args& args,
                        const std::string& interface_name = "",
                        uint16_t queue_id = default_queue_id, uint32_t frame_width = 1920,
                        uint32_t frame_height = 1080, uint32_t bit_depth = 8,
                        const std::string& video_format = "RGB888", bool hds = true,
                        const std::string& output_format = "video_buffer",
                        const std::string& memory_location = "device",
                        const std::string& name = "advanced_network_media_rx") {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());

    // Set parameters if provided
    if (!interface_name.empty()) { this->add_arg(Arg("interface_name", interface_name)); }
    this->add_arg(Arg("queue_id", queue_id));
    this->add_arg(Arg("frame_width", frame_width));
    this->add_arg(Arg("frame_height", frame_height));
    this->add_arg(Arg("bit_depth", bit_depth));
    this->add_arg(Arg("video_format", video_format));
    this->add_arg(Arg("hds", hds));
    this->add_arg(Arg("output_format", output_format));
    this->add_arg(Arg("memory_location", memory_location));
  }
};

PYBIND11_MODULE(_advanced_network_media_rx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Advanced Networking Media RX Operator Python Bindings
        ------------------------------------------------------------------
        .. currentmodule:: _advanced_network_media_rx

        This module provides Python bindings for the Advanced Networking Media RX operator,
        which receives video frames over Rivermax-enabled network infrastructure.
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<AdvNetworkMediaRxOp,
             PyAdvNetworkMediaRxOp,
             Operator,
             std::shared_ptr<AdvNetworkMediaRxOp>>(
      m, "AdvNetworkMediaRxOp", doc::AdvNetworkMediaRxOp::doc_AdvNetworkMediaRxOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    uint16_t,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "interface_name"_a = ""s,
           "queue_id"_a = AdvNetworkMediaRxOp::default_queue_id,
           "frame_width"_a = 1920,
           "frame_height"_a = 1080,
           "bit_depth"_a = 8,
           "video_format"_a = "RGB888"s,
           "hds"_a = true,
           "output_format"_a = "video_buffer"s,
           "memory_location"_a = "device"s,
           "name"_a = "advanced_network_media_rx"s,
           doc::AdvNetworkMediaRxOp::doc_AdvNetworkMediaRxOp_python)
      .def("initialize", &AdvNetworkMediaRxOp::initialize, doc::AdvNetworkMediaRxOp::doc_initialize)
      .def("setup", &AdvNetworkMediaRxOp::setup, "spec"_a, doc::AdvNetworkMediaRxOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
