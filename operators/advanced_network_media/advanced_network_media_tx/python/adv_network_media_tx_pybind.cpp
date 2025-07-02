/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../adv_network_media_tx.h"
#include "./adv_network_media_tx_pydoc.hpp"

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

class PyAdvNetworkMediaTxOp : public AdvNetworkMediaTxOp {
 public:
  /* Inherit the constructors */
  using AdvNetworkMediaTxOp::AdvNetworkMediaTxOp;

  // Define a constructor that fully initializes the object.
  PyAdvNetworkMediaTxOp(Fragment* fragment, const py::args& args,
                        const std::string& interface_name = "",
                        uint16_t queue_id = default_queue_id,
                        const std::string& video_format = "RGB888", uint32_t bit_depth = 8,
                        uint32_t frame_width = 1920, uint32_t frame_height = 1080,
                        const std::string& name = "advanced_network_media_tx") {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());

    // Set parameters if provided
    if (!interface_name.empty()) { this->add_arg(Arg("interface_name", interface_name)); }
    this->add_arg(Arg("queue_id", queue_id));
    this->add_arg(Arg("video_format", video_format));
    this->add_arg(Arg("bit_depth", bit_depth));
    this->add_arg(Arg("frame_width", frame_width));
    this->add_arg(Arg("frame_height", frame_height));
  }
};

PYBIND11_MODULE(_advanced_network_media_tx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Advanced Networking Media TX Operator Python Bindings
        ------------------------------------------------------------------
        .. currentmodule:: _advanced_network_media_tx
        
        This module provides Python bindings for the Advanced Networking Media TX operator,
        which transmits video frames over Rivermax-enabled network infrastructure.
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<AdvNetworkMediaTxOp,
             PyAdvNetworkMediaTxOp,
             Operator,
             std::shared_ptr<AdvNetworkMediaTxOp>>(
      m, "AdvNetworkMediaTxOp", doc::AdvNetworkMediaTxOp::doc_AdvNetworkMediaTxOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    uint16_t,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "interface_name"_a = ""s,
           "queue_id"_a = AdvNetworkMediaTxOp::default_queue_id,
           "video_format"_a = "RGB888"s,
           "bit_depth"_a = 8,
           "frame_width"_a = 1920,
           "frame_height"_a = 1080,
           "name"_a = "advanced_network_media_tx"s,
           doc::AdvNetworkMediaTxOp::doc_AdvNetworkMediaTxOp_python)
      .def("initialize", &AdvNetworkMediaTxOp::initialize, doc::AdvNetworkMediaTxOp::doc_initialize)
      .def("setup", &AdvNetworkMediaTxOp::setup, "spec"_a, doc::AdvNetworkMediaTxOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
