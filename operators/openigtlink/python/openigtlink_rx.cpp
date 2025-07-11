/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../openigtlink_rx.hpp"
#include "./openigtlink_rx_pydoc.hpp"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

class PyOpenIGTLinkRxOp : public OpenIGTLinkRxOp {
 public:
  /* Inherit the constructors */
  using OpenIGTLinkRxOp::OpenIGTLinkRxOp;

  // Define a constructor that fully initializes the object.
  PyOpenIGTLinkRxOp(Fragment* fragment, const py::args& args, std::shared_ptr<Allocator> allocator,
                    int port = 0, const std::string& out_tensor_name = std::string(""),
                    bool flip_width_height = true, const std::string& name = "openigtlink_rx")
      : OpenIGTLinkRxOp(ArgList{Arg{"allocator", allocator},
                                Arg{"port", port},
                                Arg{"out_tensor_name", out_tensor_name},
                                Arg{"flip_width_height", flip_width_height}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_openigtlink_rx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _openigtlink_rx
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

  py::class_<OpenIGTLinkRxOp, PyOpenIGTLinkRxOp, Operator, std::shared_ptr<OpenIGTLinkRxOp>>(
      m, "OpenIGTLinkRxOp", doc::OpenIGTLinkRxOp::doc_OpenIGTLinkRxOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<Allocator>,
                    int,
                    const std::string&,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "port"_a = 1,
           "out_tensor_name"_a = ""s,
           "flip_width_height"_a = true,
           "name"_a = "openigtlink_rx"s,
           doc::OpenIGTLinkRxOp::doc_OpenIGTLinkRxOp_python)
      .def("setup", &OpenIGTLinkRxOp::setup, "spec"_a, doc::OpenIGTLinkRxOp::doc_setup);
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan::ops
