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

#include "./emergent_source_pydoc.hpp"
#include "../emergent_source.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"
#include <holoscan/core/fragment.hpp>
#include "holoscan/core/gxf/gxf_operator.hpp"
#include <holoscan/core/operator_spec.hpp>

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

class PyEmergentSourceOp : public EmergentSourceOp {
 public:
  /* Inherit the constructors */
  using EmergentSourceOp::EmergentSourceOp;

  // Define a constructor that fully initializes the object.
  PyEmergentSourceOp(Fragment* fragment, const py::args& args,
                     // defaults here should match constexpr values in EmergentSourceOp::Setup
                     uint32_t width = 4200, uint32_t height = 2160, uint32_t framerate = 240,
                     bool rdma = false, uint32_t exposure = 3072, uint32_t gain = 4095,
         const std::string& name = "emergent_source")
      : EmergentSourceOp(ArgList{Arg{"width", width},
                                 Arg{"height", height},
                                 Arg{"framerate", framerate},
                                 Arg{"rdma", rdma},
         Arg{"exposure", exposure},
         Arg{"gain", gain}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_emergent_source, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _emergent_source
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

  py::class_<EmergentSourceOp, PyEmergentSourceOp, GXFOperator, std::shared_ptr<EmergentSourceOp>>(
      m, "EmergentSourceOp", doc::EmergentSourceOp::doc_EmergentSourceOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
        uint32_t,
        uint32_t,
                    const std::string&>(),
           "fragment"_a,
           // defaults values here should match constexpr values in C++ EmergentSourceOp::Setup
           "width"_a = 4200,
           "height"_a = 2160,
           "framerate"_a = 240,
           "rdma"_a = false,
     "exposure"_a = 3072,
     "gain"_a = 4095,
           "name"_a = "emergent_source"s,
           doc::EmergentSourceOp::doc_EmergentSourceOp_python)
      .def_property_readonly(
          "gxf_typename", &EmergentSourceOp::gxf_typename, doc::EmergentSourceOp::doc_gxf_typename)
      .def("initialize", &EmergentSourceOp::initialize, doc::EmergentSourceOp::doc_initialize)
      .def("setup", &EmergentSourceOp::setup, "spec"_a, doc::EmergentSourceOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
