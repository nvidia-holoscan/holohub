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

#include "../adv_network_rx.h"
#include "./adv_network_rx_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>

#include "../../operator_util.hpp"
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

class PyAdvNetworkOpRx : public AdvNetworkOpRx {
 public:
  /* Inherit the constructors */
  using AdvNetworkOpRx::AdvNetworkOpRx;

  // Define a constructor that fully initializes the object.
  PyAdvNetworkOpRx(Fragment* fragment, const py::args& args,
                   const std::unordered_set<std::string>& output_ports_list,
                   const std::string& name)
      : AdvNetworkOpRx(output_ports_list, ArgList{}) {
    this->add_arg(fragment->from_config("advanced_network"));
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_advanced_network_rx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _advanced_network_rx
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

  py::class_<AdvNetworkOpRx, PyAdvNetworkOpRx, Operator, std::shared_ptr<AdvNetworkOpRx>>(
      m, "AdvNetworkOpRx", doc::AdvNetworkOpRx::doc_AdvNetworkOpRx)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::unordered_set<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "output_ports_list"_a = std::unordered_set<std::string>{},
           "name"_a = "advanced_network_rx"s,
           doc::AdvNetworkOpRx::doc_AdvNetworkOpRx_python)
      .def("initialize", &AdvNetworkOpRx::initialize, doc::AdvNetworkOpRx::doc_initialize)
      .def("setup", &AdvNetworkOpRx::setup, "spec"_a, doc::AdvNetworkOpRx::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
