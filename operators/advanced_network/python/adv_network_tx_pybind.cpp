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

#include "../adv_network_tx.h"
#include "./adv_network_tx_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

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

class PyAdvNetworkOpTx : public AdvNetworkOpTx {
 public:
  /* Inherit the constructors */
  using AdvNetworkOpTx::AdvNetworkOpTx;

  // Define a constructor that fully initializes the object.
  PyAdvNetworkOpTx(Fragment* fragment, const py::args& args,
                   const std::string& name = "advanced_network_tx") {
    this->add_arg(fragment->from_config("advanced_network"));
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_advanced_network_tx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _advanced_network_tx
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

  py::class_<AdvNetworkOpTx, PyAdvNetworkOpTx, Operator, std::shared_ptr<AdvNetworkOpTx>>(
      m, "AdvNetworkOpTx", doc::AdvNetworkOpTx::doc_AdvNetworkOpTx)
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           "fragment"_a,
           "name"_a = "advanced_network_tx"s,
           doc::AdvNetworkOpTx::doc_AdvNetworkOpTx_python)
      .def("initialize", &AdvNetworkOpTx::initialize, doc::AdvNetworkOpTx::doc_initialize)
      .def("setup", &AdvNetworkOpTx::setup, "spec"_a, doc::AdvNetworkOpTx::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
