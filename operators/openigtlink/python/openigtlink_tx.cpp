/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../openigtlink_tx.hpp"
#include "./openigtlink_tx_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

class PyOpenIGTLinkTxOp : public OpenIGTLinkTxOp {
 public:
  /* Inherit the constructors */
  using OpenIGTLinkTxOp::OpenIGTLinkTxOp;

  // Define a constructor that fully initializes the object.
  PyOpenIGTLinkTxOp(Fragment* fragment, const py::args& args,
                    std::vector<holoscan::IOSpec*> receivers = std::vector<holoscan::IOSpec*>(),
                    const std::string& host_name = std::string(""), int port = 0,
                    const std::string& device_name = std::string("Holoscan"),
                    const std::vector<std::string>& input_names = std::vector<std::string>{},
                    const std::string& name = "openigtlink_tx")
      : OpenIGTLinkTxOp(ArgList{Arg{"host_name", host_name},
                                Arg{"port", port},
                                Arg{"device_name", device_name},
                                Arg{"input_names", input_names}}) {
    if (receivers.size() > 0) { this->add_arg(Arg{"receivers", receivers}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_openigtlink_tx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _openigtlink_tx
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

  py::class_<OpenIGTLinkTxOp, PyOpenIGTLinkTxOp, Operator, std::shared_ptr<OpenIGTLinkTxOp>>(
      m, "OpenIGTLinkTxOp", doc::OpenIGTLinkTxOp::doc_OpenIGTLinkTxOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::vector<holoscan::IOSpec*>,
                    const std::string&,
                    int,
                    const std::string&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "receivers"_a = std::vector<holoscan::IOSpec*>(),
           "host_name"_a = ""s,
           "port"_a = 1,
           "device_name"_a = "Holoscan"s,
           "input_names"_a = std::vector<std::string>{},
           "name"_a = "openigtlink_tx"s,
           doc::OpenIGTLinkTxOp::doc_OpenIGTLinkTxOp_python)
      .def("setup", &OpenIGTLinkTxOp::setup, "spec"_a, doc::OpenIGTLinkTxOp::doc_setup);
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan::ops
