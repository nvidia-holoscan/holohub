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

#include "../visualizer_icardio.hpp"
#include "./visualizer_icardio_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

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

class PyBasicNetworkOpRx : public BasicNetworkOpRx {
 public:
  /* Inherit the constructors */
  using BasicNetworkOpRx::BasicNetworkOpRx;

  // Define a constructor that fully initializes the object.
  PyBasicNetworkOpRx(Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
                        const std::vector<std::string>& in_tensor_names = {std::string("")},
                        const std::vector<std::string>& out_tensor_names = {std::string("")},
                        bool input_on_cuda = false,

                        const std::string& name = "visualizer_icardio")
      : BasicNetworkOpRx(ArgList{Arg{"allocator", allocator},
                                  Arg{"in_tensor_names", in_tensor_names},
                                  Arg{"out_tensor_names", out_tensor_names},
                                  Arg{"input_on_cuda", input_on_cuda}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

PYBIND11_MODULE(_basic_network_rx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _basic_network_rx
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

  py::class_<BasicNetworkOpRx,
             PyBasicNetworkOpRx,
             Operator,
             std::shared_ptr<BasicNetworkOpRx>>(
      m, "BasicNetworkOpRx", doc::BasicNetworkOpRx::doc_BasicNetworkOpRx)
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_names"_a,   // = {std::string("")},
           "out_tensor_names"_a,  // = {std::string("")},
           "input_on_cuda"_a = false,
           "name"_a = "visualizer_icardio"s,
           doc::BasicNetworkOpRx::doc_BasicNetworkOpRx_python)
      .def("initialize", &BasicNetworkOpRx::initialize, doc::BasicNetworkOpRx::doc_initialize)
      .def("setup", &BasicNetworkOpRx::setup, "spec"_a, doc::BasicNetworkOpRx::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
