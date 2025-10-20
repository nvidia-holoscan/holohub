/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../operator_util.hpp"

#include <gamma_correction/gamma_correction.hpp>
#include "gamma_correction_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <memory>
#include <optional>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

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
class PyGammaCorrectionOp : public GammaCorrectionOp {
 public:
  /* Inherit the constructors */
  using GammaCorrectionOp::GammaCorrectionOp;

  // Define a constructor that fully initializes the object.
  PyGammaCorrectionOp(Fragment* fragment, const py::args& args, const std::string& name,
                      const std::string& data_type, int32_t component_count, float gamma)
      : GammaCorrectionOp(ArgList{Arg{"data_type", data_type},
                                  Arg{"component_count", component_count},
                                  Arg{"gamma", gamma}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_gamma_correction, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK GammaCorrectionOpPython Bindings
        ---------------------------------------
        .. currentmodule:: _gamma_correction
    )pbdoc";

  py::class_<GammaCorrectionOp, PyGammaCorrectionOp, Operator, std::shared_ptr<GammaCorrectionOp>>(
      m, "GammaCorrectionOp", doc::GammaCorrectionOp::doc_GammaCorrectionOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    int32_t,
                    float>(),
           "fragment"_a,
           "name"_a = "gamma_correction"s,
           "data_type"_a = "uint8_t"s,
           "component_count"_a = 1,
           "gamma"_a = 2.2f,
           doc::GammaCorrectionOp::doc_GammaCorrectionOp_python)
      .def("setup", &GammaCorrectionOp::setup, "spec"_a, doc::GammaCorrectionOp::doc_setup);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
