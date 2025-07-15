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

#include "../vtk_renderer.hpp"
#include "./vtk_renderer_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <vector>
#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include "../../operator_util.hpp"

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

class PyVtkRendererOp : public VtkRendererOp {
 public:
  /* Inherit the constructors */
  using VtkRendererOp::VtkRendererOp;

  // Define a constructor that fully initializes the object.
  PyVtkRendererOp(Fragment* fragment, const py::args& args,
                  const uint32_t width,
                  const uint32_t height,
                  const std::vector<std::string>& labels = {std::string("")},
                  const std::string& window_name = "VTK (Kitware)",
                  const std::string& name = "vtk_renderer")
      : VtkRendererOp(ArgList{Arg{"width", width},
                              Arg{"height", height},
                              Arg{"labels", labels},
                              Arg{"window_name", window_name}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};


PYBIND11_MODULE(_vtk_renderer, m) {
  m.doc() = R"pbdoc(
        VtkRendererOp Python Bindings
        -------------------------------------
        .. currentmodule:: _vtk_renderer
    )pbdoc";
  py::class_<VtkRendererOp,
             PyVtkRendererOp,
             Operator,
             std::shared_ptr<VtkRendererOp>>(
      m,
      "VtkRendererOp",
      doc::VtkRendererOp::doc_VtkRendererOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const uint32_t,
                    const uint32_t,
                    const std::vector<std::string>&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "width"_a,
           "height"_a,
           "labels"_a,  // = {std::string("")},
           "window_name"_a = "VTK (Kitware)"s,
           "name"_a = "vtk_renderer"s,
           doc::VtkRendererOp::doc_VtkRendererOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops

