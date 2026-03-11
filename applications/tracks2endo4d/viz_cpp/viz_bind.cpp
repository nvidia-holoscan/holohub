/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "viz.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include "../../../operators/operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using holoscan::Allocator;
using holoscan::Arg;
using holoscan::ArgList;
using holoscan::Fragment;
using holoscan::Operator;
using holoscan::OperatorSpec;
using holoscan::ops::VizOp;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs */
class PyVizOp : public VizOp {
 public:
  /* Inherit the constructors */
  using VizOp::VizOp;

  // Define a constructor that fully initializes the object.
  PyVizOp(Fragment* fragment, const py::args& args, std::shared_ptr<Allocator> device_allocator,
          const std::string& name = "viz_op", uint32_t width = 1280, uint32_t height = 720,
          const std::string& window_title = "Holoviz", bool headless = false, bool verbose = false)
      : VizOp(ArgList{Arg{"device_allocator", device_allocator},
                      Arg{"width", width},
                      Arg{"height", height},
                      Arg{"window_title", window_title},
                      Arg{"headless", headless},
                      Arg{"verbose", verbose}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_tracks2endo4d_viz, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _tracks2endo4d_viz
        .. autosummary::
           :toctree: _generate
           VizOp
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<VizOp, PyVizOp, Operator, std::shared_ptr<VizOp>>(m, "VizOp", "Visualizer Operator")
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<Allocator>,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    bool,
                    bool>(),
           "fragment"_a,
           "device_allocator"_a,
           "name"_a = "viz_op"s,
           "width"_a = 1280,
           "height"_a = 720,
           "window_title"_a = "Holoviz"s,
           "headless"_a = false,
           "verbose"_a = false,
           "Visualizer Operator");
}
}  // namespace holoscan::ops
