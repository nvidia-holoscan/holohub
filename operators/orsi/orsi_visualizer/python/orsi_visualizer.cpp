/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for c++ stl

#include <memory>
#include <string>

#include "./pydoc.hpp"

#include "../../operator_util.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "../orsi_visualizer.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyOrsiVisualizationOp : public orsi::OrsiVisualizationOp {
 public:
  /* Inherit the constructors */
  using orsi::OrsiVisualizationOp::OrsiVisualizationOp;

  // Define a constructor that fully initializes the object.
  PyOrsiVisualizationOp(Fragment* fragment, const py::args& args,
                        std::vector<holoscan::IOSpec*> receivers = std::vector<holoscan::IOSpec*>(),
                        bool swizzle_video = false, const std::string& stl_file_path = "",
                        std::vector<std::string> stl_names = {},
                        std::vector<std::vector<int32_t>> stl_colors = {},
                        std::vector<int> stl_keys = {},
                        const std::string& registration_params_path = "",
                        const std::string& name = "orsi_viz_op"s)
      : orsi::OrsiVisualizationOp(ArgList{Arg{"swizzle_video", swizzle_video},
                                          Arg{"stl_file_path", stl_file_path},
                                          Arg{"registration_params_path", registration_params_path},
                                          Arg{"stl_names", stl_names},
                                          Arg{"stl_colors", stl_colors},
                                          Arg{"stl_keys", stl_keys}}) {
    if (receivers.size() > 0) {
      this->add_arg(Arg{"receivers", receivers});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_orsi_visualizer, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _orsi_visualizer
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

  py::class_<orsi::OrsiVisualizationOp,
             PyOrsiVisualizationOp,
             Operator,
             std::shared_ptr<orsi::OrsiVisualizationOp>>(
      m, "OrsiVisualizationOp", doc::OrsiVisualizationOp::doc_OrsiVisualizationOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::vector<holoscan::IOSpec*>,
                    bool,
                    const std::string&,
                    std::vector<std::string>,
                    std::vector<std::vector<int32_t>>,
                    std::vector<int>,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "receivers"_a = std::vector<holoscan::IOSpec*>(),
           "swizzle_video"_a = false,
           "stl_file_path"_a = ""s,
           "stl_names"_a = std::vector<std::string>(),
           "stl_colors"_a = std::vector<std::vector<int32_t>>{},
           "stl_keys"_a = std::vector<int>{},
           "registration_params_path"_a = ""s,
           "name"_a = "orsi_viz_op"s,
           doc::OrsiVisualizationOp::doc_OrsiVisualizationOp_python)
      .def("setup",
           &orsi::OrsiVisualizationOp::setup,
           "spec"_a,
           doc::OrsiVisualizationOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
