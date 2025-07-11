/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../prohawkop.hpp"
#include "./prohawkop_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

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

class PyProhawkOp : public ProhawkOp {
 public:
  using ProhawkOp::ProhawkOp;

  explicit PyProhawkOp(Fragment* fragment, const py::args& args,
                       const std::string& name = "prohawk_video_processing")
      : ProhawkOp() {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_prohawk_video_processing, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _prohawk_video_processing
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

  py::class_<ProhawkOp, PyProhawkOp, Operator, std::shared_ptr<ProhawkOp>>(
      m, "ProhawkOp", doc::ProhawkOp::doc_ProhawkOp)
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           "fragment"_a,
           "name"_a = "prohawk_video_processing"s,
           doc::ProhawkOp::doc_ProhawkOp_python)
      .def("initialize", &ProhawkOp::initialize, doc::ProhawkOp::doc_initialize)
      .def("setup", &ProhawkOp::setup, "spec"_a, doc::ProhawkOp::doc_setup);
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan::ops
