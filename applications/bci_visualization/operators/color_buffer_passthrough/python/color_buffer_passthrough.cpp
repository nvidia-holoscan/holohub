/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
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

#include "../cpp/color_buffer_passthrough.hpp"

#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace holoscan::ops {

using pybind11::literals::operator""_a;

class PyColorBufferPassthroughOp : public ColorBufferPassthroughOp {
 public:
  using ColorBufferPassthroughOp::ColorBufferPassthroughOp;

  PyColorBufferPassthroughOp(Fragment* fragment, const py::args& args,
                             const std::string& name = "color_buffer_passthrough")
      : ColorBufferPassthroughOp(ArgList{}) {
    (void)args;
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_color_buffer_passthrough, m) {
  py::class_<ColorBufferPassthroughOp,
             PyColorBufferPassthroughOp,
             Operator,
             std::shared_ptr<ColorBufferPassthroughOp>>(
      m, "ColorBufferPassthroughOp")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           "fragment"_a,
           "name"_a = std::string("color_buffer_passthrough"))
      .def("setup", &ColorBufferPassthroughOp::setup, "spec"_a);
}

}  // namespace holoscan::ops


