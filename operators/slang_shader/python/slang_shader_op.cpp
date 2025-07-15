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

#include "../slang_shader_op.hpp"
#include "./slang_shader_op_pydoc.hpp"

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
class PySlangShaderOp : public SlangShaderOp {
 public:
  /* Inherit the constructors */
  using SlangShaderOp::SlangShaderOp;

  // Define a constructor that fully initializes the object.
  PySlangShaderOp(Fragment* fragment, const py::args& args, const std::string& shader_source,
                  const std::string& shader_source_file, const std::string& name = "slang_shader",
                  std::optional<std::shared_ptr<Allocator>> allocator = std::nullopt)
      : SlangShaderOp(ArgList{Arg{"shader_source", shader_source},
                              Arg{"shader_source_file", shader_source_file}}) {
    if (allocator.has_value()) {
      this->add_arg(Arg{"allocator", allocator.value()});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_slang_shader, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK SlangShaderOpPython Bindings
        ---------------------------------------
        .. currentmodule:: _slang_shader
    )pbdoc";

  py::class_<SlangShaderOp, PySlangShaderOp, Operator, std::shared_ptr<SlangShaderOp>>(
      m, "SlangShaderOp", doc::SlangShaderOp::doc_SlangShaderOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    std::optional<std::shared_ptr<Allocator>>>(),
           "fragment"_a,
           "shader_source"_a = "",
           "shader_source_file"_a = "",
           "name"_a = "slang_shader"s,
           "allocator"_a = py::none(),
           doc::SlangShaderOp::doc_SlangShaderOp_python)
      .def("setup", &SlangShaderOp::setup, "spec"_a, doc::SlangShaderOp::doc_setup);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
