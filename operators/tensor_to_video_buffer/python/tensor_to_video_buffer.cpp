/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../tensor_to_video_buffer.hpp"
#include "./tensor_to_video_buffer_pydoc.hpp"
#include "../../operator_util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

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

class PyTensorToVideoBufferOp : public TensorToVideoBufferOp {
 public:
  /* Inherit the constructors */
  using TensorToVideoBufferOp::TensorToVideoBufferOp;

  // Define a constructor that fully initializes the object.
  PyTensorToVideoBufferOp(Fragment* fragment, const py::args& args,
                          const std::string& video_format,
                          const std::string& in_tensor_name = std::string(),
                          const std::string& name = "tensor_to_video_buffer")
      : TensorToVideoBufferOp(
            ArgList{Arg{"in_tensor_name", in_tensor_name}, Arg{"video_format", video_format}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_tensor_to_video_buffer, m) {
  m.doc() = R"pbdoc(
        TensorToVideoBufferOp Python Bindings
        -------------------------------------
        .. currentmodule:: _tensor_to_video_buffer
    )pbdoc";
  py::class_<TensorToVideoBufferOp,
             PyTensorToVideoBufferOp,
             Operator,
             std::shared_ptr<TensorToVideoBufferOp>>(
      m, "TensorToVideoBufferOp", doc::TensorToVideoBufferOp::doc_TensorToVideoBufferOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "video_format"_a,
           "in_tensor_name"_a = std::string(),
           "name"_a = "tensor_to_video_buffer"s,
           doc::TensorToVideoBufferOp::doc_TensorToVideoBufferOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
