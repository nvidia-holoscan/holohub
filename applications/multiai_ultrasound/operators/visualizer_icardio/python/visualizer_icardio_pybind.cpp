/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "operator_util.hpp"

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

class PyVisualizerICardioOp : public VisualizerICardioOp {
 public:
  /* Inherit the constructors */
  using VisualizerICardioOp::VisualizerICardioOp;

  // Define a constructor that fully initializes the object.
  PyVisualizerICardioOp(Fragment* fragment, const py::args& args,
                        std::shared_ptr<::holoscan::Allocator> allocator,
                        const std::vector<std::string>& in_tensor_names = {std::string("")},
                        const std::vector<std::string>& out_tensor_names = {std::string("")},
                        bool input_on_cuda = true,
                        std::string data_dir = "../data/multiai_ultrasound",
                        std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                        // TODO(grelee): handle receivers similarly to HolovizOp?  (default: {})
                        // TODO(grelee): handle transmitter similarly to HolovizOp?
                        const std::string& name = "visualizer_icardio")
      : VisualizerICardioOp(ArgList{Arg{"allocator", allocator},
                                    Arg{"in_tensor_names", in_tensor_names},
                                    Arg{"out_tensor_names", out_tensor_names},
                                    Arg{"input_on_cuda", input_on_cuda},
                                    Arg{"data_dir", data_dir}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_visualizer_icardio, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _visualizer_icardio
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

  py::class_<VisualizerICardioOp,
             PyVisualizerICardioOp,
             Operator,
             std::shared_ptr<VisualizerICardioOp>>(
      m, "VisualizerICardioOp", doc::VisualizerICardioOp::doc_VisualizerICardioOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    std::string,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_names"_a,   // = {std::string("")},
           "out_tensor_names"_a,  // = {std::string("")},
           "input_on_cuda"_a = true,
           "data_dir"_a,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "visualizer_icardio"s,
           doc::VisualizerICardioOp::doc_VisualizerICardioOp_python)
      .def("initialize", &VisualizerICardioOp::initialize, doc::VisualizerICardioOp::doc_initialize)
      .def("setup", &VisualizerICardioOp::setup, "spec"_a, doc::VisualizerICardioOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
