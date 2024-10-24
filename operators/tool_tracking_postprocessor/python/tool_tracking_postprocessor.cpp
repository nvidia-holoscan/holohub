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

#include "../tool_tracking_postprocessor.hpp"
#include "./tool_tracking_postprocessor_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"
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

static const std::vector<std::vector<float>> VIZ_TOOL_DEFAULT_COLORS = {{0.12f, 0.47f, 0.71f},
                                                                        {0.20f, 0.63f, 0.17f},
                                                                        {0.89f, 0.10f, 0.11f},
                                                                        {1.00f, 0.50f, 0.00f},
                                                                        {0.42f, 0.24f, 0.60f},
                                                                        {0.69f, 0.35f, 0.16f},
                                                                        {0.65f, 0.81f, 0.89f},
                                                                        {0.70f, 0.87f, 0.54f},
                                                                        {0.98f, 0.60f, 0.60f},
                                                                        {0.99f, 0.75f, 0.44f},
                                                                        {0.79f, 0.70f, 0.84f},
                                                                        {1.00f, 1.00f, 0.60f}};

class PyToolTrackingPostprocessorOp : public ToolTrackingPostprocessorOp {
 public:
  /* Inherit the constructors */
  using ToolTrackingPostprocessorOp::ToolTrackingPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PyToolTrackingPostprocessorOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<Allocator> device_allocator,
      float min_prob = 0.5f,
      std::vector<std::vector<float>> overlay_img_colors = VIZ_TOOL_DEFAULT_COLORS,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "tool_tracking_postprocessor")
      : ToolTrackingPostprocessorOp(ArgList{Arg{"device_allocator", device_allocator},
                                            Arg{"min_prob", min_prob},
                                            Arg{"overlay_img_colors", overlay_img_colors}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_tool_tracking_postprocessor, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _tool_tracking_postprocessor
        .. autosummary::
           :toctree: _generate
           ToolTrackingPostprocessorOp
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<ToolTrackingPostprocessorOp,
             PyToolTrackingPostprocessorOp,
             Operator,
             std::shared_ptr<ToolTrackingPostprocessorOp>>(
      m,
      "ToolTrackingPostprocessorOp",
      doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<Allocator>,
                    float,
                    std::vector<std::vector<float>>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "device_allocator"_a,
           "min_prob"_a = 0.5f,
           "overlay_img_colors"_a = VIZ_TOOL_DEFAULT_COLORS,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "tool_tracking_postprocessor"s,
           doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp_python);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
