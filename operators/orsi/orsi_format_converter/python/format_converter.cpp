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

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for c++ stl

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "./pydoc.hpp"

#include "../../operator_util.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "../format_converter.hpp"

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

class PyOrsiFormatConverterOp : public orsi::FormatConverterOp {
 public:
  /* Inherit the constructors */
  using orsi::FormatConverterOp::FormatConverterOp;

  // Define a constructor that fully initializes the object.
  PyOrsiFormatConverterOp(Fragment* fragment, const py::args& args,
                          std::shared_ptr<holoscan::Allocator> allocator,
                          const std::string& out_dtype, const std::string& in_dtype = "",
                          const std::string& in_tensor_name = "",
                          const std::string& out_tensor_name = "", float scale_min = 0.f,
                          float scale_max = 1.f, uint8_t alpha_value = static_cast<uint8_t>(255),
                          int32_t resize_height = 0, int32_t resize_width = 0,
                          int32_t resize_mode = 0,
                          const std::vector<int> out_channel_order = std::vector<int>{},
                          const std::vector<int> src_roi_rect = std::vector<int>{},
                          const std::vector<int> output_img_size = std::vector<int>{},
                          std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                          const std::string& name = "format_converter")
      : orsi::FormatConverterOp(ArgList{Arg{"in_tensor_name", in_tensor_name},
                                        Arg{"in_dtype", in_dtype},
                                        Arg{"out_tensor_name", out_tensor_name},
                                        Arg{"out_dtype", out_dtype},
                                        Arg{"scale_min", scale_min},
                                        Arg{"scale_max", scale_max},
                                        Arg{"alpha_value", alpha_value},
                                        Arg{"resize_mode", resize_mode},
                                        Arg{"out_channel_order", out_channel_order},
                                        Arg{"src_roi_rect", src_roi_rect},
                                        Arg{"output_img_size", output_img_size},
                                        Arg{"allocator", allocator}}) {
    if (cuda_stream_pool) {
      this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_orsi_format_converter, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _format_converter
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

  py::class_<orsi::FormatConverterOp,
             PyOrsiFormatConverterOp,
             Operator,
             std::shared_ptr<orsi::FormatConverterOp>>(
      m, "OrsiFormatConverterOp", doc::OrsiFormatConverterOp::doc_OrsiFormatConverterOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    float,
                    float,
                    uint8_t,
                    int32_t,
                    int32_t,
                    int32_t,
                    const std::vector<int>,
                    const std::vector<int>,
                    const std::vector<int>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "out_dtype"_a,
           "in_dtype"_a = ""s,
           "in_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "scale_min"_a = 0.f,
           "scale_max"_a = 1.f,
           "alpha_value"_a = static_cast<uint8_t>(255),
           "resize_height"_a = 0,
           "resize_width"_a = 0,
           "resize_mode"_a = 0,
           "out_channel_order"_a = std::vector<int>{},
           "src_roi_rect"_a = std::vector<int>{},
           "output_img_size"_a = std::vector<int>{},
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "format_converter"s,
           doc::OrsiFormatConverterOp::doc_OrsiFormatConverterOp_python)
      .def("initialize",
           &orsi::FormatConverterOp::initialize,
           doc::OrsiFormatConverterOp::doc_initialize)
      .def("setup",
           &orsi::FormatConverterOp::setup,
           "spec"_a,
           doc::OrsiFormatConverterOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
