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

#include "../segmentation_postprocessor.hpp"

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

class PyOrsiSegmentationPostprocessorOp : public orsi::SegmentationPostprocessorOp {
 public:
  /* Inherit the constructors */
  using orsi::SegmentationPostprocessorOp::SegmentationPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PyOrsiSegmentationPostprocessorOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<::holoscan::Allocator> allocator,
      const std::string& in_tensor_name = "", const std::string& network_output_type = "softmax"s,
      const std::string& data_format = "hwc"s, const std::string& out_tensor_name = ""s,
      const std::vector<int32_t> output_roi_rect = {},
      const std::vector<int32_t> output_img_size = {},
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "segmentation_postprocessor"s)
      : orsi::SegmentationPostprocessorOp(ArgList{Arg{"in_tensor_name", in_tensor_name},
                                                  Arg{"network_output_type", network_output_type},
                                                  Arg{"data_format", data_format},
                                                  Arg{"out_tensor_name", out_tensor_name},
                                                  Arg{"output_roi_rect", output_roi_rect},
                                                  Arg{"output_img_size", output_img_size},
                                                  Arg{"allocator", allocator}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_orsi_segmentation_postprocessor, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _orsi_segmentation_postprocessor
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

  py::class_<orsi::SegmentationPostprocessorOp,
             PyOrsiSegmentationPostprocessorOp,
             Operator,
             std::shared_ptr<orsi::SegmentationPostprocessorOp>>(
      m,
      "OrsiSegmentationPostprocessorOp",
      doc::OrsiSegmentationPostprocessorOp::doc_OrsiSegmentationPostprocessorOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::vector<int32_t>,
                    const std::vector<int32_t>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_name"_a = ""s,
           "network_output_type"_a = "softmax"s,
           "data_format"_a = "hwc"s,
           "out_tensor_name"_a = ""s,
           "output_roi_rect"_a = std::vector<int32_t>{},
           "output_img_size"_a = std::vector<int32_t>{},
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "segmentation_postprocessor"s,
           doc::OrsiSegmentationPostprocessorOp::doc_OrsiSegmentationPostprocessorOp_python)
      .def("setup",
           &orsi::SegmentationPostprocessorOp::setup,
           "spec"_a,
           doc::OrsiSegmentationPostprocessorOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
