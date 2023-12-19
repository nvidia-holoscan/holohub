/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "../segmentation_preprocessor.hpp"

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

class PyOrsiSegmentationPreprocessorOp : public orsi::SegmentationPreprocessorOp {
 public:
  /* Inherit the constructors */
  using  orsi::SegmentationPreprocessorOp::SegmentationPreprocessorOp;

  // Define a constructor that fully initializes the object.
  PyOrsiSegmentationPreprocessorOp(
      Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
      const std::string& in_tensor_name = "", const std::string& network_output_type = "softmax"s,
      const std::string& data_format = "hwc"s,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "segmentation_postprocessor"s)
      :  orsi::SegmentationPreprocessorOp(ArgList{Arg{"in_tensor_name", in_tensor_name},
                                            Arg{"network_output_type", network_output_type},
                                            Arg{"data_format", data_format},
                                            Arg{"allocator", allocator}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_orsi_segmentation_preprocessor, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _segmentation_postprocessor
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

  py::class_< orsi::SegmentationPreprocessorOp,
             PyOrsiSegmentationPreprocessorOp,
             Operator,
             std::shared_ptr< orsi::SegmentationPreprocessorOp>>(
      m,
      "OrsiSegmentationPreprocessorOp",
      doc::OrsiSegmentationPreprocessorOp::doc_OrsiSegmentationPreprocessorOp)
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_name"_a = ""s,
           "network_output_type"_a = "softmax"s,
           "data_format"_a = "hwc"s,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "segmentation_postprocessor"s,
           doc::OrsiSegmentationPreprocessorOp::doc_OrsiSegmentationPreprocessorOp_python)
      .def("setup",
           & orsi::SegmentationPreprocessorOp::setup,
           "spec"_a,
           doc::OrsiSegmentationPreprocessorOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
