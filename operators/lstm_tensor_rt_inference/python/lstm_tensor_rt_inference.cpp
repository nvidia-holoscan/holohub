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

#include "../lstm_tensor_rt_inference.hpp"
#include "./lstm_tensor_rt_inference_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "../../operator_util.hpp"
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

class PyLSTMTensorRTInferenceOp : public LSTMTensorRTInferenceOp {
 public:
  /* Inherit the constructors */
  using LSTMTensorRTInferenceOp::LSTMTensorRTInferenceOp;

  // Define a constructor that fully initializes the object.
  PyLSTMTensorRTInferenceOp(
      Fragment* fragment, const py::args& args, const std::vector<std::string>& input_tensor_names,
      const std::vector<std::string>& output_tensor_names,
      const std::vector<std::string>& input_binding_names,
      const std::vector<std::string>& output_binding_names, const std::string& model_file_path,
      const std::string& engine_cache_dir, std::shared_ptr<holoscan::Allocator> pool,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool, std::optional<int32_t> dla_core,
      const std::string& plugins_lib_namespace = "",
      const std::vector<std::string>& input_state_tensor_names = std::vector<std::string>{},
      const std::vector<std::string>& output_state_tensor_names = std::vector<std::string>{},
      bool force_engine_update = false, bool enable_fp16_ = false, bool verbose = false,
      bool relaxed_dimension_check = true, int64_t max_workspace_size = 67108864l,
      int32_t max_batch_size = 1, const std::string& name = "lstm_tensor_rt_inference")
      : LSTMTensorRTInferenceOp(ArgList{Arg{"input_tensor_names", input_tensor_names},
                                        Arg{"output_tensor_names", output_tensor_names},
                                        Arg{"input_binding_names", input_binding_names},
                                        Arg{"output_binding_names", output_binding_names},
                                        Arg{"model_file_path", model_file_path},
                                        Arg{"engine_cache_dir", engine_cache_dir},
                                        Arg{"pool", pool},
                                        Arg{"cuda_stream_pool", cuda_stream_pool},
                                        Arg{"plugins_lib_namespace", plugins_lib_namespace},
                                        Arg{"input_state_tensor_names", input_state_tensor_names},
                                        Arg{"output_state_tensor_names", output_state_tensor_names},
                                        Arg{"force_engine_update", force_engine_update},
                                        Arg{"enable_fp16_", enable_fp16_},
                                        Arg{"verbose", verbose},
                                        Arg{"relaxed_dimension_check", relaxed_dimension_check},
                                        Arg{"max_workspace_size", max_workspace_size},
                                        Arg{"max_batch_size", max_batch_size}}) {
    if (dla_core.has_value()) {
      add_arg(Arg{"dla_core", dla_core.value()});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_lstm_tensor_rt_inference, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _lstm_tensor_rt_inference
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

  py::class_<LSTMTensorRTInferenceOp,
             PyLSTMTensorRTInferenceOp,
             GXFOperator,
             std::shared_ptr<LSTMTensorRTInferenceOp>>(
      m, "LSTMTensorRTInferenceOp", doc::LSTMTensorRTInferenceOp::doc_LSTMTensorRTInferenceOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<holoscan::Allocator>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    std::optional<int32_t>,
                    const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    bool,
                    bool,
                    bool,
                    int64_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "input_tensor_names"_a,
           "output_tensor_names"_a,
           "input_binding_names"_a,
           "output_binding_names"_a,
           "model_file_path"_a,
           "engine_cache_dir"_a,
           "pool"_a,
           "cuda_stream_pool"_a,
           "dla_core"_a = py::none(),
           "plugins_lib_namespace"_a = "",
           "input_state_tensor_names"_a = std::vector<std::string>{},
           "output_state_tensor_names"_a = std::vector<std::string>{},
           "force_engine_update"_a = false,
           "enable_fp16_"_a = false,
           "verbose"_a = false,
           "relaxed_dimension_check"_a = true,
           "max_workspace_size"_a = 67108864l,
           "max_batch_size"_a = 1,
           "name"_a = "lstm_tensor_rt_inference"s,
           doc::LSTMTensorRTInferenceOp::doc_LSTMTensorRTInferenceOp_python)
      .def_property_readonly("gxf_typename",
                             &LSTMTensorRTInferenceOp::gxf_typename,
                             doc::LSTMTensorRTInferenceOp::doc_gxf_typename)
      .def("initialize",
           &LSTMTensorRTInferenceOp::initialize,
           doc::LSTMTensorRTInferenceOp::doc_initialize)
      .def("setup",
           &LSTMTensorRTInferenceOp::setup,
           "spec"_a,
           doc::LSTMTensorRTInferenceOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
