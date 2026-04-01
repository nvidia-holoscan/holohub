/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../tapnext_inference.hpp"
#include "./tapnext_inference_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

class PyTapNextInferenceOp : public TapNextInferenceOp {
 public:
  using TapNextInferenceOp::TapNextInferenceOp;

  PyTapNextInferenceOp(
      Fragment* fragment, const py::args& args,
      const std::string& model_file_path_init,
      const std::string& model_file_path_fwd,
      const std::string& engine_cache_dir,
      const std::vector<std::string>& input_tensor_names_init,
      const std::vector<std::string>& input_binding_names_init,
      const std::vector<std::string>& output_tensor_names_init,
      const std::vector<std::string>& output_binding_names_init,
      const std::vector<std::string>& input_tensor_names_fwd,
      const std::vector<std::string>& input_binding_names_fwd,
      const std::vector<std::string>& output_tensor_names_fwd,
      const std::vector<std::string>& output_binding_names_fwd,
      const std::vector<std::string>& state_tensor_names,
      std::shared_ptr<holoscan::Allocator> pool,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool,
      const std::string& plugins_lib_namespace = "",
      bool force_engine_update = false,
      bool enable_fp16 = false,
      bool verbose = false,
      bool relaxed_dimension_check = true,
      int64_t max_workspace_size = 67108864l,
      int32_t max_batch_size = 1,
      int32_t grid_size = 15,
      int32_t grid_height = 256,
      int32_t grid_width = 256,
      const std::string& name = "tapnext_inference")
      : TapNextInferenceOp(ArgList{
            Arg{"model_file_path_init", model_file_path_init},
            Arg{"model_file_path_fwd", model_file_path_fwd},
            Arg{"engine_cache_dir", engine_cache_dir},
            Arg{"input_tensor_names_init", input_tensor_names_init},
            Arg{"input_binding_names_init", input_binding_names_init},
            Arg{"output_tensor_names_init", output_tensor_names_init},
            Arg{"output_binding_names_init", output_binding_names_init},
            Arg{"input_tensor_names_fwd", input_tensor_names_fwd},
            Arg{"input_binding_names_fwd", input_binding_names_fwd},
            Arg{"output_tensor_names_fwd", output_tensor_names_fwd},
            Arg{"output_binding_names_fwd", output_binding_names_fwd},
            Arg{"state_tensor_names", state_tensor_names},
            Arg{"pool", pool},
            Arg{"cuda_stream_pool", cuda_stream_pool},
            Arg{"plugins_lib_namespace", plugins_lib_namespace},
            Arg{"force_engine_update", force_engine_update},
            Arg{"enable_fp16", enable_fp16},
            Arg{"verbose", verbose},
            Arg{"relaxed_dimension_check", relaxed_dimension_check},
            Arg{"max_workspace_size", max_workspace_size},
            Arg{"max_batch_size", max_batch_size},
            Arg{"grid_size", grid_size},
            Arg{"grid_height", grid_height},
            Arg{"grid_width", grid_width}
        }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_tapnext_inference, m) {
  m.doc() = R"pbdoc(
        TapNext Inference Operator Python Bindings
        ---------------------------------------
        .. currentmodule:: _tapnext_inference
        .. autosummary::
           :toctree: _generate
           TapNextInferenceOp
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<TapNextInferenceOp,
             PyTapNextInferenceOp,
             GXFOperator,
             std::shared_ptr<TapNextInferenceOp>>(
      m, "TapNextInferenceOp", doc::TapNextInferenceOp::doc_TapNextInferenceOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    std::shared_ptr<holoscan::Allocator>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&,
                    bool,
                    bool,
                    bool,
                    bool,
                    int64_t,
                    int32_t,
                    int32_t,
                    int32_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "model_file_path_init"_a,
           "model_file_path_fwd"_a,
           "engine_cache_dir"_a,
           "input_tensor_names_init"_a,
           "input_binding_names_init"_a,
           "output_tensor_names_init"_a,
           "output_binding_names_init"_a,
           "input_tensor_names_fwd"_a,
           "input_binding_names_fwd"_a,
           "output_tensor_names_fwd"_a,
           "output_binding_names_fwd"_a,
           "state_tensor_names"_a,
           "pool"_a,
           "cuda_stream_pool"_a,
           "plugins_lib_namespace"_a = "",
           "force_engine_update"_a = false,
           "enable_fp16"_a = false,
           "verbose"_a = false,
           "relaxed_dimension_check"_a = true,
           "max_workspace_size"_a = 67108864l,
           "max_batch_size"_a = 1,
           "grid_size"_a = 15,
           "grid_height"_a = 256,
           "grid_width"_a = 256,
           "name"_a = "tapnext_inference"s,
           doc::TapNextInferenceOp::doc_TapNextInferenceOp_python)
      .def_property_readonly("gxf_typename",
                             &TapNextInferenceOp::gxf_typename,
                             doc::TapNextInferenceOp::doc_gxf_typename)
      .def("initialize",
           &TapNextInferenceOp::initialize,
           doc::TapNextInferenceOp::doc_initialize)
      .def("setup",
           &TapNextInferenceOp::setup,
           "spec"_a,
           doc::TapNextInferenceOp::doc_setup);
}

}  // namespace holoscan::ops
