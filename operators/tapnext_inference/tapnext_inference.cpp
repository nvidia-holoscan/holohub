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

#include "tapnext_inference.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan::ops {

void TapNextInferenceOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("receivers");
  auto& out_tensor = spec.output<gxf::Entity>("transmitter");

  spec.param(model_file_path_init_, "model_file_path_init", "Init Model File Path", "Path to ONNX model for initialization.");
  spec.param(model_file_path_fwd_, "model_file_path_fwd", "Forward Model File Path", "Path to ONNX model for forward tracking.");
  
  spec.param(engine_cache_dir_, "engine_cache_dir", "Engine Cache Directory", "Path to folder for cached engine files.");
  spec.param(plugins_lib_namespace_, "plugins_lib_namespace", "Plugins Lib Namespace", "", std::string(""));
  spec.param(force_engine_update_, "force_engine_update", "Force Engine Update", "", false);

  spec.param(input_tensor_names_init_, "input_tensor_names_init", "Init Input Tensor Names", "Input tensors for Init model.");
  spec.param(input_binding_names_init_, "input_binding_names_init", "Init Input Binding Names", "Input bindings for Init model.");
  spec.param(output_tensor_names_init_, "output_tensor_names_init", "Init Output Tensor Names", "Output tensors for Init model.");
  spec.param(output_binding_names_init_, "output_binding_names_init", "Init Output Binding Names", "Output bindings for Init model.");

  spec.param(input_tensor_names_fwd_, "input_tensor_names_fwd", "Fwd Input Tensor Names", "Input tensors for Fwd model.");
  spec.param(input_binding_names_fwd_, "input_binding_names_fwd", "Fwd Input Binding Names", "Input bindings for Fwd model.");
  spec.param(output_tensor_names_fwd_, "output_tensor_names_fwd", "Fwd Output Tensor Names", "Output tensors for Fwd model.");
  spec.param(output_binding_names_fwd_, "output_binding_names_fwd", "Fwd Output Binding Names", "Output bindings for Fwd model.");

  spec.param(state_tensor_names_, "state_tensor_names", "State Tensor Names", "List of tensor names that are treated as internal states (preserved across steps).");

  spec.param(pool_, "pool", "Pool", "Allocator instance.");
  spec.param(cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool", "CUDA Stream Pool");
  
  spec.param(max_workspace_size_, "max_workspace_size", "Max Workspace Size", "", 67108864l);
  spec.param(max_batch_size_, "max_batch_size", "Max Batch Size", "", 1);
  spec.param(enable_fp16_, "enable_fp16", "Enable FP16", "", false);
  spec.param(verbose_, "verbose", "Verbose", "", false);
  spec.param(relaxed_dimension_check_, "relaxed_dimension_check", "Relaxed Dimension Check", "", true);

  spec.param(grid_size_, "grid_size", "Grid Size", "Grid size for query points", 15);
  spec.param(grid_height_, "grid_height", "Grid Height", "Image height for grid generation", 256);
  spec.param(grid_width_, "grid_width", "Grid Width", "Image width for grid generation", 256);

  spec.param(rx_, "rx", "RX", "List of receivers", {&in_tensor});
  spec.param(tx_, "tx", "TX", "Transmitter", &out_tensor);
}

}  // namespace holoscan::ops

