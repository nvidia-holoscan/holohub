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

#include "lstm_tensor_rt_inference.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan::ops {

void LSTMTensorRTInferenceOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("source_video");
  auto& out_tensor = spec.output<gxf::Entity>("tensor");

  spec.param(
      model_file_path_, "model_file_path", "Model File Path", "Path to ONNX model to be loaded.");
  spec.param(engine_cache_dir_,
             "engine_cache_dir",
             "Engine Cache Directory",
             "Path to a folder containing cached engine files to be serialized and loaded from.");
  spec.param(plugins_lib_namespace_,
             "plugins_lib_namespace",
             "Plugins Lib Namespace",
             "Namespace used to register all the plugins in this library.",
             std::string(""));
  spec.param(force_engine_update_,
             "force_engine_update",
             "Force Engine Update",
             "Always update engine regard less of existing engine file. "
             "Such conversion may take minutes. Default to false.",
             false);

  spec.param(input_tensor_names_,
             "input_tensor_names",
             "Input Tensor Names",
             "Names of input tensors in the order to be fed into the model.");
  spec.param(input_binding_names_,
             "input_binding_names",
             "Input Binding Names",
             "Names of input bindings as in the model in the same order of "
             "what is provided in input_tensor_names.");

  spec.param(input_state_tensor_names_,
             "input_state_tensor_names",
             "Input State Tensor Names",
             "Names of input state tensors that are used internally by TensorRT.",
             std::vector<std::string>{});

  spec.param(output_tensor_names_,
             "output_tensor_names",
             "Output Tensor Names",
             "Names of output tensors in the order to be retrieved "
             "from the model.");
  spec.param(output_state_tensor_names_,
             "output_state_tensor_names",
             "Output State Tensor Names",
             "Names of output state tensors that are used internally by TensorRT.",
             std::vector<std::string>{});

  spec.param(output_binding_names_,
             "output_binding_names",
             "Output Binding Names",
             "Names of output bindings in the model in the same "
             "order of of what is provided in output_tensor_names.");
  spec.param(pool_, "pool", "Pool", "Allocator instance for output tensors.");
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "Cuda Stream Pool",
             "Instance of gxf::CudaStreamPool to allocate CUDA stream.");

  spec.param(max_workspace_size_,
             "max_workspace_size",
             "Max Workspace Size",
             "Size of working space in bytes. Default to 64MB",
             67108864l);
  spec.param(dla_core_,
             "dla_core",
             "DLA Core",
             "DLA Core to use. Fallback to GPU is always enabled. "
             "Default to use GPU only.",
             ParameterFlag::kOptional);
  spec.param(max_batch_size_,
             "max_batch_size",
             "Max Batch Size",
             "Maximum possible batch size in case the first dimension is "
             "dynamic and used as batch size.",
             1);
  spec.param(enable_fp16_,
             "enable_fp16_",
             "Enable FP16 Mode",
             "Enable inference with FP16 and FP32 fallback.",
             false);

  spec.param(verbose_,
             "verbose",
             "Verbose",
             "Enable verbose logging on console. Default to false.",
             false);
  spec.param(relaxed_dimension_check_,
             "relaxed_dimension_check",
             "Relaxed Dimension Check",
             "Ignore dimensions of 1 for input tensor dimension check.",
             true);

  spec.param(rx_, "rx", "RX", "List of receivers to take input tensors", {&in_tensor});
  spec.param(tx_, "tx", "TX", "Transmitter to publish output tensors", &out_tensor);

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

}  // namespace holoscan::ops
