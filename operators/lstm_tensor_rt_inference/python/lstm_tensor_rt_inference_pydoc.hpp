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

#ifndef PYHOLOHUB_OPERATORS_LSTM_TENSOR_RT_INFERENCE_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_LSTM_TENSOR_RT_INFERENCE_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace LSTMTensorRTInferenceOp {

PYDOC(LSTMTensorRTInferenceOp, R"doc(
Operator class to perform inference using an LSTM model.
)doc")

// PyLSTMTensorRTInferenceOp Constructor
PYDOC(LSTMTensorRTInferenceOp_python, R"doc(
Operator class to perform inference using an LSTM model.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
input_tensor_names : sequence of str
    Names of input tensors in the order to be fed into the model.
output_tensor_names : sequence of str
    Names of output tensors in the order to be retrieved from the model.
input_binding_names : sequence of str
    Names of input bindings as in the model in the same order of
    what is provided in `input_tensor_names`.
output_binding_names : sequence of str
    Names of output bindings as in the model in the same order of
    what is provided in `output_tensor_names`.
model_file_path : str
    Path to the ONNX model to be loaded.
engine_cache_dir : str
    Path to a folder containing cached engine files to be serialized and loaded from.
pool : ``holoscan.resources.Allocator``
    Allocator instance for output tensors.
cuda_stream_pool : ``holoscan.resources.CudaStreamPool``
    CudaStreamPool instance to allocate CUDA streams.
plugins_lib_namespace : str
    Namespace used to register all the plugins in this library.
input_state_tensor_names : sequence of str, optional
    Names of input state tensors that are used internally by TensorRT.
output_state_tensor_names : sequence of str, optional
    Names of output state tensors that are used internally by TensorRT.
force_engine_update : bool, optional
    Always update engine regardless of whether there is an existing engine file.
    Warning: this may take minutes to complete, so is False by default.
enable_fp16 : bool, optional
    Enable inference with FP16 and FP32 fallback.
verbose : bool, optional
    Enable verbose logging to the console.
relaxed_dimension_check : bool, optional
    Ignore dimensions of 1 for input tensor dimension check.
max_workspace_size : int, optional
    Size of working space in bytes.
max_batch_size : int, optional
    Maximum possible batch size in case the first dimension is dynamic and used
    as batch size.
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace LSTMTensorRTInferenceOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_LSTM_TENSOR_RT_INFERENCE_PYDOC_HPP
