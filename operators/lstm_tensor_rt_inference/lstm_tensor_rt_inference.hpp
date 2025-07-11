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

#ifndef HOLOSCAN_OPERATORS_LSTM_TENSOR_RT_INFERENCE_HPP
#define HOLOSCAN_OPERATORS_LSTM_TENSOR_RT_INFERENCE_HPP

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to perform the inference of the LSTM model.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::lstm_tensor_rt_inference::TensorRtInference`).
 */
class LSTMTensorRTInferenceOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(LSTMTensorRTInferenceOp, holoscan::ops::GXFOperator)

  LSTMTensorRTInferenceOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::lstm_tensor_rt_inference::TensorRtInference";
  }

  // TODO(gbae): use std::expected
  void setup(OperatorSpec& spec) override;

 private:
  Parameter<std::string> model_file_path_;
  Parameter<std::string> engine_cache_dir_;
  Parameter<std::string> plugins_lib_namespace_;
  Parameter<bool> force_engine_update_;
  Parameter<std::vector<std::string>> input_tensor_names_;
  Parameter<std::vector<std::string>> input_state_tensor_names_;
  Parameter<std::vector<std::string>> input_binding_names_;
  Parameter<std::vector<std::string>> output_tensor_names_;
  Parameter<std::vector<std::string>> output_state_tensor_names_;
  Parameter<std::vector<std::string>> output_binding_names_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
  Parameter<int64_t> max_workspace_size_;
  Parameter<int32_t> dla_core_;
  Parameter<int32_t> max_batch_size_;
  Parameter<bool> enable_fp16_;
  Parameter<bool> relaxed_dimension_check_;
  Parameter<bool> verbose_;

  Parameter<std::vector<IOSpec*>> rx_;
  Parameter<IOSpec*> tx_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_LSTM_TENSOR_RT_INFERENCE_HPP */
