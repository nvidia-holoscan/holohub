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

#ifndef HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP
#define HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace holoscan::ops {

/**
 * @brief Post-processing operator for tool tracking inference with LSTMTensorRTInferenceOp.
 *
 * ==Named Inputs==
 *
 * - **in** : `nvidia::gxf::Entity` containing multiple `nvidia::gxf::Tensor`
 *   - Must contain input tensors named "probs", "scaled_coords" and "binary_masks" that
 *     correspond to the output of the LSTMTensorRTInfereceOp as used in the endoscopy
 *     tool tracking example applications.
 *
 * ==Named Outputs==
 *
 * - **out** : `nvidia::gxf::Tensor`
 *   - Binary mask and coordinates tensor, stored on the device (GPU).
 *
 * ==Parameters==
 *
 * - **device_allocator**: The holoscan::Allocator class (e.g. UnboundedAllocator or
 *   BlockMemoryPool) used for device memory allocation for the output tensors.
 * - **min_prob**: Minimum probability threshold used by the operator.
 *   Optional (default: 0.5).
 * - **overlay_img_colors**: A `vector<vector<float>>` where each inner vector is a set of three
 *   floats corresponding to normalized RGB values in range [0, 1.0].
 *   Optional (default: a 12-class qualitative color scheme).
 * - **cuda_stream_pool**: `holoscan::CudaStreamPool` instance to allocate CUDA streams.
 *   Optional (default: `nullptr`).
 */
class ToolTrackingPostprocessorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ToolTrackingPostprocessorOp)

  ToolTrackingPostprocessorOp() = default;

  void setup(OperatorSpec& spec) override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<float> min_prob_;
  Parameter<std::vector<std::vector<float>>> overlay_img_colors_;

  Parameter<std::shared_ptr<Allocator>> device_allocator_;

  CudaStreamHandler cuda_stream_handler_;

  uint32_t num_colors_ = 0;
  void *dev_colors_ = nullptr;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP */
