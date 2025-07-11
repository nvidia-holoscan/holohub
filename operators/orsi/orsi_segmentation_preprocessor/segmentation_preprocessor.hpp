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

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"
#include "segmentation_preprocessor.cuh"

namespace holoscan::ops::orsi {

class SegmentationPreprocessorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SegmentationPreprocessorOp)

  SegmentationPreprocessorOp() = default;

  // TODO(gbae): use std::expected
  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  segmentation_preprocessor::DataFormat data_format_value_;

  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<std::shared_ptr<Allocator>> allocator_;

  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> out_tensor_name_;
  Parameter<std::string> data_format_;

  Parameter<std::vector<float>> normalize_means_;
  Parameter<std::vector<float>> normalize_stds_;

  std::vector<float> means_host_;
  std::vector<float> stds_host_;

  float* means_cuda_ = nullptr;
  float* stds_cuda_ = nullptr;

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops::orsi
