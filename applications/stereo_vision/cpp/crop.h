/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_CROP
#define OPERATORS_CROP

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>


namespace holoscan::ops {

class CropOp : public Operator{
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CropOp);
  CropOp() = default;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;
 private:
  Parameter<int> x_;
  Parameter<int> y_;
  Parameter<int> width_;
  Parameter<int> height_;
};

}  // namespace holoscan::ops
#endif
