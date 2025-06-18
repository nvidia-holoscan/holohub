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

#ifndef HOLOSCAN_OPERATORS_APPEND_TIMESTAMP_HPP
#define HOLOSCAN_OPERATORS_APPEND_TIMESTAMP_HPP

#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to append timestamp to a tensor.
 *
 * This operator takes a tensor as input and appends a timestamp to it,
 * then outputs the tensor with the timestamp.
 */
class AppendTimestampOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AppendTimestampOp)

  AppendTimestampOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_APPEND_TIMESTAMP_HPP
