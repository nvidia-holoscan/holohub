/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "ucxx/api.h"

#include "ucxx_endpoint.hpp"

namespace holoscan::ops {

// Receives messages through a UcxxEndpoint.
class UcxxReceiverOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UcxxReceiverOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<uint64_t> tag_;
  holoscan::Parameter<int> buffer_size_;
  holoscan::Parameter<std::shared_ptr<UcxxEndpoint>> endpoint_;
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

  std::vector<uint8_t> buffer_;
  std::shared_ptr<ucxx::Request> request_;
};

}  // namespace holoscan::ops
