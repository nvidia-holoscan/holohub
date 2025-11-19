/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <queue>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "holoscan/holoscan.hpp"

#include "ucxx_endpoint.h"

namespace isaac::os::ops {

// Sends messages through a UcxxEndpoint.
class UcxxSenderOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UcxxSenderOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

 private:
  void on_request_complete(ucs_status_t status);

  holoscan::Parameter<uint64_t> tag_;
  holoscan::Parameter<std::shared_ptr<UcxxEndpoint>> endpoint_;

  struct SendRequest {
    std::shared_ptr<ucxx::Request> request;
    flatbuffers::FlatBufferBuilder flatbuffer_builder;
  };
  std::list<SendRequest> requests_;
};

}  // namespace isaac::os::ops
