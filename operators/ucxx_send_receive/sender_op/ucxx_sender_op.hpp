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

#include <list>

#include "holoscan/holoscan.hpp"

#include "operators/ucxx_send_receive/serialize_tensor.hpp"
#include "operators/ucxx_send_receive/ucxx_endpoint.hpp"

namespace holoscan::ops {

// Sends messages through a UcxxEndpoint.
class UcxxSenderOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UcxxSenderOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void stop() override;
  void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<uint64_t> tag_;
  holoscan::Parameter<std::shared_ptr<UcxxEndpoint>> endpoint_;
  holoscan::Parameter<bool> blocking_;
  // Cap number of in-flight requests to bound memory retention when the network/receiver stalls.
  holoscan::Parameter<uint64_t> max_in_flight_;

  struct SendRequest {
    std::shared_ptr<::ucxx::Request> header_request;  // For header
    std::shared_ptr<::ucxx::Request> data_request;    // For tensor data
    holoscan::ops::ucxx::TensorHeader header;       // Header storage (must outlive header_request)
    bool cancel_requested = false;

    // Keepalive handle to ensure any buffers passed to UCX remain valid until both requests
    // are completed.
    holoscan::gxf::Entity keepalive_entity;
  };
  std::list<SendRequest> requests_;
};

}  // namespace holoscan::ops
