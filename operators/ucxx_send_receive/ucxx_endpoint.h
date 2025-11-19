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

#include <string>

#include "holoscan/holoscan.hpp"
#include "ucxx/api.h"

namespace isaac::os {

// Manages a UCXX Endpoint that can be used to send and receive messages with UCX.
class UcxxEndpoint : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(UcxxEndpoint)

  UcxxEndpoint() = default;
  ~UcxxEndpoint();

  void setup(holoscan::ComponentSpec& spec) override;
  void initialize() override;

  std::shared_ptr<ucxx::Endpoint> endpoint() const { return endpoint_; }
  std::shared_ptr<ucxx::Worker> worker() const { return worker_; }

  std::shared_ptr<holoscan::Condition> is_alive_condition() const { return is_alive_condition_; }

 private:
  void on_connection_request(ucp_conn_request_h conn_request);
  void on_endpoint_closed(ucs_status_t status);

  holoscan::Parameter<std::string> hostname_;
  holoscan::Parameter<int> port_;
  holoscan::Parameter<bool> listen_;

  std::shared_ptr<ucxx::Context> context_{nullptr};
  std::shared_ptr<ucxx::Worker> worker_{nullptr};
  std::shared_ptr<ucxx::Listener> listener_{nullptr};
  std::shared_ptr<ucxx::Endpoint> endpoint_{nullptr};

  std::shared_ptr<holoscan::AsynchronousCondition> is_alive_condition_;
};

}  // namespace isaac::os
