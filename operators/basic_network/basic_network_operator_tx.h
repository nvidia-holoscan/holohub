/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. * SPDX-License-Identifier: Apache-2.0
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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <string>
#include "basic_network_operator_common.h"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class BasicNetworkOpTx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicNetworkOpTx);

  BasicNetworkOpTx() = default;
  ~BasicNetworkOpTx() = default;
  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<std::string> ip_addr_;
  Parameter<uint16_t> port_;
  Parameter<int32_t> retry_connect_;
  Parameter<std::string> l4_proto_p_;
  Parameter<uint16_t> max_payload_size_;
  Parameter<uint32_t> ipg_;
  Parameter<bool> delete_payload_;

  int sockfd_;
  L4Proto l4_proto_;
  struct sockaddr_in server_addr_;
  uint32_t byte_cnt_ = 0;
  uint32_t pkts_sent_ = 0;
  struct timespec ts_;
  bool connected_ = false;
};

};  // namespace holoscan::ops
