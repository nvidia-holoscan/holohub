/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_OPENIGTLINK_TX_HPP
#define HOLOSCAN_OPERATORS_OPENIGTLINK_TX_HPP

#include <holoscan/holoscan.hpp>

#include "igtlClientSocket.h"
#include "igtlImageMessage.h"
#include "igtlMath.h"
#include "igtlTimeStamp.h"

namespace holoscan::ops {

class OpenIGTLinkTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OpenIGTLinkTxOp)

  void start() override;
  void stop() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::vector<holoscan::IOSpec*>> receivers_;
  Parameter<std::vector<std::string>> input_names_;
  Parameter<std::string> device_name_;
  Parameter<std::string> host_name_;
  Parameter<int> port_;
  igtl::ClientSocket::Pointer client_socket_;
  std::map<std::string, std::string> input_;
  igtl::TimeStamp::Pointer time_stamp_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_OPENIGTLINK_TX_HPP */
