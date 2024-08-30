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

/*
 * ┌───────────────────────────────────────────────────────────────────────┐
 * │ gRPC Client Fragment (local)                                          │
 * │                                                                       │
 * │ ┌────────────────────┐   ┌─────────────────────┐   ┌────────────────┐ │
 * │ │                    │   │                     │   │                │ │
 * │ │ SayMyName Operator ┼───► gRPC Client Operator┼───► Print Operator │ │
 * │ │                    │   │                     │   │                │ │
 * │ └────────────────────┘   └─────────────────────┘   └────────────────┘ │
 * │                                                                       │
 * └───────────────────────────────────────────────────────────────────────┘
 */

#ifndef CLIENT_FRAGMENT_CC
#define CLIENT_FRAGMENT_CC

#include <fmt/format.h>
#include <holoscan/holoscan.hpp>
#include "entity_client.cc"

using namespace holoscan;

namespace holoscan {
namespace grpc_hello_world {

class PrintOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PrintOperator)

  PrintOperator() = default;
  void setup(OperatorSpec& spec) override { spec.input<std::string>("greeting"); }
  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    auto reply = op_input.receive<std::string>("greeting");
    if (reply) { HOLOSCAN_LOG_INFO("Greeter received: {}", reply.value()); }
  }
};

class SayMyNameOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SayMyNameOperator)

  SayMyNameOperator() = default;

  void start() override {}

  void setup(OperatorSpec& spec) override { spec.output<std::string>("name"); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::string user("world");
    op_output.emit(user, "name");
  }
};

template <typename GrpClientT>
class GrpcClientOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientOperator);

  GrpcClientOperator() = default;

  void start() override {
    HOLOSCAN_LOG_INFO("Starting gRPC client...");
    grpc_client_ = std::make_shared<GrpClientT>(
        GrpClientT(grpc::CreateChannel(server_address_.get(), grpc::InsecureChannelCredentials())));
  }

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(
        server_address_, "server_address", "gRPC server address", "gRPC server address");

    spec.input<std::string>("name");
    spec.output<std::string>("greeting");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto name = op_input.receive<std::string>("name");

    if (name) {
      std::string reply = grpc_client_->HelloWorld(name.value());
      op_output.emit(reply, "greeting");
    }
  }

 private:
  Parameter<std::string> server_address_;
  std::shared_ptr<GrpClientT> grpc_client_;
};

//

class GrpcClientFragment : public holoscan::Fragment {
 public:
  GrpcClientFragment(const std::string& server_address) : server_address_(server_address) {}
  void compose() override {
    auto say_my_name =
        make_operator<SayMyNameOperator>("say_my_name", make_condition<CountCondition>(1));
    auto grpc_client = make_operator<GrpcClientOperator<EntityClient>>(
        "grpc_client", Arg("server_address", server_address_));
    auto print_name = make_operator<PrintOperator>("print_name");

    add_flow(say_my_name, grpc_client);
    add_flow(grpc_client, print_name);
  }

 private:
  std::string server_address_;
};
}  // namespace grpc_hello_world
}  // namespace holoscan
#endif /* CLIENT_FRAGMENT_CC */
