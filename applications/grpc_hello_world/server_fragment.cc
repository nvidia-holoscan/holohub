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

#ifndef GRPC_SERVER_FRAGMENT
#define GRPC_SERVER_FRAGMENT

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <holoscan/holoscan.hpp>

#include "helloworld.grpc.pb.h"
#include "helloworld.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

using namespace holoscan;

namespace holoscan {
namespace grpc_hello_world {

class SayHelloOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SayHelloOperator)

  SayHelloOperator() = default;

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(
        server_address_, "server_address", "gRPC server address", "gRPC server address");
    spec.input<std::string>("in").condition(ConditionType::kNone);
    spec.output<std::string>("out").condition(ConditionType::kNone);
  }

  void start() override {
    HOLOSCAN_LOG_INFO("Starting gRPC server...");
    server_thread_ = std::thread(&SayHelloOperator::StartInternal, this);
    // SayHelloOperator::StartInternal();
  }

  void stop() override {
    HOLOSCAN_LOG_INFO("Stopping gRPC server...");
    server_->Shutdown();
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {}

 private:
  Parameter<std::string> server_address_;
  std::thread server_thread_;
  std::unique_ptr<Server> server_;

  void StartInternal() {
    // pthread_create(&server_thread_, NULL, )
    GreeterServiceImpl service;
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address_.get(), grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    HOLOSCAN_LOG_INFO("Server listening on {}", server_address_.get());
    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
  }
};

class GrpcServerFragment : public holoscan::Fragment {
 public:
  GrpcServerFragment(const std::string& server_address) : server_address_(server_address) {}
  void compose() override {
    auto say_hello =
        make_operator<SayHelloOperator>("say_hello", make_condition<CountCondition>(1), Arg("server_address", server_address_));
    add_operator(say_hello);
  }

 private:
  std::string server_address_;
};
}  // namespace grpc_hello_world
}  // namespace holoscan
#endif