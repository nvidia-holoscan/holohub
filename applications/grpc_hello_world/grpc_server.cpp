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
 * ┌────────────────────────────────┐
 * │ gRPC Server (local)            │
 * │                                │
 * │   ┌─────────────────────┐      │
 * │   │                     │      │
 * │   │ GrpcServer Operator │      │
 * │   │                     │      │
 * │   └─────────────────────┘      │
 * │                                │
 * └────────────────────────────────┘
 */

#ifndef GRPC_SERVER_CPP
#define GRPC_SERVER_CPP

#include <fmt/format.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <holoscan/holoscan.hpp>

#include "entity_server.cc"

using grpc::ServerBuilder;

using namespace holoscan;

namespace holoscan {
namespace grpc_hello_world {

template <typename GrpcServiceImplT>
class GrpcServerOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerOperator)

  GrpcServerOperator() = default;

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(
        server_address_, "server_address", "gRPC server address", "gRPC server address");
    spec.input<std::string>("in").condition(ConditionType::kNone);
    spec.output<std::string>("out").condition(ConditionType::kNone);
  }

  void start() override {
    HOLOSCAN_LOG_INFO("Starting gRPC server...");
    server_thread_ = std::thread(&GrpcServerOperator::StartInternal, this);
  }

  void stop() override {
    HOLOSCAN_LOG_INFO("Stopping gRPC server...");
    server_->Shutdown();
    if (server_thread_.joinable()) { server_thread_.join(); }
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::unique_lock<std::mutex> lock(server_callback_mutex_);
    HOLOSCAN_LOG_INFO("compute(): waiting for connection...");
    server_callback_cv_.wait(lock);
    HOLOSCAN_LOG_INFO("compute(): completed.");
  }

 private:
  Parameter<std::string> server_address_;
  std::thread server_thread_;
  std::unique_ptr<Server> server_;
  std::condition_variable server_callback_cv_;
  std::mutex server_callback_mutex_;

  void StartInternal() {
    GrpcServiceImplT service([this] {
      std::lock_guard<std::mutex> lock{this->server_callback_mutex_};
      this->server_callback_cv_.notify_all();
      HOLOSCAN_LOG_INFO("Callback triggered.");
    });
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address_.get(), grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    HOLOSCAN_LOG_INFO("Server listening on {}", server_address_.get());
    server->Wait();
  }
};

class ServerApp : public holoscan::Application {
 public:
  explicit (const std::string& server_address) : server_address_(server_address) {}
  void compose() override {
    auto say_hello = make_operator<GrpcServerOperator<HoloscanEntityServiceImpl>>(
        "say_hello", Arg("server_address", server_address_));
    add_operator(say_hello);
  }

 private:
  std::string server_address_;
};
}  // namespace grpc_hello_world
}  // namespace holoscan

int main(int argc, char** argv) {
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/grpc_hello_world.yaml";
  auto app = holoscan::make_application<holoscan::grpc_hello_world::ServerApp>("0.0.0.0:50051");
  app->config(config_path);
  app->run();
  return 0;
}
#endif /* GRPC_SERVER_CPP */
