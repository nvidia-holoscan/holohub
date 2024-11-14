/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef GRPC_GRPC_ENDOSCOPY_TOOL_TRACKING_CPP_CLOUD_GRPC_SERVICE_HPP
#define GRPC_GRPC_ENDOSCOPY_TOOL_TRACKING_CPP_CLOUD_GRPC_SERVICE_HPP

#include <fmt/format.h>
#include <holoscan/holoscan.hpp>

#include <grpc/grpc.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <grpc_server.hpp>

using grpc::Server;
using grpc::ServerBuilder;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

using namespace holoscan::ops;

/**
 * @class GrpcService
 * @brief A singleton class that manages a gRPC server for Holoscan applications.
 *
 * The GrpcService class is responsible for setting up and managing a gRPC server
 * that listens for incoming requests and processes them based on user configured applications with
 * the ApplicationFactory. It ensures that only one instance of the server is running at any given
 * time.
 *
 * @note This class cannot be copied or assigned.
 */
class GrpcService {
 public:
  GrpcService(const GrpcService&) = delete;
  GrpcService& operator=(const GrpcService&) = delete;

  static GrpcService& get_instance(uint32_t port,
                                   std::shared_ptr<ApplicationFactory> application_factory) {
    static GrpcService instance(fmt::format("0.0.0.0:{}", port), application_factory);
    return instance;
  }

  void start() {
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    service_ = std::make_unique<HoloscanEntityServiceImpl>(
        // Callback function to create a new instance of a Holoscan application when a new RPC call
        // is received.
        [this](const std::string& service_name,
               std::queue<std::shared_ptr<nvidia::gxf::Entity>>& incoming_request_queue,
               std::queue<std::shared_ptr<EntityResponse>>& outgoing_response_queue) {
          return application_factory_->create_application_instance(
              service_name, incoming_request_queue, outgoing_response_queue);
        },
        // Callback function to handle the completion of an entity stream RPC.
        [this](std::shared_ptr<HoloscanGrpcApplication> application_instance) {
          application_factory_->destroy_application_instance(application_instance);
        });

    ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    HOLOSCAN_LOG_INFO("grpc: Server listening on {}", server_address_);
    server_->Wait();
  }

  void stop() {
    HOLOSCAN_LOG_INFO("grpc: Server shutting down");
    server_->Shutdown();
  }

 private:
  std::string server_address_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<HoloscanEntityServiceImpl> service_;
  std::shared_ptr<ApplicationFactory> application_factory_;

  GrpcService(const std::string server_address,
              std::shared_ptr<ApplicationFactory> application_factory)
      : server_address_(server_address), application_factory_(application_factory) {}

  ~GrpcService() = default;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_GRPC_ENDOSCOPY_TOOL_TRACKING_CPP_CLOUD_GRPC_SERVICE_HPP */
