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

#ifndef ENTITY_CLIENT_CC
#define ENTITY_CLIENT_CC

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>
#include "holoscan.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ServerContext;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan {
namespace grpc_hello_world {

class EntityClient {
 public:
  EntityClient(std::shared_ptr<Channel> channel) : stub_(Entity::NewStub(channel)) {}

  const std::string HelloWorld(const std::string& name) {
    EntityRequest request;
    request.set_service("hello_world");
    auto parameters = request.mutable_parameters();
    parameters->insert({"name", name});

    EntityResponse reply;

    ClientContext context;

    // The actual RPC.
    Status status = stub_->Metadata(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      auto& response = *reply.mutable_parameters();
      return response["greeting"];
    } else {
      throw std::runtime_error(
          fmt::format("RPC Failed with code: {}: {}", status.error_code(), status.error_message()));
    }
  }

 private:
  std::unique_ptr<Entity::Stub> stub_;
};
}  // namespace grpc_hello_world
}  // namespace holoscan

#endif // ENTITY_CLIENT_CC