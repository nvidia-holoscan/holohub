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

#ifndef ENTITY_SERVER_CC
#define ENTITY_SERVER_CC

#include <fmt/format.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include "holoscan.grpc.pb.h"
#include "holoscan.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan {
namespace grpc_hello_world {

class HoloscanEntityServiceImpl final : public Entity::Service {
  Status Metadata(ServerContext* context, const EntityRequest* request,
                  EntityResponse* reply) override {
    auto service = request->service();

    if (service == "hello_world") {
      auto params = request->parameters();
      auto response = reply->mutable_parameters();
      (*response)["greeting"] = fmt::format("Hello {}!", params["name"]);
      return Status::OK;
    }

    return grpc::Status(grpc::StatusCode::NOT_FOUND, fmt::format("Service {} not found", service));
  }
};
}  // namespace grpc_hello_world
}  // namespace holoscan

#endif