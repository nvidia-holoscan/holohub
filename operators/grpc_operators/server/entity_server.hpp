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

#ifndef SERVER_ENTITY_SERVER_HPP
#define SERVER_ENTITY_SERVER_HPP

// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>
#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "holoscan.grpc.pb.h"
#include "holoscan.pb.h"
#include "grpc_application.hpp"

using grpc::CallbackServerContext;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

/**
 * @fn bool processing_timed_out()
 * @brief Checks if the processing has timed out.
 * @return bool True if the processing has timed out, false otherwise.
 */
namespace holoscan::ops {

/**
 * @typedef on_new_entity_stream_rpc
 * @brief Callback function type to create a new instance of a Holoscan application.
 * @param std::string& The name of the application.
 * @param std::queue<std::shared_ptr<nvidia::gxf::Entity>>& Output messages.
 * @param std::queue<std::shared_ptr<EntityResponse>>& Incoming messages.
 * @return std::shared_ptr<HoloscanGrpcApplication> Shared pointer to the created Holoscan
 * application.
 */
using on_new_entity_stream_rpc = std::function<std::shared_ptr<HoloscanGrpcApplication>(
    const std::string&, std::queue<std::shared_ptr<nvidia::gxf::Entity>>&,
    std::queue<std::shared_ptr<EntityResponse>>&)>;

/**
 * @typedef on_entity_stream_rpc_complete
 * @brief Callback function type to handle the completion of an entity stream RPC.
 * @param std::shared_ptr<HoloscanGrpcApplication> Shared pointer to the completed Holoscan
 * application.
 */
using on_entity_stream_rpc_complete = std::function<void(std::shared_ptr<HoloscanGrpcApplication>)>;

/**
 * @class HoloscanEntityServiceImpl
 * @brief Implementation of the gRPC Entity Service for Holoscan.
 *
 * @param new_entity_stream_rpc Callback function to create a new instance of a Holoscan
 * application.
 * @param entity_stream_rpc_complete Callback function to handle the completion of an entity stream
 * RPC.
 */

class HoloscanEntityServiceImpl final : public Entity::CallbackService {
 public:
  HoloscanEntityServiceImpl(on_new_entity_stream_rpc new_entity_stream_rpc,
                            on_entity_stream_rpc_complete entity_stream_rpc_complete);

  /**
   * @fn grpc::ServerBidiReactor<EntityRequest, EntityResponse>* EntityStream(CallbackServerContext*
   * context)
   * @brief Handles the bidirectional streaming of entity requests and responses.
   * @param context Pointer to the callback server context.
   * @return grpc::ServerBidiReactor<EntityRequest, EntityResponse>* Pointer to the server
   * bidirectional reactor.
   */
  grpc::ServerBidiReactor<EntityRequest, EntityResponse>* EntityStream(
      CallbackServerContext* context) override;

 private:
  class EntityStreamInternal : public grpc::ServerBidiReactor<EntityRequest, EntityResponse> {
   public:
    EntityStreamInternal(HoloscanEntityServiceImpl* server,
                         std::shared_ptr<HoloscanGrpcApplication> app,
                         on_entity_stream_rpc_complete entity_stream_rpc_complete);

    ~EntityStreamInternal();

    void OnWriteDone(bool ok) override;
    void OnReadDone(bool ok) override;
    void OnDone() override;

   private:
    void Read();
    void Write();
    void processOutgoingQueue();
    bool processing_timed_out();

    std::shared_ptr<HoloscanGrpcApplication> app_;
    on_entity_stream_rpc_complete entity_stream_rpc_complete_;
    HoloscanEntityServiceImpl* server_;
    EntityRequest request_;
    std::chrono::time_point<std::chrono::system_clock> last_network_activity_;
    bool is_read_done_ = false;

    // std::mutex read_mutex_;
    std::mutex write_mutex_;
    std::thread writer_thread_;
  };

  on_new_entity_stream_rpc new_entity_stream_rpc_;
  on_entity_stream_rpc_complete entity_stream_rpc_complete_;
};

}  // namespace holoscan::ops

#endif /* SERVER_ENTITY_SERVER_HPP */
