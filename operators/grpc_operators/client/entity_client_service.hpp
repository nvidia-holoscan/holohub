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

#ifndef CLIENT_ENTITY_CLIENT_SERVICE_HPP
#define CLIENT_ENTITY_CLIENT_SERVICE_HPP

#include <memory>
#include <string>
#include <thread>

#include <gxf/core/gxf.h>
#include <holoscan/holoscan.hpp>

#include "holoscan.pb.h"

#include "entity_client.hpp"
#include "grpc_client_request.hpp"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

/**
 * @class EntityClientService
 * @brief A service class for handling gRPC client operations.
 *
 * This class manages the lifecycle and operations of a gRPC client that communicates
 * with a server to handle entity requests and responses.
 */
class EntityClientService {
 public:
  EntityClientService(
      const std::string& server_address, const uint32_t rpc_timeout, const bool interrupt,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
      std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>
          response_queue,
      std::shared_ptr<GrpcClientRequestOp> grpc_request_operator);

  /**
   * @brief Starts the entity stream RPC operation.
   *
   * This function initiates the streaming of entities to/from the server.
   * It establishes a connection and begins sending and receiving data.
   *
   * @note currently, this function starts a new thread with start_entity_stream_internal() which
   * calls the EntityClient::EntityStream(...) method to establish a bi-directional streaming
   * connection with the server. When that function returns, the entity client is reset and the
   * Holoscan application's interrupt method is called to exit the application.
   */
  void start_entity_stream();

  /**
   * @brief Stops the entity stream RPC operation.
   *
   * This function terminates the ongoing entity stream, ensuring that any
   * resources associated with the stream are properly released.
   */
  void stop_entity_stream();

 private:
  /**
   * @brief Starts a new thread for the EntityStream RPC.
   *
   * This method is launched in a separate thread to establish a bi-directional streaming connection
   * with the server. The first parameter is a callback function that is called when a new response
   * is received from the server. It converts an EntityResponse (protobuf message) to a GXF Entity.
   * The second parameter is a callback function that is called when the RPC operation is completed.
   */
  void start_entity_stream_internal();

  const std::string server_address_;
  const uint32_t rpc_timeout_;
  const bool interrupt_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
  std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>> response_queue_;
  std::shared_ptr<GrpcClientRequestOp> grpc_request_operator_;

  std::shared_ptr<EntityClient> entity_client_;
  std::thread streaming_thread_;
};
}  // namespace holoscan::ops

#endif /* CLIENT_ENTITY_CLIENT_SERVICE_HPP */
