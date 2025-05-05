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

#ifndef CLIENT_ENTITY_CLIENT_HPP
#define CLIENT_ENTITY_CLIENT_HPP

#include <chrono>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <gxf/std/tensor.hpp>

#include "../common/asynchronous_condition_queue.hpp"
#include "../common/conditional_variable_queue.hpp"
#include "../common/tensor_proto.hpp"
#include "holoscan.grpc.pb.h"
#include "holoscan.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

using namespace holoscan::ops;

/**
 * @typedef on_new_response_available_callback
 * @brief Callback type for handling new responses from the server.
 *
 * A callback function that is invoked when a new response is received from the server.
 */
using on_new_response_available_callback =
    std::function<std::shared_ptr<nvidia::gxf::Entity>(EntityResponse& response)>;

/**
 * @typedef on_rpc_completed_callback
 * @brief Callback type for handling RPC completion events.
 *
 * This callback is invoked when an RPC operation is completed.
 */
using on_rpc_completed_callback = std::function<void()>;

/**
 * @class EntityClient
 * @brief A client class for handling gRPC communication with an entity server.
 *
 * This class is responsible for creating a gRPC client that connects to the server and sends
 * requests to the server.
 * The EntityStream(...) method implement the gRPC Callback API for bi-directional streaming using
 * the EntityStreamInternal class which is responsible for reading and writing streams.
 * Given that Holoscan Operators do not have a way to notify a downstream operator for the
 * completion of itself, a `rpc_timeout` parameter is used to close the connection if no data is
 * transmitted or received for the specified time.
 *
 * @note On a slow network, adjust the `rpc_timeout` value accordingly.
 */

class EntityClient {
 public:
  EntityClient(
      const std::string& server_address, const uint32_t rpc_timeout,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
      std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>
          response_queue);

  void EntityStream(on_new_response_available_callback response_cb,
                    on_rpc_completed_callback rpc_completed_cb);

 private:
  class EntityStreamInternal : public grpc::ClientBidiReactor<EntityRequest, EntityResponse> {
   public:
    EntityStreamInternal(EntityClient* client, on_new_response_available_callback response_cb,
                         on_rpc_completed_callback rpc_completed_cb, uint32_t rpc_timeout);

    ~EntityStreamInternal();

    void OnWriteDone(bool ok) override;
    void OnReadDone(bool ok) override;
    void OnDone(const grpc::Status& status) override;
    Status Await();

   private:
    void Read();
    void Write();
    void ProcessOutgoingQueue();
    bool network_timed_out();

    EntityClient* client_;
    EntityResponse response_;
    ClientContext context_;
    on_new_response_available_callback response_cb_;
    on_rpc_completed_callback rpc_completed_cb_;

    bool done_;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    Status status_;

    std::mutex write_mutex_;
    std::thread writer_thread_;
    int rpc_timeout_;
    std::chrono::time_point<std::chrono::system_clock> last_network_activity_;
  };

  uint32_t rpc_timeout_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
  std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>> response_queue_;

  std::shared_ptr<Channel> channel_;
  std::unique_ptr<Entity::Stub> stub_;
  EntityStreamInternal* reactor_;
};
}  // namespace holoscan::ops

#endif /* CLIENT_ENTITY_CLIENT_HPP */
