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

#ifndef SERVER_GRPC_APPLICATION_HPP
#define SERVER_GRPC_APPLICATION_HPP

#include <gxf/app/graph_entity.hpp>
#include <holoscan/holoscan.hpp>

#include "grpc_server_request.hpp"
#include "grpc_server_response.hpp"

#include "../common/tensor_proto.hpp"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

/**
 * @class HoloscanGrpcApplication
 * @brief A class that represents a gRPC application for Holoscan.
 *
 * A class extends the holoscan::Application and provides functionalities
 * to handle gRPC requests and responses using queues.
 *
 * This base class creates `incoming_request_queue` and `outgoing_response_queue` queues that can
 * be used by the derived classes to handle incoming requests and outgoing responses.
 *
 * The `streaming_enabled` BooleanCondition can be added to all operators to know when an RPC call
 * is active or not.
 *
 *
 */
class HoloscanGrpcApplication : public holoscan::Application {
 public:
  HoloscanGrpcApplication(std::queue<std::shared_ptr<nvidia::gxf::Entity>>& incoming_request_queue,
                          std::queue<std::shared_ptr<EntityResponse>>& outgoing_response_queue);

  /**
   * @brief Composes the gRPC application by setting up necessary components and configurations.
   *
   * The compose() method creates two operators:
   * - grpc_request_op: An operator that handles incoming requests.
   * - grpc_response_op: An operator that handles outgoing responses.
   * @note: Users should call this method in the derived class to set up the operators.
   */
  void compose() override;

  /**
   * @brief Sets the scheduler configuration for the Holoscan application.
   *
   * This function allows you to specify the scheduler configuration by providing
   * the name of the configuration.
   * @note: EventBasedScheduler is used when called, otherwise the default GreedyScheduler is used.
   *
   * @param config_name The name of the scheduler configuration to be set.
   */
  void set_scheduler(const std::string& config_name);

  /**
   * @brief Sets the data path.
   *
   * This function sets the path where the application will read or store data.
   *
   * @param path The path to the data directory.
   */
  void set_data_path(const std::string path);

  /**
   * @brief Enqueues an EntityRequest for processing.
   *
   * This function takes an EntityRequest object and adds it to the queue
   * for further processing. The queued items will be processed by the `GrpcServerRequestOp`
   * operator.
   *
   * @param request The entity request to be enqueued.
   */
  void enqueue_request(const EntityRequest& request);

  /**
   * @brief Enqueues an EntityResponse to be sent to the client.
   *
   * This function adds the provided response to the queue of responses
   * that will be sent to the client.
   *
   * @param response A shared pointer to the EntityResponse object that
   *                 needs to be enqueued.
   */
  void enqueue_response(std::shared_ptr<EntityResponse> response);

  /**
   * @brief Checks if a response is available.
   *
   * This function determines whether a response is currently available
   * for processing or retrieval.
   *
   * @return true if a response is available, false otherwise.
   */
  bool is_response_available();

  /**
   * @brief Dequeues a response from the internal queue.
   *
   * This function retrieves and removes the next available response from the
   * internal queue of responses. If the queue is empty, the function is blcocked until a response
   * is available.
   *
   * @return A shared pointer to an EntityResponse object, representing the
   *         dequeued response. If the queue is empty, the returned pointer
   *         may be null.
   */
  std::shared_ptr<EntityResponse> dequeue_response();

  /**
   * @brief Retrieves the RPC call timeout value.
   *
   * This function returns the timeout configured with the `GrpcServerRequestOp` operator.
   *
   * @return uint32_t The RPC call timeout value in milliseconds.
   */
  const uint32_t rpc_timeout();

  /**
   * @brief Starts the streaming process.
   *
   * This function sets the streaming_enabled BooleanCondition to true.
   * @note All operators should include the `streaming_enabled` BooleanCondition so the application
   * pipeline can be controlled by the gRPC service.
   */
  void start_streaming();

  /**
   * @brief Stops the streaming process.
   *
   * This function sets the streaming_enabled BooleanCondition to false.
   */
  void stop_streaming();

 protected:
  std::shared_ptr<GrpcServerRequestOp> grpc_request_op;
  std::shared_ptr<GrpcServerResponseOp> grpc_response_op;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> request_queue;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>> response_queue;
  std::shared_ptr<BooleanCondition> streaming_enabled;
  std::string data_path;
};

}  // namespace holoscan::ops

#endif /* SERVER_GRPC_APPLICATION_HPP */
