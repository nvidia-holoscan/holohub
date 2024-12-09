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

#ifndef SERVER_APPLICATION_FACTORY_HPP
#define SERVER_APPLICATION_FACTORY_HPP

#include <functional>
#include <memory>
#include <queue>

#include <gxf/core/entity.hpp>
#include <holoscan/holoscan.hpp>

#include "holoscan.pb.h"
#include "grpc_application.hpp"

using holoscan::entity::EntityResponse;

namespace holoscan::ops {

class HoloscanGrpcApplication;

/**
 * @struct ApplicationInstance
 * @brief Represents an instance of an Holoscan application.
 *
 * This structure holds a shared pointer to a HoloscanGrpcApplication instance
 * and a future object for asynchronous operations.
 *
 * @var ApplicationInstance::instance
 * A shared pointer to the HoloscanGrpcApplication instance.
 *
 * @var ApplicationInstance::future
 * A future object representing the result of an asynchronous operation.
 */
struct ApplicationInstance {
  std::shared_ptr<HoloscanGrpcApplication> instance;
  std::future<void> future;
  DataFlowTracker* tracker = nullptr;
};

/**
 * @typedef create_application_instance_func
 * @brief A function type for creating an instance of a Holoscan application.
 *
 * This function type takes a queue of incoming requests and a queue of outgoing responses
 * and returns an ApplicationInstance object.
 */
using create_application_instance_func = std::function<ApplicationInstance(
    std::queue<std::shared_ptr<nvidia::gxf::Entity>> incoming_request_queue,
    std::queue<std::shared_ptr<EntityResponse>> outgoing_response_queue)>;

/**
 * @class ApplicationFactory
 * @brief A factory class for creating and managing instances of HoloscanGrpcApplications.
 *
 * The ApplicationFactory class is responsible for registering, creating, and destroying
 * instances of HoloscanGrpcApplication. It follows the singleton pattern to ensure that
 * only one instance of the factory exists.
 *
 * Register each gRPC service with a Holoscan application with the Application Factory.
 * This decouples the application creation from the gRPC service and the application pipeline.
 *
 * @note Copy constructor and assignment operator are deleted to prevent copying.
 */

class ApplicationFactory {
 public:
  ApplicationFactory(const ApplicationFactory&) = delete;
  ApplicationFactory& operator=(const ApplicationFactory&) = delete;

  /**
   * @brief Get the singleton instance of ApplicationFactory.
   *
   * @return A shared pointer to the singleton instance of ApplicationFactory.
   */
  static std::shared_ptr<ApplicationFactory> get_instance();

  /**
   * @brief Register an application creation function with a service name.
   *
   * @param service_name The name of the service.
   * @param func The function to create an application instance.
   */
  void register_application(const std::string& service_name, create_application_instance_func func);

  /**
   * @brief Create an instance of HoloscanGrpcApplication.
   *
   * @param service_name The name of the service.
   * @param incoming_request_queue A queue for incoming requests.
   * @param outgoing_response_queue A queue for outgoing responses.
   * @return A shared pointer to the created HoloscanGrpcApplication instance.
   */
  std::shared_ptr<HoloscanGrpcApplication> create_application_instance(
      const std::string& service_name,
      std::queue<std::shared_ptr<nvidia::gxf::Entity>>& incoming_request_queue,
      std::queue<std::shared_ptr<EntityResponse>>& outgoing_response_queue);

  /**
   * @brief Destroy an instance of HoloscanGrpcApplication.
   *
   * @param application_instance A shared pointer to the HoloscanGrpcApplication instance to be
   * destroyed.
   */
  void destroy_application_instance(std::shared_ptr<HoloscanGrpcApplication> application_instance);

 private:
  /**
   * @class ApplicationFactoryDeleter
   * @brief A custom deleter for ApplicationFactory.
   *
   * This class defines a custom deleter for the ApplicationFactory singleton instance since
   * ApplicationFactory's destructor is private, it cannot be accessed by the shared pointer.
   */
  class ApplicationFactoryDeleter {
   public:
    void operator()(ApplicationFactory* factory) { delete factory; }
  };

  ApplicationFactory();
  ~ApplicationFactory();

  std::map<std::string, create_application_instance_func> application_registry_;
  std::map<std::string, ApplicationInstance> application_instances_;
};
}  // namespace holoscan::ops
#endif /* SERVER_APPLICATION_FACTORY_HPP */
