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

#include "application_factory.hpp"

namespace holoscan::ops {

ApplicationFactory::ApplicationFactory() {
  HOLOSCAN_LOG_DEBUG("ApplicationFactory created");
}

ApplicationFactory::~ApplicationFactory() {
  HOLOSCAN_LOG_DEBUG("ApplicationFactory destroyed");
}

std::shared_ptr<ApplicationFactory> ApplicationFactory::get_instance() {
  // May be less efficient to new the instance but since this is a Singleton and thread-safe.
  static std::shared_ptr<ApplicationFactory> instance(new ApplicationFactory(),
                                                      ApplicationFactoryDeleter());
  return instance;
}

void ApplicationFactory::register_application(const std::string& service_name,
                                              create_application_instance_func func) {
  if (application_registry_.find(service_name) != application_registry_.end()) {
    HOLOSCAN_LOG_WARN("Overwriting existing application registry: {}", service_name);
  }
  application_registry_[service_name] = func;
}

std::shared_ptr<HoloscanGrpcApplication> ApplicationFactory::create_application_instance(
    const std::string& service_name,
    std::queue<std::shared_ptr<nvidia::gxf::Entity>>& incoming_request_queue,
    std::queue<std::shared_ptr<EntityResponse>>& outgoing_response_queue) {
  if (application_registry_.find(service_name) == application_registry_.end()) {
    HOLOSCAN_LOG_ERROR("Application not found in registry: {}", service_name);
    return nullptr;
  }

  if (application_instances_.find(service_name) != application_instances_.end()) {
    HOLOSCAN_LOG_WARN("Another application instance is running: {}", service_name);
    return nullptr;
  }

  HOLOSCAN_LOG_INFO("Creating application instance for {}", service_name);

  create_application_instance_func func = application_registry_[service_name];
  application_instances_[service_name] = func(incoming_request_queue, outgoing_response_queue);
  application_instances_[service_name].instance->start_streaming();
  return application_instances_[service_name].instance;
}

void ApplicationFactory::destroy_application_instance(
    std::shared_ptr<HoloscanGrpcApplication> application_instance) {
  for (auto& [service_name, instance] : application_instances_) {
    if (instance.instance == application_instance) {
      instance.instance->stop_streaming();
      instance.future.wait_for(std::chrono::seconds(1));
      if (instance.tracker != nullptr) { instance.tracker->print(); }
      application_instances_.erase(service_name);
      HOLOSCAN_LOG_INFO("Application instance deleted for {}", service_name);
      return;
    }
  }
}
}  // namespace holoscan::ops
