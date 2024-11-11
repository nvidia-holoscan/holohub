/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RMAX_APPS_LIB_SERVICES_RMAX_BASE_SERVICE_H_
#define RMAX_APPS_LIB_SERVICES_RMAX_BASE_SERVICE_H_

#include <string>
#include <thread>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::services;

namespace ral {
namespace services {

struct RmaxBaseServiceConfig {
  std::shared_ptr<AppSettings> app_settings;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib;
};

/**
 * @class IRmaxServicesSynchronizer
 * @brief Interface for synchronizing Rmax services.
 *
 * This interface provides a method for synchronizing the start of multiple Rmax services.
 * Implementations of this interface should provide the necessary synchronization mechanisms
 * to ensure that all services start together.
 */
class IRmaxServicesSynchronizer {
 public:
  virtual ~IRmaxServicesSynchronizer() = default;
  /**
   * @brief Wait for all services to reach the start point and then proceed.
   *
   * This method should be implemented to provide the necessary synchronization
   * to ensure that all services reach the start point before any of them proceed.
   */
  virtual void wait_for_start() = 0;
};

constexpr int INVALID_CORE_NUMBER = -1;

/**
 * @brief: Base calls for Rivermax service.
 *
 * This is a base class offers common operations for Rivermax service.
 * The user of this interface must implement it's pure virtual methods and
 * can override it's virtual methods.
 */
class RmaxBaseService {
 protected:
  /* Indicator on whether the object created correctly */
  ReturnStatus m_obj_init_status;
  /* Service settings pointer */
  std::shared_ptr<AppSettings> m_service_settings;
  /* Rmax apps lib facade */
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> m_rmax_apps_lib;
  /* Local NIC address */
  sockaddr_in m_local_address;
  /* Header memory allocator */
  std::shared_ptr<MemoryAllocator> m_header_allocator;
  /* Payload memory allocator */
  std::shared_ptr<MemoryAllocator> m_payload_allocator;
  /* Service signal handler */
  std::shared_ptr<SignalHandler> m_signal_handler;
  /* Thread objects container */
  std::vector<std::thread> m_threads;
  /* Statistics reader */
  std::unique_ptr<StatisticsReader> m_stats_reader;

  std::string m_service_description;

 public:
  /**
   * @brief: RmaxBaseService constructor.
   *
   * @param [in] service_description: Service description string.
   */
  explicit RmaxBaseService(const std::string& service_description);
  virtual ~RmaxBaseService();
  /**
   * @brief: Runs the service.
   *
   * This is the main entry point to the service.
   * The initialization flow, using @ref ral::apps::RmaxBaseApp::initialize call, should be run
   * before calling to this method.
   *
   * @return: Status of the operation.
   */
  virtual ReturnStatus run(IRmaxServicesSynchronizer* sync_obj = nullptr) = 0;

  virtual ReturnStatus get_init_status() const { return m_obj_init_status; }

 protected:
  /**
   * @brief: Initializes service common default settings.
   *
   * The user of this interface can override function, in order to override
   * service specific default setting.
   */
  virtual void initialize_common_default_service_settings();
  /**
   * @brief: Parse Configuration.
   *
   * It will be called as part of the @ref ral::apps::RmaxBaseService::initialize process.
   */
  virtual ReturnStatus parse_configuration(const RmaxBaseServiceConfig& cfg) {
    return ReturnStatus::success;
  }
  /**
   * @brief: Does post config parsing initialization.
   *
   * Use this method to do any needed post config parsing service initialization.
   * It will be called as part of the @ref ral::apps::RmaxBaseService::initialize process.
   */
  virtual void post_parse_config_initialization() {}
  /**
   * @brief: Initializes memory allocators.
   *
   * @return: Status of the operation.
   */
  virtual ReturnStatus initialize_memory_allocators();
  /**
   * @brief: Runs service initialization flow.
   *
   * Use this method to run service initialization flow using the other methods in this
   * interface.
   * It will eventually initialize Rivermax library.
   *
   * The user can override the proposed initialization flow.
   *
   * @param [in] cfg: Service configuration
   *
   * @return: Status of the operation, on success: @ref
   * ral::lib:services::ReturnStatus::obj_init_success.
   */
  virtual ReturnStatus initialize(const RmaxBaseServiceConfig& cfg);
  /**
   * @brief: Initializes Rivermax library resources.
   *
   * Use this method to initialize Rivermax library configuration. It should
   * use @ref ral::lib::RmaxAppsLibFacade::initialize_rivermax method do so.
   *
   * @return: Status of the operation.
   */
  virtual ReturnStatus initialize_rivermax_resources() = 0;
  /**
   * @brief: Cleans up Rivermax library resources.
   *
   * Use this method to clean up any resources associated with Rivermax library.
   * It will be called implicitly on class destruction.
   *
   * There is no need to call @ref rmax_cleanup, it will be done implicitly.
   *
   * @return: Status of the operation.
   */
  virtual ReturnStatus cleanup_rivermax_resources() { return ReturnStatus::success; }
  /**
   * @brief: Sets Rivermax clock.
   *
   * Use this method to set Rivermax clock.
   *
   * @return: Status of the operation.
   */
  virtual ReturnStatus set_rivermax_clock();
  /**
   * @brief: Initializes the local NIC address.
   *
   * @return: Status of the operation.
   */
  virtual ReturnStatus initialize_connection_parameters();
  /**
   * @brief: Runs service threads.
   *
   * This method will run service IONodes as the context for @ref std::thread.
   * The IONode should override the operator () as it's worker method.
   *
   * @param [in] io_nodes: A container of IONodes, it should follow STL containers interface.
   */
  template <typename T>
  void run_threads(T& io_nodes);
  /**
   * @brief: Runs statistics reader thread.
   */
  void run_stats_reader();

  /**
   * @brief: Check if need not run the statistics reader.
   *
   * @return: true if need to run the statistics reader.
   */
  bool is_run_stats_reader() {
    return (m_service_settings->statistics_reader_core != INVALID_CORE_NUMBER);
  }
};

template <typename T>
void RmaxBaseService::run_threads(T& io_nodes) {
  for (auto& io_node : io_nodes) { m_threads.push_back(std::thread(std::ref(*io_node))); }

  for (auto& thread : m_threads) { thread.join(); }
}

}  // namespace services
}  // namespace ral

#endif /* RMAX_APPS_LIB_SERVICES_RMAX_SERVICE_BASE_APP_H_ */
