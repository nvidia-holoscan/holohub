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

#ifndef RMAX_APPS_LIB_SERVICES_IPO_RECEIVER_SERVICE_H_
#define RMAX_APPS_LIB_SERVICES_IPO_RECEIVER_SERVICE_H_

#include <string>
#include <vector>
#include <memory>
#include <climits>
#include <unordered_set>
#include <mutex>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"
#include "ipo_receiver_io_node.h"
#ifdef RMAX_APPS_LIB_FLAT_STRUCTURE
#include "ipo_chunk_consumer_base.h"
#else
#include "receivers/ipo_chunk_consumer_base.h"
#endif
#include "rmax_base_service.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::services;

namespace ral {
namespace services {
namespace rmax_ipo_receiver {
struct RmaxIPOReceiverConfig : RmaxBaseServiceConfig {
  uint32_t max_path_differential_us;
  bool is_extended_sequence_number;
  bool register_memory;
  size_t max_chunk_size;
  uint32_t rx_stats_period_report_ms;
};
/**
 * Service constants.
 */
constexpr const char* SERVICE_DESCRIPTION = "NVIDIA Rivermax IPO Receiver Service ";
/**
 * @brief: IPO Receiver service.
 *
 * This is an example of usage service for Rivermax Inline Packet Ordering RX API.
 */
class RmaxIPOReceiverService : public RmaxBaseService {
 private:
  static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 262144;
  static constexpr int USECS_IN_SECOND = 1000000;

  static bool m_rivermax_lib_initialized;
  static std::mutex m_rivermax_lib_mutex;

  /* Sender objects container */
  std::vector<std::unique_ptr<IPOReceiverIONode>> m_receivers;
  /* Stream per thread distribution */
  std::unordered_map<size_t, size_t> m_streams_per_thread;
  /* Network recv flows */
  std::vector<std::vector<FourTupleFlow>> m_flows;
  /* NIC device interfaces */
  std::vector<rmx_device_iface> m_device_ifaces;
  /* Memory regions for header memory allocated for each device interface */
  std::vector<rmx_mem_region> m_header_mem_regions;
  /* Memory regions for payload memory allocated for each device interface */
  std::vector<rmx_mem_region> m_payload_mem_regions;
  // Maximum Path Differential for "Class B: Moderate-Skew" receivers defined
  // by SMPTE ST 2022-7:2019 "Seamless Protection Switching of RTP Datagrams".
  uint64_t m_max_path_differential_us = 50000;
  bool m_is_extended_sequence_number = false;
  bool m_register_memory = false;
  byte_t* m_header_buffer = nullptr;
  byte_t* m_payload_buffer = nullptr;
  size_t m_num_paths_per_stream = 0;
  size_t m_max_chunk_size = 1024;
  bool m_service_running = false;
  uint32_t m_rx_stats_period_report_ms = 1000;
  IIPOChunkConsumer* m_chunk_consumer = nullptr;

 public:
  /**
   * @brief: RmaxIPOReceiverService class constructor.
   *
   * @param [in] cfg: service configuration
   */
  explicit RmaxIPOReceiverService(const RmaxIPOReceiverConfig& cfg);
  virtual ~RmaxIPOReceiverService() = default;
  ReturnStatus run(IRmaxServicesSynchronizer* sync_obj = nullptr) override;
  bool is_alive() { return m_service_running; }
  bool set_chunk_consumer(IIPOChunkConsumer* chunk_consumer);

  /**
   * @brief: Returns streams statistics.
   *
   * @return: Pair of vectors of statistics. First vector contains stream
   *          statistics, second vector contains path statistics.
   */
  std::pair<std::vector<IPORXStatistics>, std::vector<std::vector<IPOPathStatistics>>>
  get_streams_statistics() const;

 private:
  /**
   * @brief: Parse Configuration.
   *
   * It will be called as part of the @ref ral::apps::RmaxBaseService::initialize process.
   */
  ReturnStatus parse_configuration(const RmaxBaseServiceConfig& cfg) final;
  ReturnStatus initialize_connection_parameters() final;
  ReturnStatus initialize_rivermax_resources() final;
  /**
   * @brief: Initializes network receive flows.
   *
   * This method initializes the receive flows that will be used
   * in the service. These flows will be distributed
   * in @ref ral::services::RmaxIPOReceiverService::distribute_work_for_threads
   * between service threads.
   * The service supports unicast and multicast UDPv4 receive flows.
   */
  void configure_network_flows();
  /**
   * @brief: Distributes work for threads.
   *
   * This method is responsible for distributing work to threads, by
   * distributing number of streams per receiver thread uniformly.
   * In future development, this can be extended to different
   * streams per thread distribution policies.
   */
  void distribute_work_for_threads();
  /**
   * @brief: Initializes receiver I/O nodes.
   *
   * This method is responsible for initialization of
   * @ref ral::io_node::IPOReceiverIONode objects to work. It will initiate
   * objects with the relevant parameters. The objects initialized in this
   * method, will be the contexts to the std::thread objects will run in
   * @ref ral::services::RmaxBaseService::run_threads method.
   */
  void initialize_receive_io_nodes();
  /**
   * @brief: Allocates service memory and registers it if requested.
   *
   * This method is responsible for allocation of the required memory for
   * the service using @ref ral::lib::services::MemoryAllocator interface.
   * The allocation policy of the service is allocating one big memory
   * block. This memory block will be distributed to the different
   * components of the service.
   *
   * If @ref m_register_memory is set then this function also registers
   * allocated memory using @ref rmax_register_memory on all devices.
   *
   * @return: Returns status of the operation.
   */
  ReturnStatus allocate_service_memory();
  /**
   * @brief Unregister previously registered memory.
   *
   * Unregister memory using @ref rmax_deregister_memory.
   */
  void unregister_service_memory();
  /**
   * @brief: Distributes memory for receivers.
   *
   * This method is responsible for distributing the memory allocated
   * by @ref allocate_service_memory to the receivers of the service.
   */
  void distribute_memory_for_receivers();
  /**
   * @brief: Returns the memory size for all the receive streams.
   *
   * This method calculates the sum of memory sizes for all IONodes and their IPO streams.
   * Inside an IPO stream memory size is not summed along redundant streams,
   * they are only checked for equal requirements.
   *
   * @param [out] hdr_mem_size: Required header memory size.
   * @param [out] pld_mem_size: Required payload memory size.
   *
   * @return: Return status of the operation.
   */
  ReturnStatus get_total_ipo_streams_memory_size(size_t& hdr_mem_size, size_t& pld_mem_size);
  /**
   * @brief: Allocates memory and aligns it to page size.
   *
   * @param [in]  header_size:  Requested header memory size.
   * @param [in]  payload_size: Requested payload memory size.
   * @param [out] header:       Allocated header memory pointer.
   * @param [out] payload:      Allocated payload memory pointer.
   *
   * @return: True if successful.
   */
  bool allocate_and_align(size_t header_size, size_t payload_size, byte_t*& header,
                          byte_t*& payload);
};

}  // namespace rmax_ipo_receiver
}  // namespace services
}  // namespace ral
#endif  // RMAX_APPS_LIB_SERVICES_IPO_RECEIVER_SERVICE_H_
