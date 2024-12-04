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

#include <string>
#include <cstring>
#include <mutex>
#include <rivermax_api.h>

#include "rt_threads.h"

#include "rmax_base_service.h"
#include "api/rmax_apps_lib_api.h"

using namespace ral::services;
using namespace ral::lib::services;

namespace {
static const std::map<AllocatorTypeUI, AllocatorType> UI_ALLOCATOR_TYPE_MAP{
    {AllocatorTypeUI::Auto, AllocatorType::HugePageDefault},
    {AllocatorTypeUI::HugePageDefault, AllocatorType::HugePageDefault},
    {AllocatorTypeUI::Malloc, AllocatorType::Malloc},
    {AllocatorTypeUI::HugePage2MB, AllocatorType::HugePage2MB},
    {AllocatorTypeUI::HugePage512MB, AllocatorType::HugePage512MB},
    {AllocatorTypeUI::HugePage1GB, AllocatorType::HugePage1GB}};
}

class MemoryAllocatorFactory {
 public:
  static std::shared_ptr<MemoryAllocator> getAllocator(AllocatorType type,
                                                       std::shared_ptr<AppSettings> app_settings);

 private:
  static std::map<AllocatorType, std::shared_ptr<MemoryAllocator>> s_allocators;
  static std::mutex s_mutex;
};

std::map<AllocatorType, std::shared_ptr<MemoryAllocator>> MemoryAllocatorFactory::s_allocators;
std::mutex MemoryAllocatorFactory::s_mutex;

std::shared_ptr<MemoryAllocator> MemoryAllocatorFactory::getAllocator(
    AllocatorType type, std::shared_ptr<AppSettings> app_settings) {
  std::lock_guard<std::mutex> lock(s_mutex);

  auto it = s_allocators.find(type);
  if (it != s_allocators.end()) { return it->second; }

  std::shared_ptr<MemoryAllocator> allocator;
  allocator = MemoryAllocator::get_memory_allocator(type, app_settings);
  s_allocators[type] = allocator;
  return allocator;
}

RmaxBaseService::RmaxBaseService(const std::string& service_description)
    : m_obj_init_status(ReturnStatus::obj_init_failure),
      m_service_settings(nullptr),
      m_rmax_apps_lib(nullptr),
      m_signal_handler(nullptr),
      m_stats_reader(nullptr),
      m_service_description(service_description) {
  memset(&m_local_address, 0, sizeof(m_local_address));
}

RmaxBaseService::~RmaxBaseService() {
  if (m_obj_init_status != ReturnStatus::obj_init_success) { return; }

  for (auto& thread : m_threads) {
    if (thread.joinable()) { thread.join(); }
  }

  cleanup_rivermax_resources();
}

void RmaxBaseService::initialize_common_default_service_settings() {
  m_service_settings->destination_ip = DESTINATION_IP_DEFAULT;
  m_service_settings->destination_port = DESTINATION_PORT_DEFAULT;
  m_service_settings->num_of_threads = NUM_OF_THREADS_DEFAULT;
  m_service_settings->num_of_total_streams = NUM_OF_TOTAL_STREAMS_DEFAULT;
  m_service_settings->num_of_total_flows = NUM_OF_TOTAL_FLOWS_DEFAULT;
  m_service_settings->internal_thread_core = CPU_NONE;
  m_service_settings->app_threads_cores =
      std::vector<int>(m_service_settings->num_of_threads, CPU_NONE);
  m_service_settings->rate = {0, 0};
  m_service_settings->num_of_chunks = NUM_OF_CHUNKS_DEFAULT;
  m_service_settings->num_of_packets_in_chunk = NUM_OF_PACKETS_IN_CHUNK_DEFAULT;
  m_service_settings->packet_payload_size = PACKET_PAYLOAD_SIZE_DEFAULT;
  m_service_settings->packet_app_header_size = PACKET_APP_HEADER_SIZE_DEFAULT;
  m_service_settings->sleep_between_operations_us = SLEEP_BETWEEN_OPERATIONS_US_DEFAULT;
  m_service_settings->sleep_between_operations = false;
  m_service_settings->print_parameters = false;
  m_service_settings->use_checksum_header = false;
  m_service_settings->hw_queue_full_sleep_us = 0;
  m_service_settings->gpu_id = INVALID_GPU_ID;
  m_service_settings->allocator_type = AllocatorTypeUI::Auto;
  m_service_settings->statistics_reader_core = INVALID_CORE_NUMBER;
  m_service_settings->session_id_stats = UINT_MAX;
}

ReturnStatus RmaxBaseService::initialize_memory_allocators() {
  const auto alloc_type_iter = UI_ALLOCATOR_TYPE_MAP.find(m_service_settings->allocator_type);
  if (alloc_type_iter == UI_ALLOCATOR_TYPE_MAP.end()) {
    std::cerr << "Unknown UI allocator type "
              << static_cast<int>(m_service_settings->allocator_type) << std::endl;
    return ReturnStatus::failure;
  }
  AllocatorType allocator_type = alloc_type_iter->second;
  AllocatorType header_allocator_type;
  AllocatorType payload_allocator_type;
  if (m_service_settings->gpu_id != INVALID_GPU_ID) {
    header_allocator_type = allocator_type;
    payload_allocator_type = AllocatorType::Gpu;
  } else {
    header_allocator_type = allocator_type;
    payload_allocator_type = allocator_type;
  }
  m_header_allocator =
      MemoryAllocatorFactory::getAllocator(header_allocator_type, m_service_settings);
  if (m_header_allocator == nullptr) {
    std::cerr << "Failed to create header memory allocator" << std::endl;
    return ReturnStatus::failure;
  }
  m_payload_allocator =
      MemoryAllocatorFactory::getAllocator(payload_allocator_type, m_service_settings);
  if (m_payload_allocator == nullptr) {
    std::cerr << "Failed to create payload memory allocator" << std::endl;
    return ReturnStatus::failure;
  }
  return ReturnStatus::success;
}

ReturnStatus RmaxBaseService::initialize(const RmaxBaseServiceConfig& cfg) {
  if (cfg.rmax_apps_lib == nullptr) {
    m_rmax_apps_lib = std::make_shared<ral::lib::RmaxAppsLibFacade>();
  } else {
    m_rmax_apps_lib = cfg.rmax_apps_lib;
  }

  m_signal_handler = m_rmax_apps_lib->get_signal_handler(true);

  ReturnStatus rc = parse_configuration(cfg);
  if (rc == ReturnStatus::failure) { return rc; }

  if (m_service_settings == nullptr) {
    m_service_settings = std::make_shared<AppSettings>();
    initialize_common_default_service_settings();
  }

  post_parse_config_initialization();

  rc = initialize_memory_allocators();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to initialize memory allocators" << std::endl;
    m_obj_init_status = ReturnStatus::memory_allocation_failure;
    return m_obj_init_status;
  }

  rc = initialize_rivermax_resources();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to initialize Rivermax resources" << std::endl;
    return rc;
  }

  rc = initialize_connection_parameters();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to initialize application connection parameters" << std::endl;
    return rc;
  }

  rc = set_rivermax_clock();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to set Rivermax clock" << std::endl;
    return rc;
  }

  return ReturnStatus::obj_init_success;
}

ReturnStatus RmaxBaseService::set_rivermax_clock() {
  return ReturnStatus::success;
}

ReturnStatus RmaxBaseService::initialize_connection_parameters() {
  memset(&m_local_address, 0, sizeof(sockaddr_in));
  m_local_address.sin_family = AF_INET;
  int rc = inet_pton(AF_INET, m_service_settings->local_ip.c_str(), &m_local_address.sin_addr);
  if (rc != 1) {
    std::cerr << "Failed to parse local network address: " << m_service_settings->local_ip
              << std::endl;
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

void RmaxBaseService::run_stats_reader() {
  if (!is_run_stats_reader()) { return; }

  m_stats_reader.reset(new StatisticsReader());
  m_stats_reader->set_cpu_core_affinity(m_service_settings->statistics_reader_core);
  if (m_service_settings->session_id_stats != UINT_MAX) {
    std::cout << "Set presen session id: " << m_service_settings->session_id_stats << std::endl;
    m_stats_reader->set_session_id(m_service_settings->session_id_stats);
  }
  m_threads.push_back(std::thread(std::ref(*m_stats_reader)));
}
