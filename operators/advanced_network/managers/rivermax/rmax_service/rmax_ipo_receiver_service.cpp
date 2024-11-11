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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>

#include <rivermax_api.h>

#include "rt_threads.h"
#include "rmax_ipo_receiver_service.h"
#include "api/rmax_apps_lib_api.h"
#include "ipo_receiver_io_node.h"
#include "rmax_base_service.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::services::rmax_ipo_receiver;

bool RmaxIPOReceiverService::m_rivermax_lib_initialized = false;
std::mutex RmaxIPOReceiverService::m_rivermax_lib_mutex;

RmaxIPOReceiverService::RmaxIPOReceiverService(const RmaxIPOReceiverConfig& cfg)
    : RmaxBaseService(SERVICE_DESCRIPTION) {
  m_obj_init_status = initialize(cfg);
}
ReturnStatus RmaxIPOReceiverService::parse_configuration(const RmaxBaseServiceConfig& cfg) {
  const RmaxIPOReceiverConfig& ipo_service_cfg = static_cast<const RmaxIPOReceiverConfig&>(cfg);
  m_service_settings = ipo_service_cfg.app_settings;
  m_max_path_differential_us = ipo_service_cfg.max_path_differential_us;
  m_is_extended_sequence_number = ipo_service_cfg.is_extended_sequence_number;
  m_register_memory = ipo_service_cfg.register_memory;
  m_max_chunk_size = ipo_service_cfg.max_chunk_size;
  m_rx_stats_period_report_ms = ipo_service_cfg.rx_stats_period_report_ms;

  return ReturnStatus::success;
}

ReturnStatus RmaxIPOReceiverService::initialize_connection_parameters() {
  m_num_paths_per_stream = m_service_settings->source_ips.size();
  if (m_num_paths_per_stream == 0) {
    std::cerr << "Must be at least one source IP" << std::endl;
    return ReturnStatus::failure;
  }
  if (m_service_settings->destination_ips.size() != m_num_paths_per_stream) {
    std::cerr << "Must be the same number of destination multicast IPs as number of source IPs"
              << std::endl;
    return ReturnStatus::failure;
  }
  if (m_service_settings->local_ips.size() != m_num_paths_per_stream) {
    std::cerr << "Must be the same number of NIC addresses as number of source IPs" << std::endl;
    return ReturnStatus::failure;
  }
  if (m_service_settings->destination_ports.size() != m_num_paths_per_stream) {
    std::cerr << "Must be the same number of destination ports as number of source IPs"
              << std::endl;
    return ReturnStatus::failure;
  }

  m_device_ifaces.resize(m_num_paths_per_stream);
  for (size_t i = 0; i < m_num_paths_per_stream; ++i) {
    in_addr device_address;
    if (inet_pton(AF_INET, m_service_settings->local_ips[i].c_str(), &device_address) != 1) {
      std::cerr << "Failed to parse address of device " << m_service_settings->local_ips[i]
                << std::endl;
      return ReturnStatus::failure;
    }
    rmx_status status = rmx_retrieve_device_iface_ipv4(&m_device_ifaces[i], &device_address);
    if (status != RMX_OK) {
      std::cerr << "Failed to get device: " << m_service_settings->local_ips[i]
                << " with status: " << status << std::endl;
      return ReturnStatus::failure;
    }
  }

  if (m_register_memory && m_service_settings->packet_app_header_size == 0) {
    std::cerr << "Memory registration is supported only in header-data split mode!" << std::endl;
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

ReturnStatus RmaxIPOReceiverService::run(IRmaxServicesSynchronizer* sync_obj) {
  if (m_obj_init_status != ReturnStatus::obj_init_success) { return m_obj_init_status; }

  try {
    distribute_work_for_threads();
    configure_network_flows();
    initialize_receive_io_nodes();

    if (m_receivers.empty()) {
      std::cerr << "No receivers initialized" << std::endl;
      return ReturnStatus::failure;
    }

    ReturnStatus rc = allocate_service_memory();
    if (rc == ReturnStatus::failure) {
      std::cerr << "Failed to allocate the memory required for the service" << std::endl;
      return rc;
    }
    distribute_memory_for_receivers();

    m_service_running = true;

    if (sync_obj) { sync_obj->wait_for_start(); }

    run_threads(m_receivers);

    m_service_running = false;

    unregister_service_memory();
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

ReturnStatus RmaxIPOReceiverService::initialize_rivermax_resources() {
  std::lock_guard<std::mutex> lock(m_rivermax_lib_mutex);

  if (RmaxIPOReceiverService::m_rivermax_lib_initialized) { return ReturnStatus::success; }

  rt_set_realtime_class();

  ReturnStatus ret = m_rmax_apps_lib->initialize_rivermax(m_service_settings->internal_thread_core);

  if (ret == ReturnStatus::success) { RmaxIPOReceiverService::m_rivermax_lib_initialized = true; }

  return ret;
}

void RmaxIPOReceiverService::configure_network_flows() {
  std::vector<std::string> ip_prefix_str;
  std::vector<int> ip_last_octet;
  uint16_t src_port = 0;

  assert(m_num_paths_per_stream > 0);
  ip_prefix_str.resize(m_num_paths_per_stream);
  ip_last_octet.resize(m_num_paths_per_stream);
  for (size_t i = 0; i < m_num_paths_per_stream; ++i) {
    auto ip_vec = CLI::detail::split(m_service_settings->destination_ips[i], '.');
    if (ip_vec.size() != 4) {
      std::cerr << "Invalid IP address format: " << m_service_settings->destination_ips[i]
                << std::endl;
      return;
    }
    ip_prefix_str[i] = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    ip_last_octet[i] = std::stoi(ip_vec[3]);
  }

  m_flows.reserve(m_service_settings->num_of_total_streams);
  size_t id = 0;
  for (size_t flow_index = 0; flow_index < m_service_settings->num_of_total_streams; ++flow_index) {
    std::vector<FourTupleFlow> paths;
    for (size_t i = 0; i < m_num_paths_per_stream; ++i) {
      std::ostringstream ip;
      ip << ip_prefix_str[i]
         << (ip_last_octet[i] + flow_index * m_num_paths_per_stream) % IP_OCTET_LEN;
      paths.emplace_back(id++,
                         m_service_settings->source_ips[i],
                         src_port,
                         ip.str(),
                         m_service_settings->destination_ports[i]);
    }
    m_flows.push_back(paths);
  }
}

void RmaxIPOReceiverService::distribute_work_for_threads() {
  m_streams_per_thread.reserve(m_service_settings->num_of_threads);
  for (int stream = 0; stream < m_service_settings->num_of_total_streams; stream++) {
    m_streams_per_thread[stream % m_service_settings->num_of_threads]++;
  }
}

void RmaxIPOReceiverService::initialize_receive_io_nodes() {
  size_t streams_offset = 0;
  for (size_t rx_idx = 0; rx_idx < m_service_settings->num_of_threads; rx_idx++) {
    int recv_cpu_core =
        m_service_settings
            ->app_threads_cores[rx_idx % m_service_settings->app_threads_cores.size()];

    auto flows = std::vector<std::vector<FourTupleFlow>>(
        m_flows.begin() + streams_offset,
        m_flows.begin() + streams_offset + m_streams_per_thread[rx_idx]);
    m_receivers.push_back(
        std::unique_ptr<IPOReceiverIONode>(new IPOReceiverIONode(*m_service_settings,
                                                                 m_max_path_differential_us,
                                                                 m_is_extended_sequence_number,
                                                                 m_max_chunk_size,
                                                                 m_service_settings->local_ips,
                                                                 rx_idx,
                                                                 recv_cpu_core,
                                                                 m_chunk_consumer)));

    m_receivers[rx_idx]->initialize_streams(streams_offset, flows);
    m_receivers[rx_idx]->print_statistics_settings((m_rx_stats_period_report_ms > 0 ? true : false),
                                                   m_rx_stats_period_report_ms);
    streams_offset += m_streams_per_thread[rx_idx];
  }
}

bool RmaxIPOReceiverService::allocate_and_align(size_t header_size, size_t payload_size,
                                                byte_t*& header, byte_t*& payload) {
  header = payload = nullptr;
  if (header_size) {
    header = static_cast<byte_t*>(
        m_header_allocator->allocate_aligned(header_size, m_header_allocator->get_page_size()));
  }
  payload = static_cast<byte_t*>(
      m_payload_allocator->allocate_aligned(payload_size, m_payload_allocator->get_page_size()));
  return payload && (header_size == 0 || header);
}

ReturnStatus RmaxIPOReceiverService::allocate_service_memory() {
  size_t hdr_mem_size;
  size_t pld_mem_size;
  ReturnStatus rc = get_total_ipo_streams_memory_size(hdr_mem_size, pld_mem_size);
  if (rc != ReturnStatus::success) { return rc; }

  bool alloc_successful =
      allocate_and_align(hdr_mem_size, pld_mem_size, m_header_buffer, m_payload_buffer);

  if (alloc_successful) {
    std::cout << "Allocated " << hdr_mem_size << " bytes for header"
              << " at address " << static_cast<void*>(m_header_buffer) << " and " << pld_mem_size
              << " bytes for payload"
              << " at address " << static_cast<void*>(m_payload_buffer) << std::endl;
  } else {
    std::cerr << "Failed to allocate memory" << std::endl;
    return ReturnStatus::failure;
  }

  m_header_mem_regions.resize(m_num_paths_per_stream);
  m_payload_mem_regions.resize(m_num_paths_per_stream);

  if (!m_register_memory) { return ReturnStatus::success; }

  for (size_t i = 0; i < m_num_paths_per_stream; ++i) {
    m_header_mem_regions[i].addr = m_header_buffer;
    m_header_mem_regions[i].length = hdr_mem_size;
    m_header_mem_regions[i].mkey = 0;
    if (hdr_mem_size) {
      rmx_mem_reg_params mem_registry;
      rmx_init_mem_registry(&mem_registry, &m_device_ifaces[i]);
      rmx_status status = rmx_register_memory(&m_header_mem_regions[i], &mem_registry);
      if (status != RMX_OK) {
        std::cerr << "Failed to register header memory on device "
                  << m_service_settings->local_ips[i] << " with status: " << status << std::endl;
        return ReturnStatus::failure;
      }
    }
  }
  for (size_t i = 0; i < m_num_paths_per_stream; ++i) {
    rmx_mem_reg_params mem_registry;
    rmx_init_mem_registry(&mem_registry, &m_device_ifaces[i]);
    m_payload_mem_regions[i].addr = m_payload_buffer;
    m_payload_mem_regions[i].length = pld_mem_size;
    rmx_status status = rmx_register_memory(&m_payload_mem_regions[i], &mem_registry);
    if (status != RMX_OK) {
      std::cerr << "Failed to register payload memory on device "
                << m_service_settings->local_ips[i] << " with status: " << status << std::endl;
      return ReturnStatus::failure;
    }
  }

  return ReturnStatus::success;
}

void RmaxIPOReceiverService::unregister_service_memory() {
  if (!m_register_memory) { return; }

  if (m_header_buffer) {
    for (size_t i = 0; i < m_header_mem_regions.size(); ++i) {
      rmx_status status = rmx_deregister_memory(&m_header_mem_regions[i], &m_device_ifaces[i]);
      if (status != RMX_OK) {
        std::cerr << "Failed to deregister header memory on device "
                  << m_service_settings->local_ips[i] << " with status: " << status << std::endl;
      }
    }
  }
  for (size_t i = 0; i < m_payload_mem_regions.size(); ++i) {
    rmx_status status = rmx_deregister_memory(&m_payload_mem_regions[i], &m_device_ifaces[i]);
    if (status != RMX_OK) {
      std::cerr << "Failed to deregister payload memory on device "
                << m_service_settings->local_ips[i] << " with status: " << status << std::endl;
    }
  }
}

ReturnStatus RmaxIPOReceiverService::get_total_ipo_streams_memory_size(size_t& hdr_mem_size,
                                                                       size_t& pld_mem_size) {
  hdr_mem_size = 0;
  pld_mem_size = 0;

  if (!m_header_allocator || !m_payload_allocator) {
    std::cerr << "Memory allocators are not initialized" << std::endl;
    return ReturnStatus::failure;
  }

  for (const auto& receiver : m_receivers) {
    for (const auto& stream : receiver->get_streams()) {
      size_t hdr_buf_size, pld_buf_size;
      ReturnStatus rc = stream->query_buffer_size(hdr_buf_size, pld_buf_size);
      if (rc != ReturnStatus::success) {
        std::cerr << "Failed to query buffer size for stream " << stream->get_id()
                  << " of receiver " << receiver->get_index() << std::endl;
        return rc;
      }
      hdr_mem_size += m_header_allocator->align_length(hdr_buf_size);
      pld_mem_size += m_payload_allocator->align_length(pld_buf_size);
    }
  }

  std::cout << "Service requires " << hdr_mem_size << " bytes of header memory and " << pld_mem_size
            << " bytes of payload memory" << std::endl;

  return ReturnStatus::success;
}

void RmaxIPOReceiverService::distribute_memory_for_receivers() {
  byte_t* hdr_ptr = m_header_buffer;
  byte_t* pld_ptr = m_payload_buffer;

  for (auto& receiver : m_receivers) {
    for (auto& stream : receiver->get_streams()) {
      size_t hdr, pld;

      stream->set_buffers(hdr_ptr, pld_ptr);
      if (m_register_memory) {
        stream->set_memory_keys(m_header_mem_regions, m_payload_mem_regions);
      }
      stream->query_buffer_size(hdr, pld);

      if (hdr_ptr) { hdr_ptr += m_header_allocator->align_length(hdr); }
      pld_ptr += m_payload_allocator->align_length(pld);
    }
  }
}

bool RmaxIPOReceiverService::set_chunk_consumer(IIPOChunkConsumer* chunk_consumer) {
  if (is_alive()) {
    std::cerr << "Cannot set chunk consumer while service is running" << std::endl;
    return false;
  }
  m_chunk_consumer = chunk_consumer;
  return true;
}

std::pair<std::vector<IPORXStatistics>, std::vector<std::vector<IPOPathStatistics>>>
RmaxIPOReceiverService::get_streams_statistics() const {
  std::vector<IPORXStatistics> streams_stats;
  std::vector<std::vector<IPOPathStatistics>> streams_path_stats;

  // Populate streams_stats and streams_path_stats
  for (const auto& receiver : m_receivers) {
    auto [stream_stats, stream_path_stats] = receiver->get_streams_statistics();
    streams_stats.insert(streams_stats.end(), stream_stats.begin(), stream_stats.end());
    streams_path_stats.insert(
        streams_path_stats.end(), stream_path_stats.begin(), stream_path_stats.end());
  }

  return {streams_stats, streams_path_stats};
}
