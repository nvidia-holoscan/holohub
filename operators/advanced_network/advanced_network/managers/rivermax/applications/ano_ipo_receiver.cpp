/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ano_ipo_receiver.h"

namespace holoscan::advanced_network {

void ANOIPOReceiverSettings::init_default_values()
{
  AppSettings::init_default_values();
  app_memory_alloc = true;
  max_path_differential_us = 50000;
  is_extended_sequence_number = false;
  num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;
  min_packets_in_rx_chunk = 0;
  max_packets_in_rx_chunk = 0;
}

ReturnStatus ANOIPOReceiverSettingsValidator::validate(const std::shared_ptr<ANOIPOReceiverSettings>& settings) const
{
  if (settings->thread_settings.empty()) {
    std::cerr << "Must be at least one thread" << std::endl;
    return ReturnStatus::failure;
  }
  for (const auto& thread : settings->thread_settings) {
    if (thread.stream_network_settings.empty()) {
      std::cerr << "Must be at least one stream" << std::endl;
      return ReturnStatus::failure;
    }
    for (const auto& stream : thread.stream_network_settings) {
      if (stream.source_ips.empty()) {
        std::cerr << "Must be at least one source IP" << std::endl;
        return ReturnStatus::failure;
      }
      if (stream.destination_ips.size() != stream.source_ips.size()) {
        std::cerr << "Must be the same number of destination IPs as number of source IPs" << std::endl;
        return ReturnStatus::failure;
      }
      if (stream.local_ips.size() != stream.source_ips.size()) {
        std::cerr << "Must be the same number of NIC addresses as number of source IPs" << std::endl;
        return ReturnStatus::failure;
      }
      if (stream.destination_ports.size() != stream.source_ips.size()) {
        std::cerr << "Must be the same number of destination ports as number of source IPs" << std::endl;
        return ReturnStatus::failure;
      }
    }
  }
  return ReturnStatus::success;
}

ANOIPOReceiverApp::ANOIPOReceiverApp(std::shared_ptr<ISettingsBuilder<ANOIPOReceiverSettings>> settings_builder) :
  RmaxReceiverBaseApp(),
  m_settings_builder(std::move(settings_builder)),
  m_thread_streams_flows(),
  m_devices_ips()
{
}

ReturnStatus ANOIPOReceiverApp::initialize_connection_parameters()
{
  for (const auto& thread : m_ipo_receiver_settings->thread_settings) {
    for (const auto& stream : thread.stream_network_settings) {
      m_num_paths_per_stream = stream.local_ips.size();   // For now assume all streams have the same number of paths
      for (size_t i = 0; i < m_num_paths_per_stream; ++i) {
        in_addr device_address;
        rmx_device_iface device_iface;
        if (inet_pton(AF_INET, stream.local_ips[i].c_str(), &device_address) != 1) {
          std::cerr << "Failed to parse address of device " << stream.local_ips[i] << std::endl;
          return ReturnStatus::failure;
        }
        rmx_status status = rmx_retrieve_device_iface_ipv4(&device_iface, &device_address);
        if (status != RMX_OK) {
          std::cerr << "Failed to get device: " << stream.local_ips[i] << " with status: " << status << std::endl;
          return ReturnStatus::failure;
        }
        m_devices_ips.push_back(stream.local_ips[i]);
        m_device_interfaces.push_back(device_iface);
      }
    }
  }
  return ReturnStatus::success;
}

ReturnStatus ANOIPOReceiverApp::initialize_app_settings()
{
  if (m_settings_builder == nullptr) {
    std::cerr << "Settings builder is not initialized" << std::endl;
    return ReturnStatus::failure;
  }
  m_ipo_receiver_settings = std::make_shared<ANOIPOReceiverSettings>();
  ReturnStatus rc = m_settings_builder->build(m_ipo_receiver_settings);
  if (rc == ReturnStatus::success) {
    m_app_settings = m_ipo_receiver_settings;
    return ReturnStatus::success;
  }
  if (rc != ReturnStatus::success_cli_help) {
    std::cerr << "Failed to build settings" << std::endl;
  }
  m_obj_init_status = rc;
  return rc;
}

void ANOIPOReceiverApp::run_receiver_threads()
{
  run_threads(m_receivers);
}

void ANOIPOReceiverApp::configure_network_flows()
{
  int thread_index = 0;
  uint16_t source_port = 0;
  for (const auto& thread : m_ipo_receiver_settings->thread_settings) {
    std::vector<std::vector<ReceiveFlow>> streams;
    int internal_stream_index = 0;
    for (const auto& stream : thread.stream_network_settings) {
      size_t num_of_paths = stream.local_ips.size();
      std::vector<ReceiveFlow> paths;
      for (size_t i = 0; i < num_of_paths; ++i) {
        paths.emplace_back(stream.stream_id, stream.source_ips[i], source_port, stream.destination_ips[i], stream.destination_ports[i]);
      }
      streams.push_back(paths);
      m_stream_id_map[stream.stream_id] = std::make_pair(thread_index, internal_stream_index);
      internal_stream_index++;
    }
    m_thread_streams_flows.push_back(streams);
    thread_index++;
  }
}

void ANOIPOReceiverApp::initialize_receive_io_nodes()
{
  int number_of_threads = m_ipo_receiver_settings->thread_settings.size();
  size_t streams_offset = 0;
  for (int thread_index = 0; thread_index < number_of_threads; ++thread_index) {
    int number_of_streams = m_thread_streams_flows[thread_index].size();
    m_receivers.push_back(std::unique_ptr<ReceiverIONodeBase>(new IPOReceiverIONode(
      *m_app_settings,
      m_ipo_receiver_settings->max_path_differential_us,
      m_ipo_receiver_settings->is_extended_sequence_number,
      m_devices_ips,
      thread_index,
      m_ipo_receiver_settings->app_threads_cores[thread_index],
      *m_memory_utils)));
    static_cast<IPOReceiverIONode*>(m_receivers[thread_index].get())->initialize_streams(streams_offset, m_thread_streams_flows[thread_index]);
    streams_offset += number_of_streams;
  }
}

void ANOIPOReceiverApp::distribute_work_for_threads()
{
  m_streams_per_thread.reserve(m_ipo_receiver_settings->thread_settings.size());
  m_ipo_receiver_settings->num_of_total_streams = 0;
  int thread_index = 0;
  for (const auto& thread : m_ipo_receiver_settings->thread_settings) {
    m_streams_per_thread[thread_index] = thread.stream_network_settings.size();
    m_ipo_receiver_settings->num_of_total_streams += thread.stream_network_settings.size();
    thread_index++;
  }
}

ReturnStatus ANOIPOReceiverApp::find_internal_stream_index(size_t external_stream_index, size_t& thread_index, size_t& internal_stream_index) {
  if (m_stream_id_map.find(external_stream_index) == m_stream_id_map.end()) {
    std::cerr << "Invalid stream index " << external_stream_index << std::endl;
    return ReturnStatus::failure;
  }

  thread_index = m_stream_id_map[external_stream_index].first;
  internal_stream_index = m_stream_id_map[external_stream_index].second;

  return ReturnStatus::success;
}

}  // namespace holoscan::advanced_network
