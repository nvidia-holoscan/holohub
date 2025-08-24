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

#include "ano_rtp_receiver.h"

namespace holoscan::advanced_network {

void ANORTPReceiverSettings::init_default_values()
{
  AppSettings::init_default_values();
  app_memory_alloc = true;
  num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;
  is_extended_sequence_number = false;
}

ReturnStatus ANORTPReceiverSettingsValidator::validate(const std::shared_ptr<ANORTPReceiverSettings>& settings) const
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
      ReturnStatus rc = ValidatorUtils::validate_ip4_address(stream.source_ip);
      if (rc != ReturnStatus::success) {
        return rc;
      }
      rc = ValidatorUtils::validate_ip4_address(stream.local_ip);
      if (rc != ReturnStatus::success) {
          return rc;
      }
      rc = ValidatorUtils::validate_ip4_address(stream.destination_ip);
      if (rc != ReturnStatus::success) {
          return rc;
      }
      rc = ValidatorUtils::validate_ip4_port(stream.destination_port);
      if (rc != ReturnStatus::success) {
          return rc;
      }
    }
  }
  return ReturnStatus::success;
}

ANORTPReceiverApp::ANORTPReceiverApp(std::shared_ptr<ISettingsBuilder<ANORTPReceiverSettings>> settings_builder) :
  RmaxReceiverBaseApp(),
  m_settings_builder(std::move(settings_builder)),
  m_threads_streams(),
  m_devices_ips()
{
  m_num_paths_per_stream = 1;  // RTP receiver supports only single path per stream
}

ReturnStatus ANORTPReceiverApp::initialize_connection_parameters()
{
  for (const auto& thread : m_rtp_receiver_settings->thread_settings) {
    for (const auto& stream : thread.stream_network_settings) {
      in_addr device_address;
      rmx_device_iface device_iface;
      if (inet_pton(AF_INET, stream.local_ip.c_str(), &device_address) != 1) {
        std::cerr << "Failed to parse address of device " << stream.local_ip << std::endl;
        return ReturnStatus::failure;
      }
      rmx_status status = rmx_retrieve_device_iface_ipv4(&device_iface, &device_address);
      if (status != RMX_OK) {
        std::cerr << "Failed to get device: " << stream.local_ip << " with status: " << status << std::endl;
        return ReturnStatus::failure;
      }
      m_devices_ips.push_back(stream.local_ip);
      m_device_interfaces.push_back(device_iface);
      return ReturnStatus::success; // RTP receiver application currently supports transmission on a single NIC only
    }
  }
  return ReturnStatus::success;
}

ReturnStatus ANORTPReceiverApp::initialize_app_settings()
{
  if (m_settings_builder == nullptr) {
    std::cerr << "Settings builder is not initialized" << std::endl;
    return ReturnStatus::failure;
  }
  m_rtp_receiver_settings = std::make_shared<ANORTPReceiverSettings>();
  ReturnStatus rc = m_settings_builder->build(m_rtp_receiver_settings);
  if (rc == ReturnStatus::success) {
    m_app_settings = m_rtp_receiver_settings;
    return ReturnStatus::success;
  }
  if (rc != ReturnStatus::success_cli_help) {
    std::cerr << "Failed to build settings" << std::endl;
  }
  m_obj_init_status = rc;
  return rc;
}

void ANORTPReceiverApp::run_receiver_threads()
{
    run_threads(m_receivers);
}

void ANORTPReceiverApp::configure_network_flows()
{
  uint16_t source_port = 0;
  int thread_index = 0;
  for (const auto& thread : m_rtp_receiver_settings->thread_settings) {
    std::vector<ReceiveFlow> streams;
    int internal_stream_index = 0;
    for (const auto& stream : thread.stream_network_settings) {
      ReceiveFlow flow(stream.stream_id, stream.source_ip, source_port, stream.destination_ip, stream.destination_port);
      streams.push_back(flow);
      m_stream_id_map[stream.stream_id] = std::make_pair(thread_index, internal_stream_index);
      internal_stream_index++;
    }
    m_threads_streams.push_back(streams);
    thread_index++;
  }
}

void ANORTPReceiverApp::initialize_receive_io_nodes()
{
  m_rtp_receiver_settings->num_of_threads = m_rtp_receiver_settings->thread_settings.size();
  size_t streams_offset = 0;
  for (int thread_index = 0; thread_index < m_rtp_receiver_settings->num_of_threads; ++thread_index) {
    m_receivers.push_back(std::unique_ptr<ReceiverIONodeBase>(new RTPReceiverIONode(
      *m_app_settings,
      m_rtp_receiver_settings->is_extended_sequence_number,
      { m_devices_ips[0] },  // Pass a single device IP
      thread_index,
      m_rtp_receiver_settings->app_threads_cores[thread_index],
      *m_memory_utils)));
    static_cast<RTPReceiverIONode*>(m_receivers[thread_index].get())->initialize_streams(streams_offset, m_threads_streams[thread_index]);
    streams_offset += m_streams_per_thread[thread_index];
  }
}

void ANORTPReceiverApp::distribute_work_for_threads()
{
  m_streams_per_thread.reserve(m_rtp_receiver_settings->thread_settings.size());
  m_rtp_receiver_settings->num_of_total_streams = 0;
  int thread_index = 0;
  for (const auto& thread : m_rtp_receiver_settings->thread_settings) {
    m_streams_per_thread[thread_index] = thread.stream_network_settings.size();
    m_rtp_receiver_settings->num_of_total_streams += thread.stream_network_settings.size();
    thread_index++;
  }
}

ReturnStatus ANORTPReceiverApp::find_internal_stream_index(size_t external_stream_index, size_t& thread_index, size_t& internal_stream_index) {
  if (m_stream_id_map.find(external_stream_index) == m_stream_id_map.end()) {
    std::cerr << "Invalid stream index " << external_stream_index << std::endl;
    return ReturnStatus::failure;
  }

  thread_index = m_stream_id_map[external_stream_index].first;
  internal_stream_index = m_stream_id_map[external_stream_index].second;

  return ReturnStatus::success;
}
}  // namespace holoscan::advanced_network
